from torch.optim import AdamW
from transformers import get_scheduler, Adafactor
from tqdm import trange, tqdm
from torch.nn.utils import clip_grad_norm_
import pandas as pd
import torch
import os
from utils.focal_loss import MultiFocalLoss, FocalLoss
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.tools import get_rank


class Trainer:
    def __init__(self, model, tokenizer, args, logger, early_stopping=None):
        self.device = 'cuda'
        if args.use_ddp:
            self.device = dist.get_rank() % torch.cuda.device_count()
            self.model = model.to(self.device)

            local_rank = int(os.environ['LOCAL_RANK'])
            self.model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.args = args
        self.early_stopping = early_stopping
        self.logger = logger
        self.set_optimizer()

    def set_optimizer(self):
        optimizer_name = self.args.optimizer.lower()
        if optimizer_name == "adamw":
            no_decay = ["bias", "LayerNorm.weight"]
            # it's always good practice to set no decay to biase and LayerNorm parameters
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if 'bert' in n and not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay, "lr": self.args.bert_learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if 'bert' in n and any(nd in n for nd in no_decay)],
                    "weight_decay": 0., "lr": self.args.bert_learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if not 'bert' in n and not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if not 'bert' in n and any(nd in n for nd in no_decay)],
                    "weight_decay": 0., "lr": self.args.learning_rate
                }
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        elif optimizer_name == "adafactor":
            self.optimizer = Adafactor(
                self.model.parameters(), scale_parameter=False, relative_step=False,
                warmup_init=False, lr=self.args.learning_rate
            )
        else:
            raise NameError("Unknown optimizer")
    
    def set_scheduler(self, train_dataloader):
        num_gpus = dist.get_world_size() if dist.is_initialized() else 1
        total_batch_size = self.args.train_batch_size * self.args.grad_accum_steps * num_gpus
        total_steps = int(len(train_dataloader.dataset) / total_batch_size * self.args.epochs)
        warmup_steps = int(total_steps * self.args.warmup_proportion)
        self.scheduler = get_scheduler(
            "linear",
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def print_basic_info(self, train_dataloader, dev_dataloader):
        self.logger.info("Start Training...")
        num_gpus = dist.get_world_size() if dist.is_initialized() else 1
        total_batch_size = self.args.train_batch_size * self.args.grad_accum_steps * num_gpus
        self.logger.info(f"Train Examples Size:{len(train_dataloader.dataset)}")
        self.logger.info(f"  Dev Examples Size:{len(dev_dataloader.dataset)}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.grad_accum_steps}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Num Epochs = {self.args.epochs}")
        self.logger.info(f"  1 epoch has {int(len(train_dataloader.dataset)/total_batch_size)} effective steps")
        total_steps = int(len(train_dataloader.dataset) / total_batch_size * self.args.epochs)
        self.logger.info(f"  Total optimization steps = {total_steps}")
    
    def model_update(self):
        # self.accelerator.clip_grad_value_(self.model.parameters(), self.args.grad_clip)
        clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.model.zero_grad()
    
    def sum_loss(self, loss_dict, new_loss_dict):
        for key, value in new_loss_dict.items():
            if key in loss_dict.keys():
                loss_dict[key] += value.item() if hasattr(value, 'item') else value
            else:
                loss_dict[key] = value.item() if hasattr(value, 'item') else value
        return loss_dict

    def train(self, train_dataloader, dev_dataloader, test_dataloader=None):
        self.set_scheduler(train_dataloader)
        self.print_basic_info(train_dataloader, dev_dataloader)

        step = 0
        effective_step = 0
        train_loss_dict = {}
        
        for epoch in trange(1, self.args.epochs + 1, desc='Epoch'):
            self.model.train()

            if self.args.use_ddp:
                train_dataloader.sampler.set_epoch(epoch)
            for batch in tqdm(train_dataloader, desc='Training'):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                _, loss = self.model(**batch)

                train_loss_dict = self.sum_loss(train_loss_dict, {'loss': loss})
                # if len(self.args.n_gpu) >= 2:
                #     loss = loss.mean()
                if self.args.grad_accum_steps > 1:
                    loss = loss / self.args.grad_accum_steps
                loss.backward()
                step += 1
                if step % self.args.grad_accum_steps == 0:
                    self.model_update()
                    effective_step += 1
                    # 打印训练的loss信息
                    if effective_step % self.args.logging_steps == 0 and get_rank() == 0:
                        for key, value in train_loss_dict.items():
                            train_loss_dict[key] = round(value / self.args.logging_steps / self.args.grad_accum_steps, 6)
                        self.logger.info(f"At epoch {epoch} and train step {effective_step} Train Loss: {train_loss_dict}")
                        train_loss_dict = {}
                    
                    # 以step为单位验证
                    if self.args.eval_steps != -1 and effective_step % self.args.eval_steps == 0:
                        if get_rank() == 0:
                            self.eval_and_update_model(epoch, effective_step, dev_dataloader, test_dataloader)
                            # self.early_stopping.stop_training = True
                        # if self.args.use_ddp:
                        #     print(f"a, {get_rank()}: {self.early_stopping.stop_training}")
                        #     dist.barrier()  # 等rank 0评估结束
                        #     self.synchronize_stop_training()    # 同步 stop_training 状态
                        #     print(f"b, {get_rank()}: {self.early_stopping.stop_training}")
                        #     dist.barrier()  #确保所有进程都同步好
                        # print(f"c, {get_rank()}: {self.early_stopping.stop_training}")
                        
                        if self.early_stopping and self.early_stopping.stop_training:
                            break

            # 以epoch为单位验证
            if self.args.eval_steps == -1 and get_rank() == 0:
                self.eval_and_update_model(epoch, effective_step, dev_dataloader, test_dataloader)

            if self.early_stopping and self.early_stopping.stop_training:
                break

    def eval_and_update_model(self, epoch, step, dev_dataloader, test_dataloader):
        self.logger.info(f"At epoch {epoch} and train steps {step}")
        self.logger.info("dev results:")
        dev_result = self.eval(dev_dataloader, do_save=False, show_reslut=False)

        test_score = None
        # if test_dataloader is not None:
        if test_dataloader is not None and dev_result['macro_f1'] > 0.52:    # 减少测试集的评估次数
            self.logger.info("test results:")
            test_result = self.eval(test_dataloader, do_save=False, show_reslut=False)
            test_score = test_result['macro_f1']

        if self.early_stopping:
            self.early_stopping.update(step, dev_result['macro_f1'], self.model, test_score)
    

    def synchronize_stop_training(self):
        # 创建一个张量来保存每个进程的stop_training状态
        local_rank = int(os.environ['LOCAL_RANK'])
        stop_tensor = torch.tensor([self.early_stopping.stop_training], dtype=torch.int).to(local_rank)

        # 使用 all_reduce 来同步所有进程的 stop_training 状态
        dist.all_reduce(stop_tensor, op=dist.ReduceOp.SUM)

        # 如果所有进程的stop_training值都为1，则全体停止训练
        if stop_tensor.item() > 0:
            self.early_stopping.stop_training = True
        else:
            self.early_stopping.stop_training = False



    def eval(self, data_loader, do_save=False, show_reslut=True):
        self.model.eval()
        self.logger.info(f"Eval ... ")
        id2label = data_loader.dataset.id2label

        with torch.no_grad():
            loss_dict = {}
            all_preds, all_labels = [], []
            all_input_ids = []
            for i, batch in tqdm(enumerate(data_loader), desc=f'eval'):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                (logits, labels), loss = self.model(**batch)
                loss_dict = self.sum_loss(loss_dict, {'loss': loss})

                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().detach())
                all_labels.append(labels.cpu().detach())

                if do_save:
                    all_input_ids.append(batch['input_ids'].cpu().detach())
                if show_reslut and i == 0:
                    # 浅浅展示一下分类结果
                    input_ids_flat = batch['input_ids'].view(-1, self.args.max_seq_len)
                    bsz, max_turns, _ = batch['input_ids'].shape
                    valid_mask = torch.zeros(bsz * max_turns, dtype=torch.bool).to(self.device)
                    for i in range(bsz):
                        start_idx = i * max_turns
                        valid_mask[start_idx: start_idx + batch['num_turns'][i].item()] = True

                    valid_input_ids = input_ids_flat[valid_mask]  # [有效样本数, max_seq_len]
                    self.show_results(all_preds[0], all_labels[0], valid_input_ids, id2label)
            
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 计算loss
            for key, value in loss_dict.items():
                loss_dict[key] = round(value / len(data_loader), 6)
            self.logger.info(f"eval loss = {loss_dict}")

            result = loss_dict
            
            # 计算指标
            metric_results = self.cal_metric(all_preds, all_labels)
            for key, value in metric_results.items():
                result[key] = value
                self.logger.info(f'{key}: {value*100:.2f}')
            
            # 导出结果，仅用于测试阶段
            if do_save:
                all_input_ids = torch.cat(all_input_ids, dim=0)
                utterances = self.tokenizer.batch_decode(all_input_ids, skip_special_tokens=True)
                
                pred_classes = [id2label[id.item()] for id in all_preds]
                true_classes = [id2label[id.item()] for id in all_labels]

                result_to_save = {'utterances': utterances, 'ground_labels': true_classes, 'pred_labels': pred_classes}

                df = pd.DataFrame(result_to_save)
                model_id = self.args.ckpt_dir.split("/")[-2]
                df.to_csv(f'{self.args.test_results_dir}/{model_id}_results.csv')
                self.logger.info(f"Saved results to {self.args.test_results_dir}/{model_id}_results.csv")

                with open(f'{self.args.test_results_dir}/{model_id}_metric.txt', 'w') as f:
                    for key, value in result.items():
                        f.write(f'{key}: {value}\n')
                        
                self.logger.info(f"Saved metric to {self.args.test_results_dir}/{model_id}_metric.txt")

        self.model.train()
        return result
    

    def cal_metric(self, preds, labels):
        micro_p = precision_score(y_true=labels, y_pred=preds, average='micro')
        micro_r = recall_score(y_true=labels, y_pred=preds, average='micro')
        micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')

        macro_p = precision_score(y_true=labels, y_pred=preds, average='macro')
        macro_r = recall_score(y_true=labels, y_pred=preds, average='macro')
        macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
        
        # from sklearn.metrics import classification_report
        # from sklearn.metrics import confusion_matrix
        # id2text = self.taxonomy.get_id2label()
        # cr = classification_report(y_true=labels, y_pred=preds, target_names=[id2text[id] for id in range(len(id2text))])
        # cm = confusion_matrix(y_true=[id2text[id.item()] for id in labels], y_pred=[id2text[id.item()] for id in preds], labels=[id2text[id] for id in range(len(id2text))])
        # self.logger.info(f"classification report:\n{cr}")
        # self.logger.info(f"confusion matrix\n{cm}")

        return {
            'micro_p': micro_p,
            'micro_r': micro_r,
            'micro_f1': micro_f1,
            'macro_p': macro_p,
            'macro_r': macro_r,
            'macro_f1': macro_f1
        }

    def show_results(self, preds, labels, input_ids, id2label):
        """
        preds、labels: [bsz, ]
        input_ids: [bsz, max_seq_len]
        """
        pred_classes = [id2label[id.item()] for id in preds]
        true_classes = [id2label[id.item()] for id in labels]

        utterances = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        for pred_class, true_class, utterance in zip(pred_classes, true_classes, utterances):
            s = "\n" + "\n".join([
                "utterance:\t" + utterance,
                "pred_class:\t" + pred_class,
                "true_class\t" + true_class
            ])
            self.logger.info(s)
