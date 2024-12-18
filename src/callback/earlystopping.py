import numpy as np
import os
import torch
# from utils import logger
import logging
logger = logging.getLogger("train")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, ckpt_dir, patience=7, min_delta=0, mode='max', max_to_save=1):
        """
        Args:
            ckpt_dir : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            max_to_save: 最多保存最后几次ckpts
        """
        self.ckpt_dir = ckpt_dir
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stop_training = False
        
        assert mode in ['min','max']
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        
        self.model_list = []    # 保存最后五次ckpt
        self.max_to_save = max_to_save

    def update(self, step, current_score, model, test_score=None):
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            logger.info(f"step {step}: improve from {self.best_score:.4f} to {current_score:.4f}")
            self.best_score = current_score
            self.wait = 0

            '''Saves model when score improve.'''
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            # 超过最大保存次数
            if len(self.model_list) == self.max_to_save:
                os.remove(os.path.join(self.ckpt_dir, f"{self.model_list[0]}"))
                self.model_list.pop(0)

            if test_score is None:
                model_name = f"model_{step}_{current_score*100:.2f}.tar"
            else:
                model_name = f"model_{step}_{current_score*100:.2f}_{test_score*100:.2f}.tar"

            self.model_list.append(model_name)
            torch.save({'state_dict': model_to_save.state_dict()}, os.path.join(self.ckpt_dir, model_name))
            
            logger.info(f'Saved model to {self.ckpt_dir}/{model_name}')
        else:
            self.wait += 1
            logger.info(f'EarlyStopping wait: {self.wait} out of {self.patience}')
            if self.wait >= self.patience:
                logger.info(f"current score: {current_score}, current step: {step}")
                logger.info(f"{self.patience} steps with no improvement after which training will be stopped")
                self.stop_training = True



class WMAEarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, ckpt_dir, patience=7, min_delta=0, mode='max', max_to_save=1,
                 window_size=5, decay=0.9):
        """
        Args:
            ckpt_dir : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            max_to_save: 最多保存最后几次ckpts
        """
        self.ckpt_dir = ckpt_dir
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stop_training = False
        
        assert mode in ['min','max']
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater

        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        
        self.model_list = []    # 保存最后五次ckpt
        self.max_to_save = max_to_save

        self.history = []
        self.window_size = window_size
        self.decay = decay

    def compute_weighted_average(self, current_score):
        """
        计算加权移动平均（WMA），较近的轮次赋予更高权重
        :return: 加权平均值
        """
        self.history.append(current_score)

        if len(self.history) < self.window_size:
            # 还未收集足够的历史数据，无法进行加权平均
            weighted_avg = np.mean(self.history)
        else:
            # 计算加权移动平均
            scores = self.history[-self.window_size:]
            weights = np.array([self.decay**(len(scores)-i-1) for i in range(len(scores))])
            weighted_avg = np.dot(weights, scores) / np.sum(weights)
        return weighted_avg


    def update(self, step, current_score, model, test_score=None):
        weighted_avg = self.compute_weighted_average(current_score)
        if current_score >= weighted_avg:
            self.wait = 0
            # TODO: 移动平均分数 保存 还是最好分数保存
            if current_score > self.best_score:
                self.best_score = current_score
                logger.info(f"step {step}: improve from {self.best_score:.4f} to {current_score:.4f}")

                '''Saves model when score improve.'''
                if not os.path.exists(self.ckpt_dir):
                    os.makedirs(self.ckpt_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                # 超过最大保存次数
                if len(self.model_list) == self.max_to_save:
                    os.remove(os.path.join(self.ckpt_dir, f"{self.model_list[0]}"))
                    self.model_list.pop(0)

                if test_score is None:
                    model_name = f"model_{step}_{current_score*100:.2f}.tar"
                else:
                    model_name = f"model_{step}_{current_score*100:.2f}_{test_score*100:.2f}.tar"

                self.model_list.append(model_name)
                torch.save({'state_dict': model_to_save.state_dict()}, os.path.join(self.ckpt_dir, model_name))
                
                logger.info(f'Saved model to {self.ckpt_dir}/{model_name}')
        else:
            self.wait += 1
            logger.info(f'EarlyStopping wait: {self.wait} out of {self.patience}')
            if self.wait >= self.patience:
                logger.info(f"current score: {current_score}, current step: {step}")
                logger.info(f"{self.patience} steps with no improvement after which training will be stopped")
                self.stop_training = True
