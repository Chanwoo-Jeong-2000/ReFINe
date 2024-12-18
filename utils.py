import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=50, verbose=False, delta=0, path=None):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_epoch = None
        self.best_metrics = None
        self.path = path

    def __call__(self, epoch, metrics, model=None):
        score = metrics[-1][1] # recall@20

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_metrics = metrics
            if model is not None:
                print('Save:', self.path)
                torch.save(model.state_dict(), self.path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_metrics = metrics
            self.counter = 0
            if model is not None:
                print('Save:', self.path)
                torch.save(model.state_dict(), self.path)