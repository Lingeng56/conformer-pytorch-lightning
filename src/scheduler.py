from torch.optim.lr_scheduler import _LRScheduler


class TransformerLR(_LRScheduler):

    def __init__(self, optimizer, peak_lr, warmup, last_epoch=-1, verbose=False):
        self.peak_lr = peak_lr
        self.warmup = warmup
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count == 0:
            curr_lr = 0.0
        else:
            curr_lr = self._step_count * self.peak_lr / self.warmup
        values = [curr_lr for _ in self.optimizer.param_groups]
        return values
