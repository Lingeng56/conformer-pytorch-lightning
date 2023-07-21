from torch.optim.lr_scheduler import _LRScheduler


class TransformerLR(_LRScheduler):

    def __init__(self, optimizer, d_model, warmup, multiplier, last_epoch=-1, verbose=False):
        self.d_model = d_model
        self.warmup = warmup
        self.multiplier = multiplier
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        self._step_count = max(1, self._step_count)
        curr_lr = self.multiplier * (self.d_model ** -0.5) * min(self._step_count ** (-0.5), self._step_count * (self.warmup ** (-1.5)))
        values = [curr_lr for _ in  self.optimizer.param_groups]
        return values