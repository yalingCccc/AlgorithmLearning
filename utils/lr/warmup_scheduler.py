from math import cos, pi

class LRScheduler(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    warmup_steps: int
        number of warmup steps used before this scheduler starts decay
    warmup_begin_lr: float
        if using warmup, the learning rate from which it starts warming up
    warmup_mode: string
        warmup can be done in two modes.
        'linear' mode gradually increases lr with each step in equal increments
        'constant' mode keeps lr at warmup_begin_lr for warmup_steps
    """
    def __init__(self, base_lr=0.01,
                 warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        self.base_lr = base_lr
        assert isinstance(warmup_steps, int)
        self.warmup_steps = warmup_steps

        self.warmup_final_lr = base_lr
        self.warmup_begin_lr = warmup_begin_lr
        if self.warmup_begin_lr > self.warmup_final_lr:
            raise ValueError("Base lr has to be higher than warmup_begin_lr")
        if self.warmup_steps < 0:
            raise ValueError("Warmup steps has to be positive or 0")
        if warmup_mode not in ['linear', 'constant']:
            raise ValueError("Supports only linear and constant modes of warmup")
        self.warmup_mode = warmup_mode

    def get_warmup_lr(self, num_update):
        assert num_update < self.warmup_steps
        if self.warmup_mode == 'linear':
            increase = (self.warmup_final_lr - self.warmup_begin_lr) \
                       * float(num_update) / float(self.warmup_steps)
            return self.warmup_begin_lr + increase
        elif self.warmup_mode == 'constant':
            return self.warmup_begin_lr
        else:
            raise ValueError("Invalid warmup mode %s"%self.warmup_mode)

    def __call__(self, optimizer, num_update):
        """Return a new learning rate.

        The ``num_update`` is the upper bound of the number of updates applied to
        every weight.

        Assume the optimizer has updated *i*-th weight by *k_i* times, namely
        ``optimizer.update(i, weight_i)`` is called by *k_i* times. Then::

            num_update = max([k_i for all i])

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")

class CosineScheduler(LRScheduler):
    """ Reduce the learning rate according to a cosine function

    Calculate the new learning rate by::

       final_lr + (start_lr - final_lr) * (1+cos(pi * nup/max_nup))/2
       if nup < max_nup, 0 otherwise.

    Parameters
    ----------
        max_update: int
            maximum number of updates before the decay reaches 0
        base_lr: float
            base learning rate
        final_lr: float
            final learning rate after all steps
        warmup_steps: int
            number of warmup steps used before this scheduler starts decay
        warmup_begin_lr: float
            if using warmup, the learning rate from which it starts warming up
        warmup_mode: string
            warmup can be done in two modes.
            'linear' mode gradually increases lr with each step in equal increments
            'constant' mode keeps lr at warmup_begin_lr for warmup_steps
    """

    def __init__(self, max_update,
                 base_lr=0.01,         # 预热阶段和cos阶段的交界值
                 final_lr=0,           # cos阶段的最终值
                 warmup_steps=0,       # 分多少步加热
                 warmup_begin_lr=0,    # 预热阶段的初始值
                 warmup_mode='linear'   # 预热阶段：'linear'线性增加，'constant'固定值
                 ):
        super(CosineScheduler, self).__init__(base_lr, warmup_steps, warmup_begin_lr, warmup_mode)
        assert isinstance(max_update, int)
        if max_update < 1:
            raise ValueError("maximum number of updates must be strictly positive")
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.max_steps = self.max_update - self.warmup_steps

    def adjust_learning_rate(self, optimizer, num_update):
        if num_update < self.warmup_steps:
            lr = self.get_warmup_lr(num_update)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif num_update <= self.max_update:
            lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                (1 + cos(pi * (num_update - self.warmup_steps) / self.max_steps)) / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr