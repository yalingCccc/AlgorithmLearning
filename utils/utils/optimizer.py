from torch import optim

def get_optimizer(opt_fn):
    opt_fn = opt_fn.lower()
    if opt_fn == 'adam':
        return optim.Adam
    elif opt_fn == 'sgd':
        return optim.SGD
    else:
        print("未定义的优化器")
        return None