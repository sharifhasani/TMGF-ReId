from .scheduler import CosineLRScheduler, WarmupMultiStepLR


def create_scheduler(cfg, optimizer):
    scheduler_type = cfg.OPTIM.SCHEDULER_TYPE
    if scheduler_type == 'cosine':
        num_epochs = cfg.TRAIN.EPOCHS
        # type 1
        # lr_min = 0.01 * cfg.SOLVER.BASE_LR
        # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
        # type 2
        lr_min = 0.002 * cfg.OPTIM.BASE_LR
        warmup_lr_init = 0.01 * cfg.OPTIM.BASE_LR
        # type 3
        # lr_min = 0.001 * cfg.SOLVER.BASE_LR
        # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

        warmup_t = cfg.OPTIM.WARMUP_EPOCHS
        noise_range = None

        lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_epochs,
                lr_min=lr_min,
                t_mul= 1.,
                decay_rate=0.1,
                warmup_lr_init=warmup_lr_init,
                warmup_t=warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
                noise_range_t=noise_range,
                noise_pct= 0.67,
                noise_std= 1.,
                noise_seed=42,
            )
    elif scheduler_type == 'warmup':
        lr_scheduler = WarmupMultiStepLR(optimizer, cfg.OPTIM.MILESTONES, gamma=cfg.OPTIM.GAMMA,
                                         warmup_factor=cfg.OPTIM.WARMUP_FACTOR,
                                         warmup_iters=cfg.OPTIM.WARMUP_EPOCHS)
    else:
        raise ValueError(f'Invalid scheduler type {scheduler_type}!')

    return lr_scheduler