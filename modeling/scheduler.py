import torch.optim.lr_scheduler as lr_scheduler

_scheduler_entrypoints = {
    "step": lr_scheduler.StepLR,
    "plateau": lr_scheduler.ReduceLROnPlateau,
    "cosine": lr_scheduler.CosineAnnealingLR,
}

def create_sched(sched, optim, max_epochs, **kargs):
    if sched in _scheduler_entrypoints:
        sched_constructor = _scheduler_entrypoints[sched]
        is_plateau = False
        if sched == "step":
            scheduler = sched_constructor(optim, step_size=10, **kargs)
        elif sched == "plateau":
            scheduler = sched_constructor(optim, **kargs)
            is_plateau = True
        elif sched == "cosine":
            scheduler = sched_constructor(optim, max_epochs, **kargs)
        
        return scheduler, is_plateau
    else:
        raise RuntimeError(f"Unknown scheduler ({sched})")