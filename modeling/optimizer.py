import torch
import torch.optim as optim
from lion_pytorch import Lion

_optimizer_entrypoints = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "lion": Lion,
}

def create_optim(optim, model, lr, **kargs):
    if optim in _optimizer_entrypoints:
        optim_constructor = _optimizer_entrypoints[optim]
        optimizer = optim_constructor(model.parameters(), lr=lr, **kargs)
        return optimizer
    else:
        raise RuntimeError(f"Unknown optimizer ({optim})")