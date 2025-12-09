from dataclasses import dataclass

@dataclass
class OptConfig:
    """Optimizer configuration for training."""
    learning_rate: float = 3.63e-4
    find_learning_rate: bool = False
    betas = [0.9, 0.999]
    weight_decay: float = 0.001
    lr_scheduler_interval: str = "step"
    lr_scheduler_freq: int = 100
    lr_scheduler_tmax: int = 2500
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5

    # early stopping
    early_stopping: bool = True
    patience: int = 10
    monitor: str = "val_loss"
    mode: str = "min"