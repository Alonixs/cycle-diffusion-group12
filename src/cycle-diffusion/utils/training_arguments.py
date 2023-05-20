from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    cfg: str = None
    custom_z_name: str = None  # NEW
    disable_wandb: bool = False  # NEW

    verbose: bool = field(
        default=False,
    )
