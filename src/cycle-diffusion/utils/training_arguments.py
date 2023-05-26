from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    cfg: str = None
    custom_z_name: str = None  # NEW
    disable_wandb: bool = False  # NEW
    save_images: bool = False # NEW
    img_type: str = None # NEW

    verbose: bool = field(
        default=False,
    )
