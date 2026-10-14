import json
from dataclasses import asdict, dataclass
from pathlib import Path

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}


@dataclass
class TrainConfig:
    input_size: int = 224
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    seed: int = 42
    patience: int = 5
    pretrained: bool = True
    strategy: str = "baseline"
    device: str = "auto"
    workers: int = 0
    threads: int = 2
    max_batches: int | None = None
    smoke: bool = False

    def validate(self):
        if min(self.input_size, self.batch_size, self.epochs, self.patience, self.threads) <= 0:
            raise ValueError("Sizes, epochs, patience and threads must be positive")
        if self.input_size < 32 or self.batch_size < 2:
            raise ValueError("Input size >=32 and training batch size >=2 required")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.workers < 0:
            raise ValueError("Invalid optimizer or worker configuration")
        if self.strategy not in ("baseline", "class_weighted", "balanced_sampling"):
            raise ValueError("Unknown disease balancing strategy")
        if self.max_batches is not None and (self.max_batches < 1 or not self.smoke):
            raise ValueError("Batch limits require explicitly marked smoke mode")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda")

    @classmethod
    def read(cls, path):
        result = cls(**json.loads(Path(path).read_text()))
        result.validate()
        return result

    def dict(self):
        return asdict(self)
