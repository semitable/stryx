from pathlib import Path
from dataclasses import dataclass


@dataclass
class Ctx:
    schema: type
    configs_dir: Path
    runs_dir: Path
