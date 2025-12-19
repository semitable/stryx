from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Ctx:
    schema: type
    configs_dir: Path
    runs_dir: Path
    func: Callable[[Any], Any]