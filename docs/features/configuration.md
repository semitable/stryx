# Configuration

Stryx uses **Pydantic** models as the single source of truth for configuration.

## Defining a Schema

```python
from pydantic import BaseModel, Field

class OptimizerConfig(BaseModel):
    lr: float = 1e-4
    weight_decay: float = 0.01

class Config(BaseModel):
    exp_name: str = "default"
    seed: int = 42
    optim: OptimizerConfig = Field(default_factory=OptimizerConfig)
```

## Nested Configs & Unions

Stryx supports nested BaseModels and Unions (polymorphism).

```python
from typing import Literal, Union

class Adam(BaseModel):
    kind: Literal["adam"] = "adam"
    beta1: float = 0.9

class SGD(BaseModel):
    kind: Literal["sgd"] = "sgd"
    momentum: float = 0.9

class Config(BaseModel):
    optim: Union[Adam, SGD] = Field(default_factory=Adam, discriminator="kind")
```

Switching between them in CLI:
```bash
# Use Adam (default)
uv run train.py try optim.beta1=0.8

# Switch to SGD
uv run train.py try optim={'kind':'sgd','momentum':0.95}
```
