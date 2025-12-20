import os
from pydantic import BaseModel
import stryx

class Config(BaseModel):
    crash: bool = False

@stryx.cli(schema=Config)
def main(cfg: Config):
    rank = os.getenv("RANK", "0")
    print(f"Rank {rank} running")
    if cfg.crash and rank == "0":
        raise ValueError("Boom")
    return {"rank": rank, "status": "ok"}

if __name__ == "__main__":
    main()
