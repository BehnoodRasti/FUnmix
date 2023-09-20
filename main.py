"""
Main file to handle experiments configuration using MLXP
"""

import logging
import logging.config

import mlxp

def set_seeds(seed):
    import torch
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)

@mlxp.launch(config_path="./config", seeding_function=set_seeds)
def main(ctx: mlxp.Context) -> None:

    cfg = ctx.config
    mode = cfg.mode

    logging.config.dictConfig(cfg)

    log = logging.getLogger(__name__)
    log.debug(f"Config:\n{cfg}")
    log.info(f"Unmixing mode: {mode.upper()}")

    if mode == "semi":
        from src.semisupervised import main as _main
    elif mode == "supervised":
        from src.supervised import main as _main
    else:
        raise ValueError(f"Mode {mode} is invalid")

    try:
        _main(ctx)
    except Exception:
        log.error("Exception caught", exc_info=True)

if __name__ == "__main__":
    main()

