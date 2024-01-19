"""
Semi-supervised unmixing methods main source file
"""
import mlxp
from mlxp.launcher import _instance_from_config
import logging
import numpy as np

from src.data.utils import SVD_projection
from src.utils.metrics import SRE, aRMSE, compute_metric
from src.data import Estimate

log = logging.getLogger(__name__)


def main(ctx: mlxp.Context) -> None:
    log.info("Semi-Supervised Unmixing - [START]...")
    cfg = ctx.config
    logger = ctx.logger

    # Get noise
    noise = _instance_from_config(cfg.noise)
    # Get HSI
    hsi = _instance_from_config(cfg.data)
    # Print HSI information
    log.info(hsi)
    # Get data
    Y, r, D = hsi.get_data()
    # Get image dimensions
    h, w = hsi.get_img_shape()
    # Clip data
    #Y[Y <= 0] = 0
    #Y[Y >= 1] = 1
    # Apply noise
    Y = noise.apply(Y)
    # L2 normalization
    if cfg.l2_normalization:
        Y = Y / np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
    # Apply SVD projection
    if cfg.projection:
        Y = SVD_projection(Y, r)
    # Build model
    model = _instance_from_config(cfg.model)
    # Solve unmixing
    A_hat, B_hat = model.compute_abundances(Y, D, r=r, h=h, w=w)

    A_full = B_hat @ A_hat
    E_hat = D @ B_hat

    logger.log_artifact(Estimate(E_hat, A_hat, h, w), "estimates")

    if hsi.has_GT():
        # Get ground truth
        _, A_gt = hsi.get_GT()
        # NOTE: Select only the first relevant components by using the dataset index
        index = hsi.get_index()
        A1 = A_full[index]
        # Get labels
        labels = hsi.get_labels()
        # Compute and log metrics
        logger.log_metrics(
            compute_metric(
                SRE(),
                A_gt,
                A1,
                labels,
                detail=False,
                on_endmembers=False,
            ),
            log_name="SRE",
        )
        logger.log_metrics(
            compute_metric(
                aRMSE(),
                A_gt,
                A1,
                labels,
                detail=True,
                on_endmembers=False,
            ),
            log_name="aRMSE",
        )
    log.info("Semi-Supervised Unmixing - [END]")
