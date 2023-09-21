"""
Noise generation utilities
"""
import logging

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class AdditiveWhiteGaussianNoise:
    def __init__(self, SNR=None):
        self.SNR = SNR

    def apply(self, Y):
        """
        Compute sigmas for the desired SNR given a flattened input HSI Y
        """
        log.debug(f"Y shape => {Y.shape}")
        assert len(Y.shape) == 2
        p, n = Y.shape
        log.info(f"Desired SNR => {self.SNR}")

        #######
        # Fit #
        #######
        if self.SNR is None:
            sigmas = np.zeros(p)
        else:
            assert self.SNR > 0, "SNR must be strictly positive"
            # Uniform across bands
            sigmas = np.ones(p)
            # Normalization
            sigmas /= np.linalg.norm(sigmas)
            log.debug(f"Sigmas after normalization: {np.round(sigmas[0], 3)}")
            # Compute sigma mean based on SNR
            num = np.sum(Y**2) / n
            denom = 10 ** (self.SNR / 10)
            sigmas_mean = np.sqrt(num / denom)
            log.debug(f"Sigma mean based on SNR: {np.round(sigmas_mean, 3)}")
            # Noise variance
            sigmas *= sigmas_mean
            log.debug(f"Final sigmas value: {np.round(sigmas[0], 3)}")

        #############
        # Transform #
        #############
        noise = np.diag(sigmas) @ np.random.randn(p, n)

        # Return additive noise
        return Y + noise
