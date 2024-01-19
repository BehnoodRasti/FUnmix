"""
Endmembers extractor methods
"""
import logging

import numpy as np
import numpy.linalg as LA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseExtractor:
    def __init__(self):
        self.seed = None
        self.time = -1

    def extract_endmembers(self, Y, r, seed=0, *args, **kwargs):
        return NotImplementedError

    def __repr__(self):
        msg = f"{self.__class__.__name__}_seed{self.seed}"
        return msg

    def print_time(self, timer):
        msg = f"{self} took {self.time:.2f} seconds..."
        return msg


class RandomPositiveMatrix(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, Y, r, seed=0, *args, **kwargs):
        p, n = Y.shape
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)
        return generator.random(size=(p, r))


class RandomPixels(BaseExtractor):
    def __init__(self):
        super().__init__()

    def extract_endmembers(self, Y, r, seed=0, *args, **kwargs):
        p, n = Y.shape
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)
        indices = generator.integers(low=0, high=n, size=r)
        pixels = Y[:, indices]
        assert pixels.shape == (p, r)
        return pixels


class VCA(BaseExtractor):
    def __init__(self, seed=0, snr_input=None):
        super().__init__()
        self.seed = seed
        self.snr_input = 0 if snr_input is None else snr_input

    def extract_endmembers(self, Y, r, *args, **kwargs):
        """
        Vertex Component Analysis

        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm
        under a non-specified Copyright (c)
        Translation of last version at 22-February-2018 
        (Matlab version 2.1 (7-May-2004))

        more details on:
        Jose M. P. Nascimento and Jose M. B. Dias
        "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
        submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
        """
        p, n = Y.shape
        generator = np.random.default_rng(seed=self.seed)

        #############################################
        # SNR Estimates
        #############################################

        if self.snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean
            Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(n))[0][
                :, :r
            ]  # computes the R-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

            SNR = self.estimate_snr(Y, y_m, x_p)

            logger.info(f"SNR estimated = {SNR}[dB]")
        else:
            SNR = self.snr_input
            logger.info(f"input SNR = {SNR}[dB]\n")

        SNR_th = 15 + 10 * np.log10(r)
        #############################################
        # Choosing Projective Projection or
        #          projection to p-1 subspace
        #############################################

        if SNR < SNR_th:
            logger.info("... Select proj. to R-1")

            d = p - 1
            if self.snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = np.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(n))[0][
                    :, :d
                ]  # computes the p-projection matrix
                x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

            Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

            x = x_p[:d, :]  #  x_p =  Ud.T * Y_o is on a R-dim subspace
            c = np.amax(np.sum(x**2, axis=0)) ** 0.5
            y = np.vstack((x, c * np.ones((1, n))))
        else:
            logger.info("... Select the projective proj.")

            d = p
            Ud = LA.svd(np.dot(Y, Y.T) / float(n))[0][
                :, :d
            ]  # computes the p-projection matrix

            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(
                Ud, x_p[:d, :]
            )  # again in dimension L (note that x_p has no null mean)

            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
            y = x / np.dot(u.T, x)

        #############################################
        # VCA algorithm
        #############################################

        indices = np.zeros((r), dtype=int)
        A = np.zeros((r, r))
        A[-1, 0] = 1

        for i in range(r):
            w = generator.random(size=(r, 1))
            f = w - np.dot(A, np.dot(LA.pinv(A), w))
            f = f / np.linalg.norm(f)

            v = np.dot(f.T, y)

            indices[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indices[i]]  # same as x(:,indice(i))

        E = Yp[:, indices]

        logger.debug(f"Indices chosen to be the most pure: {indices}")
        self.indices = indices

        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        p, n = Y.shape  # L number of bands (channels), N number of pixels
        r, n = x.shape  # p number of endmembers (reduced dimension)

        P_y = np.sum(Y**2) / float(n)
        P_x = np.sum(x**2) / float(n) + np.sum(r_m**2)
        snr_est = 10 * np.log10((P_x - p / p * P_y) / (P_y - P_x))

        return snr_est
