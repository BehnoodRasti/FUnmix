"""
HSI data base class
"""
import logging
import os
from dataclasses import dataclass

import scipy.io as sio
import numpy as np
from mlxp.data_structures.artifacts import Artifact

from src import EPS

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

INTEGER_VALUES = ("h", "w", "m", "p", "r", "n")


class HSI:
    def __init__(
        self,
        dataset: str,
        data_dir: str = "./data",
    ) -> None:
        # Populate with Null data
        # integers
        self.h = 0
        self.w = 0
        self.m = 0
        self.p = 0
        self.r = 0
        self.n = 0
        # arrays
        self.Y = np.zeros((self.p, self.n))
        self.E = np.zeros((self.p, self.r))
        self.A = np.zeros((self.r, self.n))
        self.D = np.zeros((self.p, self.m))
        self.labels = []
        self.index = []

        # Locate and check data file
        self.name = dataset
        filename = f"{self.name}.mat"
        path = os.path.join(data_dir, filename)
        log.debug(f"Path to be opened: {path}")
        assert os.path.isfile(path)

        # Open data file
        data = sio.loadmat(path)
        log.debug(f"Data keys: {data.keys()}")

        # Populate attributes based on data file values
        for key in filter(
            lambda k: not k.startswith("__"),
            data.keys(),
        ):
            self.__setattr__(
                key, data[key].item() if key in INTEGER_VALUES else data[key]
            )

        if "n" not in data.keys():
            self.n = self.h * self.w

        # Check data
        assert self.n == self.h * self.w
        assert self.Y.shape == (self.p, self.n)

        self.has_dict = False
        if "D" in data.keys():
            self.has_dict = True
            assert self.D.shape == (self.p, self.m)

        if "index" in data.keys():
            self.index = list(self.index.squeeze())

    def get_data(self):
        return (
            self.Y,
            self.r,
            self.D,
        )

    def get_HSI_dimensions(self):
        return {
            "bands": self.p,
            "pixels": self.n,
            "lines": self.h,
            "samples": self.w,
            "atoms": self.m,
        }

    def get_img_shape(self):
        return (
            self.h,
            self.w,
        )

    def get_labels(self):
        return self.labels

    def get_index(self):
        return self.index

    def __repr__(self) -> str:
        msg = f"HSI => {self.name}\n"
        msg += "------------------------------\n"
        msg += f"{self.p} bands,\n"
        msg += f"{self.h} lines, {self.w} samples ({self.n} pixels),\n"
        msg += f"{self.r} endmembers ({self.labels}),\n"
        msg += f"{self.m} atoms\n"
        msg += f"GlobalMinValue: {self.Y.min()}, GlobalMaxValue: {self.Y.max()}\n"
        return msg


class HSIWithGT(HSI):
    def __init__(
        self,
        dataset,
        data_dir,
    ):
        super().__init__(
            dataset=dataset,
            data_dir=data_dir,
        )

        # Sanity check on ground truth
        assert self.E.shape == (self.p, self.r)
        assert self.A.shape == (self.r, self.n)

        try:
            assert len(self.labels) == self.r
            tmp_labels = list(self.labels)
            self.labels = [s.strip(" ") for s in tmp_labels]

        except Exception:
            # Create numeroted labels
            self.labels = [f"#{ii}" for ii in range(self.r)]

        # Check physical constraints
        # Abundance Sum-to-One Constraint (ASC)
        assert np.allclose(
            self.A.sum(0),
            np.ones(self.n),
            rtol=1e-3,
            atol=1e-3,
        )
        # Abundance Non-negative Constraint (ANC)
        assert np.all(self.A >= -EPS)
        # Endmembers Non-negative Constraint (ENC)
        assert np.all(self.E >= -EPS)

    def get_GT(self):
        return (
            self.E,
            self.A,
        )

    def has_GT(self):
        return True


class RealHSI(HSI):
    def __init__(
        self,
        dataset,
        data_dir,
        r=3,
    ):
        super().__init__(
            dataset=dataset,
            data_dir=data_dir,
        )
        self.r = r
        # Create labels
        self.labels = [f"#{ii}" for ii in range(self.r)]

    def has_GT(self):
        return False


@dataclass
class Estimate(Artifact):
    ext = ".mat"

    def __init__(self, Ehat, Ahat, h, w):
        data = {"E": Ehat, "A": Ahat.reshape(-1, h, w)}
        super().__init__(obj=data, ext=self.ext)

    def _save(self, fname="estimates"):
        sio.savemat(f"{fname}{self.ext}", self.obj)


if __name__ == "__main__":
    print("TODO!")
