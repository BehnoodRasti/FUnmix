"""
Implement metrics used in sparse unmixing
"""

import numpy as np
import numpy.linalg as LA


class Metric:
    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        assert type(X) == type(Xref)
        return X, Xref

    def __call__(self, X, Xref):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}"


class RMSE(Metric):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        X, Xref = self._check_input(X, Xref)
        return 100 * np.sqrt(((X - Xref) ** 2).mean())


class SRE(Metric):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        X, Xref = self._check_input(X, Xref)
        return 20 * np.log10(LA.norm(Xref, "fro") / LA.norm(Xref - X, "fro"))
