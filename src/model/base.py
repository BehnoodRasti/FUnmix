"""
Model related globals
"""
import torch

class UnmixingModel:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def print_time(self, timer):
        print(f"{self} took {timer:.2f}s")
