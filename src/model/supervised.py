import time
from tqdm import tqdm
import torch

from src.model.base import UnmixingModel

class SupervisedUnmixingModel(UnmixingModel):
    def __init__(self):
        super().__init__()

    def compute_abundances(self, Y, E, *args, **kwargs):
        raise NotImplementedError(f"Solver not implemented for {self}")

class FastFCLS(SupervisedUnmixingModel):
    def __init__(self,mu=1, T=10000):
        self.mu = mu
        self.T =T

    @torch.no_grad() # NOTE: No gradients needed
    def compute_abundances(
        self,
        Y,
        E,
        *args,
        **kwargs,
    ):
        # Problem dimensions
        p, n = Y.shape
        r = E.shape[1]

        # Problem objective
        def loss(a):
            return 0.5 * ((Y - E @ a) ** 2).sum()

        # Timing
        tic = time.time()

        # Initialization
        A = (1 / r) * torch.ones((r, n))
        L = torch.zeros((r, n))
        S = L
        Y = torch.Tensor(Y)
        E = torch.Tensor(E)
        # Send matrices on GPU
        E = E.to(self.device)
        Y = Y.to(self.device)
        A = A.to(self.device)
        S = S.to(self.device)
        L = L.to(self.device)
        eye_r = torch.eye(r).to(self.device) 
        ones_r = torch.ones(r).to(self.device) 
        ones_n = torch.ones(n).to(self.device) 

        EtY = E.t() @ Y
        EtE = E.t() @ E
        Qinv = EtE + self.mu * eye_r
        Z = torch.linalg.solve(Qinv, ones_r)
        c = -1 / torch.dot(ones_r, Z)

        U = eye_r + c * torch.outer(Z, ones_r)
        V = c * torch.outer(Z, ones_n)

        Initloss = loss(A)
        print(f"Initial loss => {Initloss:.3e}")
        progress = tqdm(range(self.T))
        for ii in progress:  
            updateloss = loss(A)
            progress.set_postfix_str(f"loss={updateloss:.4e}")

            A = U @ torch.linalg.solve(Qinv, EtY + self.mu * (S - L)) - V
            S = (A + L)
            S[S<=0] = 0
            L = L + A - S

        timer = time.time() - tic
        self.print_timer(timer)
        print(f"Final loss => {loss(A):.2e}")
        A = A.cpu().numpy()
        return A

# TODO: Implement FCLS
# TODO: Implement DecompSimplex
# TODO: Implement SUnSAL with lambda = 0 (Python)



