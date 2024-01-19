import time
from tqdm import tqdm
import torch
from math import sqrt
import numpy as np
import torch.nn as nn

from src.model.base import UnmixingModel
from src import EPS

class SemiSupervisedUnmixingModel(UnmixingModel):
    def __init__(self):
        super().__init__()

    def compute_abundances(self, Y, D, r, *args, **kwargs):
        raise NotImplementedError(f"Solver is not implemented for {self}")

class FaSUn(SemiSupervisedUnmixingModel):
    def __init__(self, mu1=100, mu2=10, mu3=1, TA=5, TB=5, T=5000):
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.TA = TA
        self.TB = TB
        self.T =T

    @torch.no_grad() # NOTE: No gradients needed
    def compute_abundances(
        self,
        Y,
        D,
        r,
        *args,
        **kwargs,
    ):
        
        # Problem dimensions
        p, n = Y.shape
        m = D.shape[1]

        def loss(a, b):
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum()
        # Timing
        tic = time.time()

        A = (1 / r) * torch.ones((r, n))
        B = (1 / m) * torch.ones((m, r))
        L1 = torch.zeros((r, n))
        L2 = torch.zeros((m, r))
        L3 = torch.zeros((p, r))
        S1 = L1
        S2 = L2
        S3 = L3
        Y = torch.Tensor(Y)
        D = torch.Tensor(D)
        # Send matrices on GPU
        D = D.to(self.device)
        Y = Y.to(self.device)
        A = A.to(self.device)
        B = B.to(self.device)
        S1 = S1.to(self.device)
        S2 = S2.to(self.device)
        S3 = S3.to(self.device)
        L1 = L1.to(self.device)
        L2 = L2.to(self.device)
        L3 = L3.to(self.device)
        eye_r = torch.eye(r).to(self.device) 
        eye_m = torch.eye(m).to(self.device) 
        ones_r = torch.ones(r).to(self.device) 
        ones_m = torch.ones(m).to(self.device) 
        ones_n = torch.ones(n).to(self.device) 

        Q1inv = self.mu3 * D.t() @ D + self.mu2 * eye_m
        Z1 = torch.linalg.solve(Q1inv, ones_m)
        c1 = -1 / torch.dot(ones_m, Z1) 

        U1 = eye_m + c1 * torch.outer(Z1, ones_m)
        V1 = c1 * torch.outer(Z1, ones_r)

        Initloss = loss(A, B)
        print(f"Initial loss => {Initloss:.3e}")
        progress = tqdm(range(self.T))
        for ii in progress:
            updateloss = loss(A, B)
            progress.set_postfix_str(f"loss={updateloss:.4e}")
            
            Q2inv = self.mu3 * S3.t() @ S3 + self.mu1 * eye_r
            Z2 = torch.linalg.solve(Q2inv, ones_r)
            c2 = -1 / torch.dot(ones_r, Z2)
            U2 = eye_r + c2 * torch.outer(Z2, ones_r)
            V2 = c2 * torch.outer(Z2, ones_n)

            for jj in range(self.TA):
                WA = S3.t() @ Y + self.mu1 * (S1 - L1)
                A = U2 @ torch.linalg.solve(Q2inv, WA) - V2
                S1 = (A + L1)
                S1[S1<=0] = 0
                L1 = L1 + A - S1

            Q3inv = A @ A.t() + self.mu3 * eye_r

            for jj in range(self.TB):
                WB = self.mu3 * D.t() @ (S3 - L3) + self.mu2 * (S2 - L2)
                B = U1 @ torch.linalg.solve(Q1inv, WB) - V1
                S2 = (B + L2)
                S2[S2<=0] = 0
                S3 = torch.linalg.solve(Q3inv, Y @ A.t() + self.mu3 * (D @ B + L3),
                                        left=False)
                L2 = L2 + B - S2
                L3 = L3 + D @ B - S3

        timer = time.time() - tic
        self.print_time(timer)
        print(f"Final loss => {loss(A, B):.2e}")
        A = A.cpu().numpy()
        B = B.cpu().numpy()
        return A, B

class SUnShrink(SemiSupervisedUnmixingModel):
    def __init__(self, mu1=100, mu2=10, mu3=1, TA=5, TB=5, T=5000,
                 lambd=0.1, hard=True,):
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.TA = TA
        self.TB = TB
        self.T =T
        self.lambd = lambd

        self.hard = hard
        self.shrink = (torch.nn.Softshrink(self.lambd / self.mu2) 
            if not self.hard 
            else torch.nn.Hardshrink(sqrt(2 * self.lambd / self.mu2)))
        print(f"Using hard thresholding? {self.hard}")

    @torch.no_grad() # NOTE: No gradients needed
    def compute_abundances(
        self,
        Y,
        D,
        r,
        *args,
        **kwargs,
    ):
        
        # Problem dimensions
        p, n = Y.shape
        m = D.shape[1]

        def loss(a, b):
            penalty = (
                self.lambd * b.abs().sum() 
                if not self.hard 
                else self.lambd * b[b.abs() < EPS].sum()
            )
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum() + penalty
        # Timing
        tic = time.time()

        A = (1 / r) * torch.ones((r, n))
        #B = (1 / m) * torch.ones((m, r))
        B = torch.zeros((m, r))
        L1 = torch.zeros((r, n))
        L2 = torch.zeros((m, r))
        L3 = torch.zeros((p, r))
        S1 = L1
        S2 = L2
        S3 = L3
        Y = torch.Tensor(Y)
        D = torch.Tensor(D)
        # Send matrices on GPU
        D = D.to(self.device)
        Y = Y.to(self.device)
        A = A.to(self.device)
        B = B.to(self.device)
        S1 = S1.to(self.device)
        S2 = S2.to(self.device)
        S3 = S3.to(self.device)
        L1 = L1.to(self.device)
        L2 = L2.to(self.device)
        L3 = L3.to(self.device)
        eye_r = torch.eye(r).to(self.device) 
        eye_m = torch.eye(m).to(self.device) 
        ones_r = torch.ones(r).to(self.device) 
        ones_n = torch.ones(n).to(self.device) 

        Q1inv = self.mu3 * D.t() @ D + self.mu2 * eye_m

        Initloss = loss(A, B)
        print(f"Initial loss => {Initloss:.3e}")
        progress = tqdm(range(self.T))
        for ii in progress:
            updateloss = loss(A, B)
            progress.set_postfix_str(f"loss={updateloss:.4e}")
            
            Q2inv = self.mu3 * S3.t() @ S3 + self.mu1 * eye_r
            Z2 = torch.linalg.solve(Q2inv, ones_r)
            c2 = -1 / torch.dot(ones_r, Z2)
            U2 = eye_r + c2 * torch.outer(Z2, ones_r)
            V2 = c2 * torch.outer(Z2, ones_n)

            for jj in range(self.TA):
                WA = S3.t() @ Y + self.mu1 * (S1 - L1)
                A = U2 @ torch.linalg.solve(Q2inv, WA) - V2
                S1 = (A + L1)
                S1[S1<=0] = 0
                L1 = L1 + A - S1

            Q3inv = A @ A.t() + self.mu3 * eye_r

            for jj in range(self.TB):
                WB = self.mu3 * D.t() @ (S3 - L3) + self.mu2 * (S2 - L2)
                #B = U1 @ torch.linalg.solve(Q1inv, WB) - V1
                B = torch.linalg.solve(Q1inv, WB)
                S2 = self.shrink(B + L2)
                S2[S2<=0] = 0
                # TODO: upper bound?
                S2[S2>=1] = 1
                S3 = torch.linalg.solve(Q3inv, Y @ A.t() + self.mu3 * (D @ B + L3),
                                        left=False)
                L2 = L2 + B - S2
                L3 = L3 + D @ B - S3

        timer = time.time() - tic
        self.print_time(timer)
        print(f"Final loss => {loss(A, B):.2e}")
        A = A.cpu().numpy()
        B = S2.cpu().numpy()
        return A, B

class EDAA(SemiSupervisedUnmixingModel):
    def __init__(self, T=100, K1=5, K2=5, M=50, *args, **kwargs):
        super().__init__()
        self.T = T
        self.K1 = K1
        self.K2 = K2
        self.M = M

    @torch.no_grad()
    def compute_abundances(self, Y, D, r, *args, **kwargs):
        best_B = None
        best_A = None

        p, n = Y.shape
        p, m = D.shape
        
        def loss(a, b):
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum()

        def residual_l1(a, b):
            return (Y - (D @ b) @ a).abs().sum()

        def grad_A(a, b):
            DB = D @ b
            return - DB.t() @ (Y - DB @ a)
    
        def grad_B(a, b):
            return -D.t() @ ((Y - D @ b @ a) @ a.t())

        def update(a, b):
            return torch.softmax(torch.log(a) + b, dim=0)

        def computeLA(b):
            DB = D @ b
            S = torch.linalg.svdvals(DB)
            return S[0] ** 2

        def max_correl(e):
            return np.max(np.corrcoef(e.T) - np.eye(r))

        results = {}

        tic = time.time()

        # Data to tensor
        Y = torch.Tensor(Y)
        D = torch.Tensor(D)

        for mm in tqdm(range(self.M)):
            torch.manual_seed(mm)
            generator = np.random.RandomState(mm)

            B = torch.softmax(0.1 * torch.rand((m, r)), dim=0)
            A = (1 / r) * torch.ones((r, n))
            #B = torch.zeros((m, r))

            # Send matrices to GPU
            Y = Y.to(self.device)
            D = D.to(self.device)
            A = A.to(self.device)
            B = B.to(self.device)

            # Random step size factor
            gamma = 2 ** generator.randint(-3, 4)

            # Compute step sizes
            eta_1 = gamma / computeLA(B)
            #eta_1 = gamma
            eta_2 = eta_1 * ((r / n) ** 0.5)

            for _ in range(self.T):
                for _ in range(self.K1):
                    A = update(A, -eta_1 * grad_A(A, B))

                for _ in range(self.K2):
                    B = update(B, -eta_2 * grad_B(A, B))

            fit_m = residual_l1(A, B).item()
            E = (D @ B).cpu().numpy()
            A = A.cpu().numpy()
            B = B.cpu().numpy()
            mu_m = max_correl(E)
            results[mm] = {"mu_m": mu_m, 
                          "Em": E, 
                          "Am": A, 
                          "Bm": B, 
                          "fit_m": fit_m, 
                          "gamma_m": gamma,}

        min_fit_l1 = np.min([v["fit_m"] for v in results.values()])

        def fit_l1_cutoff(idx, tol=0.05):
            val = results[idx]["fit_m"]
            return (abs(val - min_fit_l1) / abs(val)) < tol

        sorted_indices = sorted(filter(fit_l1_cutoff, results),
                                key=lambda x: results[x]["mu_m"])

        best_result_idx = sorted_indices[0]
        best_result = results[best_result_idx]

        best_A = best_result["Am"]
        best_B = best_result["Bm"]

        timer = time.time() - tic
        self.print_time(timer)

        return best_A, best_B


class X_step(nn.Module, SemiSupervisedUnmixingModel):
    def __init__(self, D, mu_init, *args, **kwargs):

        super().__init__(*args, **kwargs)

        p, m = D.shape
        self.W = nn.Linear(p, m, bias=False)
        self.B = nn.Linear(m, m, bias=False)

        # init
        Q = D.T @ D + mu_init * torch.eye(m)

        self.W.weight.data = torch.linalg.solve(Q, D.T)
        self.B.weight.data = torch.linalg.solve(Q, mu_init * torch.eye(m))

    def forward(self, y, s, l):
        return self.W(y) + self.B(s + l)

class S_step(nn.Module):
    def __init__(self, bias_init=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # dimensions
        #p, m = D.shape
        self.bias = nn.Parameter(data=bias_init * torch.ones(1),
                             requires_grad=True)

    def forward(self, x, s, l):
        return nn.functional.relu(x + l - self.bias)

class L_step(nn.Module):
    def __init__(self, eta_init=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = nn.Parameter(data=eta_init * torch.ones(1),
                                requires_grad=True)

    def forward(self, x, s, l):
        # NOTE: Which sign here in front of self.eta?
        return l - self.eta * (x - s)


class UnrolledFCLS(nn.Module, SemiSupervisedUnmixingModel):
    def __init__(self,
                 lr,
                 epochs,
                 batchsize,
                 nblocks,
                 mu,
                 bias,
                 eta,
                 tied=False,
                 *args,
                 **kwargs,):
        nn.Module.__init__(self)
        SemiSupervisedUnmixingModel.__init__(self)

        self.lr = lr
        self.epochs = epochs
        self.batchsize = batchsize
        self.nblocks = nblocks
        self.tied = tied
        # Hyperparameters
        self.mu_init = mu
        self.bias_init = bias
        self.eta_init = eta

    def init_architecture(self, Y, D):
        self.p, self.m = D.shape
        
        self.x_steps = nn.ModuleList()
        self.s_steps = nn.ModuleList()
        self.l_steps = nn.ModuleList()


        if self.tied:
            x_step = X_step(D, self.mu_init)
            s_step = S_step(self.bias_init)
            l_step = L_step(self.eta_init)

            for _ in range(self.nblocks):
                self.x_steps.append(x_step)
                self.s_steps.append(s_step)
                self.l_steps.append(l_step)

        else:
            for _ in range(self.nblocks):
                self.x_steps.append(X_step(D, self.mu_init))
                self.s_steps.append(S_step(self.bias_init))
                self.l_steps.append(L_step(self.eta_init))

    def forward(self, y):
        bs, _ = y.shape

        s = torch.zeros((bs, self.m)).to(self.device)
        l = torch.zeros((bs, self.m)).to(self.device)
        for ii in range(self.nblocks):
            x = self.x_steps[ii](y, s, l)
            s = self.s_steps[ii](x, s, l)
            l = self.l_steps[ii](x, s, l)

        
        #abund = torch.softmax(x, dim=0)
        #abund = s / (s.sum(1, keepdims=True) + 1e-12)
        #abund = torch.softmax(x, dim=1)
        #breakpoint()
        abund = s / (s.sum(1, keepdims=True) + 1e-12)

        output = nn.functional.linear(abund, self.D)
        return abund, output

    
    def compute_abundances(self, Y, D, r, *args, **kwargs):
        tic = time.time()
        
        D = torch.Tensor(D)
        self.D = D.to(self.device)
        self.init_architecture(Y, D)
        self = self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_db = torch.utils.data.TensorDataset(torch.Tensor(Y.T))
        dataloader = torch.utils.data.DataLoader(
            train_db,
            batch_size=self.batchsize,
            shuffle=True,
        )

        progress = tqdm(range(self.epochs))
        self.train()

        for ii in progress:
            running_loss = 0
            for x, y in enumerate(dataloader):
                y = y[0].to(self.device)

                abund, output = self(y)
                
                loss = nn.functional.mse_loss(y, output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            progress.set_postfix_str(f"loss={running_loss:.2e}")

        self.eval()
        with torch.no_grad():
            abund, _ = self(torch.Tensor(Y.T).to(self.device))
            X = abund.cpu().numpy().T

        B = np.eye(D.shape[1])
        timer = time.time() - tic
        self.print_time(timer)
        breakpoint()

        return X, B



