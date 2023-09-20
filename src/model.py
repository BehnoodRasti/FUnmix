import time
from tqdm import tqdm
import torch
from math import sqrt

class FaSUn:
    def __init__(self, mu1=100, mu2=10, mu3=1, TA=5, TB=5, T=5000):
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.TA = TA
        self.TB = TB
        self.T =T
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.running_time = -1


    @torch.no_grad() # NOTE: No gradients needed
    def solve(
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

        self.running_time = time.time() - tic
        print(f"FaSUn took {self.running_time:.1f}s")
        print(f"Final loss => {loss(A, B):.2e}")
        A = A.cpu().numpy()
        B = B.cpu().numpy()
        return A, B



class FastFCLS:
    def __init__(self,mu=1, T=10000):
        self.mu = mu
        self.T =T
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.running_time = -1

    @torch.no_grad() # NOTE: No gradients needed
    def solve(
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

        self.running_time = time.time() - tic
        print(f"FastFCLS took {self.running_time:.1f}s")
        print(f"Final loss => {loss(A):.2e}")
        A = A.cpu().numpy()
        return A



class SUnShrink:
    def __init__(self, mu1=100, mu2=10, mu3=1, TA=5, TB=5, T=5000,
                 lambd=0.1, hard=True,):
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.TA = TA
        self.TB = TB
        self.T =T
        self.lambd = lambd

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.running_time = -1

        self.shrink = (torch.nn.Softshrink(self.lambd / self.mu2) 
            if not hard 
            else torch.nn.Hardshrink(sqrt(2 * self.lambd / self.mu2)))
        print(f"Using hard thresholding? {hard}")

    @torch.no_grad() # NOTE: No gradients needed
    def solve(
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

        # TODO: Change loss function to include penalty
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
                S3 = torch.linalg.solve(Q3inv, Y @ A.t() + self.mu3 * (D @ B + L3),
                                        left=False)
                L2 = L2 + B - S2
                L3 = L3 + D @ B - S3

        self.running_time = time.time() - tic
        print(f"FaSUn took {self.running_time:.1f}s")
        print(f"Final loss => {loss(A, B):.2e}")
        A = A.cpu().numpy()
        B = B.cpu().numpy()
        return A, B


