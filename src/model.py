import time
import numpy as np
from tqdm import tqdm
import torch
from src import EPS
import torch.nn.functional as F
import scipy.linalg
class FASUn:
    def __init__(self,mu1=100,mu2=10,mu3=1, TA=5, TB=5,T=5000):
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.TA = TA
        self.TB = TB
        self.T =T
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.running_time = -1
    def solve(
        self,
        Y,
        D,
        p,
        *args,
        **kwargs,
    ):
        
        L, N = Y.shape
        LibS=D.shape[1]
        def residual(a, b):
            return 0.5 * ((Y - (D @ b) @ a) ** 2).sum()

        def lossTot(a, b):
            return residual(a, b) #+ lamb * b.abs().sum()
        results = {}
        #eps=sys.float_info.epsilon
        tic = time.time()

        #for m in tqdm(range(self.M)):
        #    torch.manual_seed(m + seed)
        #    generator = np.random.RandomState(m + seed)
        #    with torch.no_grad():
        A = (1 / p) * torch.ones((p, N))
        B=(1 / LibS) * torch.ones((LibS, p))
        L1 = torch.zeros((p,N))
        L2 = torch.zeros((LibS,p))
        L3 = torch.zeros((L,p))
        S1=L1
        S2=L2
        S3=L3
        inv1=torch.zeros((p,p))
        inv2=torch.zeros((1,1))
        inv3=torch.zeros((LibS,LibS))
        inv4=inv2
        # V, SS, U = scipy.linalg.svd(Y, full_matrices=False)
        # PC=np.diag(SS)@U
        # Y_DN=V[:,:p]@PC[:p,:]
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
        eyep=torch.eye(p).to(self.device) 
        eyeLibS=torch.eye(LibS).to(self.device) 
        onesp=torch.ones((p,1)).to(self.device) 
        onesLibS=torch.ones((LibS,1)).to(self.device) 
        onesn=torch.ones((N,1)).to(self.device) 
        eyeL=torch.eye(L).to(self.device) 
        # Random Step size factor
        DtY=D.t()@Y
        inv3=torch.inverse(self.mu3*D.t()@D+self.mu2*eyeLibS)#+eyeLibS*eps)
        inv4=-1/(onesLibS.t()@inv3@onesLibS)
        Initloss = lossTot(A, B)
        print(f"Initial loss => {Initloss:.3e}")
        updateloss_tmp=10*Initloss
        progress = tqdm(range(self.T))
        count=0
        for ii in progress:  
            updateloss = lossTot(A, B)
            # if ii>3000 and 0<updateloss_tmp-updateloss <1e-4: 
            #     #
            #     count=count+1
                
            # if count==1: 
            #     #print(f"{updateloss_tmp-updateloss}")
            #     break
            # #
            # updateloss_tmp=updateloss
            progress.set_postfix_str(f"loss={updateloss:.4e}")
            for jj in range(self.TA):
                inv1=torch.inverse(self.mu3*S3.t()@S3+self.mu1*eyep)
                inv2=-1/(onesp.t()@inv1@onesp)
                A=(inv1+inv1@onesp@inv2@onesp.t()@inv1)@(S3.t()@Y+self.mu1*(S1-L1))-inv1@onesp@(inv2*onesn.t())
                S1= (A+L1)
                S1[S1<=0]=0
                L1 = L1 + A - S1
                #print(lossTot(A,B).item())
            for jj in range(self.TB):
                B=(inv3+inv3@onesLibS@inv4@onesLibS.t()@inv3)@(self.mu3*D.t()@(S3-L3)+self.mu2*(S2-L2))-inv3@onesLibS@(inv4*onesp.t())
                S2= (B+L2)
                S2[S2<=0]=0
                S3=(Y@A.t() + self.mu3*(D@B+L3))@torch.inverse(A@A.t()+self.mu3*eyep)
                #S3[S3<=0]=0
                L2 = L2 + B - S2
                L3 = L3 + D@B - S3
                # print(lossTot(A,B).item())
                # inv1=torch.inverse(self.mu3*S3.t()@S3+self.mu1*eyep)
                # inv2=-1/(onesp.t()@inv1@onesp)
                # A=(inv1+inv1@onesp@inv2@onesp.t()@inv1)@(S3.t()@Y+self.mu1*(S1-L1))-inv1@onesp@(inv2*onesn.t())
                # B=(inv3+inv3@onesLibS@inv4@onesLibS.t()@inv3)@(self.mu3*D.t()@(S3-L3)+self.mu2*(S2-L2))-inv3@onesLibS@(inv4*onesp.t())
                # S3=(Y@A.t() + self.mu3*(D@B+L3))@torch.inverse(A@A.t()+self.mu3*eyep)
                # S1= (A+L1)
                # S1[S1<=0]=0
                # S2= (B+L2)
                # S2[S2<=0]=0
                # L1 = L1 + A - S1
                # L2 = L2 + B - S2
                # L3 = L3 + D@B - S3
                # print(lossTot(A,B).item())
                # print(f"Initial loss => {lossTot(A, B)}")
            # if ii % 100 == 0:
              #     print(f"Initial loss => {lossTot(A, B)}")
                # clear_output(wait=False)
            self.running_time = time.time() - tic
        print(f"FASUn took {self.running_time:.1f}s")
        print(f"Final loss => {lossTot(A, B):.2e}")
        E = (D @ B).cpu().numpy()
        A = A.cpu().numpy()
        B = B.cpu().numpy()
        return A, B

