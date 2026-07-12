import torch
import ot
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
def gpu_cdist(
    X: torch.Tensor,
    Y: torch.Tensor,
    squared: bool = False,
) -> torch.Tensor:
    """
    X: [B, N, D] (CUDA tensor)
    Y: [B, M, D] (CUDA tensor)
    return: [B, N, M]，每个 batch 内 X 与 Y 的 pairwise 距离矩阵
    """
    assert X.ndim == 3 and Y.ndim == 3, f"X, Y must be [B, N, D] / [B, M, D], got {X.shape}, {Y.shape}"
    dist = torch.cdist(X, Y, p=2)  # [B, N, M]

    if squared:
        return dist * dist
    return dist
import torch




class Match_Linear_POT:
    def __init__(self, x, y):
        """
        Args:
            x: Source points, (B, N, D)
            y: Target points, (B, M, D)
            C: [B, N, M]

        """
        assert x.device == y.device, "x, y must be on the same device"
        N, M = x.shape[1], y.shape[1]
        self.a = (torch.ones(N) / N).to(x.device)
        self.b = (torch.ones(M) / M).to(y.device)
        self.Cs = gpu_cdist(x, y, squared=True)
        max_val, _ = self.Cs.max(dim=0, keepdim=True)
        self.Cs = self.Cs / max_val 


    def sample_joint_sinkhorn(self, reg=1e-1, numItermax=1000):
        """Solve entropy-regularized OT using Sinkhorn algorithm."""
        gammas = []
        for i in range(len(self.Cs)):
                gammas.append(ot.sinkhorn(self.a, self.b, self.Cs[i], reg=reg, numItermax=numItermax))
        gamma = torch.stack(gammas)
        return  self.sample_joint_from_gamma(gamma)


    def sample_joint_from_gamma(self, gamma: torch.Tensor, generator=None):
        """
        gamma: [B, N, M]
        返回:
        src_ixs: [B, N]
        tgt_ixs: [B, N]
        """
        b, n, m = gamma.shape

        # 展平成 [B, N*M]
        flat = gamma.view(b, -1)

        # 沿着最后一维做归一化，每个 batch 一条分布
        flat_sum = flat.sum(dim=1, keepdim=True)  # [B, 1]
        eps = 1e-12
        flat = flat / (flat_sum + eps)            # [B, N*M]

        # batched multinomial:
        # input: [B, N*M] -> output: [B, num_samples]
        indices = torch.multinomial(
            flat,
            num_samples=n,
            replacement=True,
            generator=generator,
        )  # [B, N]

        # 映射回 (src, tgt) 下标
        src_ixs = indices // m   # [B, N]
        tgt_ixs = indices % m    # [B, N]

        return src_ixs, tgt_ixs

class ConstantNoisePath:
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale
        self.path = AffineProbPath(scheduler=CondOTScheduler())
    def sample_path(self, t, x0, x1):
        # AffineProbPath.sample might restrict t to be 1D via assert_sample_shape.
        # If t is [B, N], we flatten it to [B*N], and also flatten x0/x1 temporarily.
        
        orig_shape = x0.shape # [B, N, D]
        
        if t.ndim > 1:
            t_flat = t.reshape(-1) # [B*N]
            x0_flat = x0.reshape(-1, x0.shape[-1]) # [B*N, D]
            x1_flat = x1.reshape(-1, x1.shape[-1]) # [B*N, D]
            
            path_sample = self.path.sample(t=t_flat, x_0=x0_flat, x_1=x1_flat)
            
            # Reshape back to [B, N, D]
            xt = path_sample.x_t.reshape(orig_shape)
            ut = path_sample.dx_t.reshape(orig_shape)
            
            xt = xt + self.noise_scale * torch.rand_like(x0)
            return xt, ut
        else:
            path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
            xt = path_sample.x_t + self.noise_scale * torch.rand_like(x0)     
            ut = path_sample.dx_t
            return xt, ut