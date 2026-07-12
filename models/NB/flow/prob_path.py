import torch
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

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
