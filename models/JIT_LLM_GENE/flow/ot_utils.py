import torch
import ot
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from models.JIT_LLM_GENE.flow.ot_utils import batch_ot_plan_from_features

# class ConstantNoisePath:
#     def __init__(self, noise_scale):
#         self.noise_scale = noise_scale
#         self.path = AffineProbPath(scheduler=CondOTScheduler())
        
#     def sample_path(self, t, x0, x1):
#         # AffineProbPath.sample might restrict t to be 1D via assert_sample_shape.
#         # If t is [B, N], we flatten it to [B*N], and also flatten x0/x1 temporarily.
        
#         orig_shape = x0.shape # [B, N, D]
        
#         if t.ndim > 1:
#             t_flat = t.reshape(-1) # [B*N]
#             x0_flat = x0.reshape(-1, x0.shape[-1]) # [B*N, D]
#             x1_flat = x1.reshape(-1, x1.shape[-1]) # [B*N, D]
            
#             path_sample = self.path.sample(t=t_flat, x_0=x0_flat, x_1=x1_flat)
            
#             # Reshape back to [B, N, D]
#             xt = path_sample.x_t.reshape(orig_shape)
#             ut = path_sample.dx_t.reshape(orig_shape)
            
#             xt = xt + self.noise_scale * torch.rand_like(x0)
#             return xt, ut
#         else:
#             path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)
#             xt = path_sample.x_t + self.noise_scale * torch.rand_like(x0)     
#             ut = path_sample.dx_t
#             return xt, ut

class ConstantNoisePath:
    def __init__(self, noise_scale, if_OT=False):
        self.noise_scale = noise_scale
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.if_OT = if_OT

    def sample_path(self, t, x0, x1):
        # AffineProbPath.sample might restrict t to be 1D via assert_sample_shape.
        # If t is [B, N], we flatten it to [B*N], and also flatten x0/x1 temporarily.
        if self.if_OT:
            P, match_idx, C = batch_ot_plan_from_features(
            x0, x1, eps=0.05, iters=50, cost="cosine", return_hard_match=True)
            
            idx = match_idx.unsqueeze(-1).expand(-1, -1, x1.size(-1))  # [B, N, M]
            x1 = torch.gather(x1, dim=1, index=idx)     
            

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