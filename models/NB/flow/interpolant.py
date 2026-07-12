import torch
from models.NB.flow.prob_path import ConstantNoisePath


class Interpolant:
    def __init__(self, conf):
        self.noise_scale = conf["probability_path"]["constant_noise"]
        self.probability_path = ConstantNoisePath(noise_scale=self.noise_scale)

    def interpolate(self, src_exp, tgt_exp, device):
        t = self.sample_t((src_exp.shape[0],)).to(device)
        if src_exp.shape[0] > 1:
            t = t.squeeze(-1)
        exp_t, exp_vf = self.probability_path.sample_path(t, src_exp, tgt_exp)
        return exp_t, exp_vf, t
        
    def denoise(self, exp_t, exp_vf, d_t):
        # exp_1: [B, n_cells, n_genes]
        # exp_t: [B, n_cells, n_genes]
        # t: [B]
        # d_t: [B]
        return exp_t + d_t[:, None, None] * exp_vf

    def sample_t(self, shape):
        return torch.rand(shape)
