import torch
from scvi.distributions import ZeroInflatedNegativeBinomial
from models.PRIOR_LLM.flow.prob_path import ConstantNoisePath


class Interpolant:
    def __init__(self, conf):

        self.noise_scale = conf["probability_path"]["constant_noise"]
        self.prior_sample_type = conf["prior_distribution"]
        self.prior_kwargs = conf["prior_kwargs"]
        self.probability_path = ConstantNoisePath(noise_scale=self.noise_scale, if_OT=conf["probability_path"]["if_OT"])
        self.prior = PriorSampler(prior_sample_type=self.prior_sample_type, **self.prior_kwargs)
    

    def corrupt(self, exp, device):
        # exp: [B, n_cells, n_genes]
        t = self.sample_t((exp.shape[0],)).to(device)
        if exp.shape[0] > 1:
            t = t.squeeze(-1)
        exp_0 = self.prior.sample(exp.shape).to(device)
        exp_t, exp_vf = self.probability_path.sample_path(t, exp_0, exp)
        return exp_t, exp_vf, t

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

class PriorSampler:
    def __init__(self, prior_sample_type, **kwargs):
        self.prior_sample_type = prior_sample_type

        if prior_sample_type == "gaussian":
            self.prior_sampler = gaussian_prior
        elif prior_sample_type == "zero":
            self.prior_sampler = all_zeros
        elif prior_sample_type == "zinb":
            # https://github.com/scverse/scvi-tools/blob/main/src/scvi/distributions/_negative_binomial.py#L433
            

            prior_sampler = ZeroInflatedNegativeBinomial(
                                        total_count=kwargs.get("total_count", 1),
                                        logits=kwargs.get("logits", 0.1),
                                        zi_logits=kwargs.get("zi_logits", 0),  # real number
                                    )
            self.prior_sampler = lambda shape: prior_sampler.sample(shape).squeeze(-1)
        else:
            raise ValueError("Invalid prior sample type")

    def sample(self, shape):
        return self.prior_sampler(shape)


def gaussian_prior(shape):
            return torch.randn(shape)


def all_zeros(shape):
            return torch.zeros(shape)