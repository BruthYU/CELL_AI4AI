import torch
from typing import Tuple, Optional

@torch.no_grad()
def sinkhorn_log_domain(
    C: torch.Tensor,
    eps: float = 0.05,
    iters: int = 50,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Solve entropic OT via log-domain Sinkhorn (batched).

    Args:
        C:    [B, N, M] cost matrix
        eps:  entropic regularization strength
        iters: sinkhorn iterations
        a:    [B, N] source weights (sum to 1). If None -> uniform.
        b:    [B, M] target weights (sum to 1). If None -> uniform.

    Returns:
        P: [B, N, M] transport plan, rows sum to a, cols sum to b
    """
    assert C.dim() == 3, "C must be [B, N, M]"
    B, N, M = C.shape
    device = C.device
    dtype = C.dtype

    if a is None:
        a = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    if b is None:
        b = torch.full((B, M), 1.0 / M, device=device, dtype=dtype)

    # logK = -C/eps
    logK = -C / eps  # [B, N, M]

    # Initialize dual variables in log-space
    log_u = torch.zeros((B, N), device=device, dtype=dtype)
    log_v = torch.zeros((B, M), device=device, dtype=dtype)

    log_a = torch.log(a + 1e-12)
    log_b = torch.log(b + 1e-12)

    for _ in range(iters):
        # log_u = log_a - logsumexp(logK + log_v, dim=-1)
        log_u = log_a - torch.logsumexp(logK + log_v.unsqueeze(1), dim=-1)
        # log_v = log_b - logsumexp(logK^T + log_u, dim=-1)
        log_v = log_b - torch.logsumexp(logK.transpose(1, 2) + log_u.unsqueeze(1), dim=-1)

    # P = diag(u) K diag(v) in log space: logP = log_u + logK + log_v
    logP = log_u.unsqueeze(-1) + logK + log_v.unsqueeze(1)
    P = torch.exp(logP)  # [B, N, M]
    return P


def batch_ot_plan_from_features(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    iters: int = 50,
    cost: str = "cosine",
    return_hard_match: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Batched OT between token sets (x and y) per sample.

    Args:
        x: [B, N, D] (here N=512, D=512)
        y: [B, M, D] (here M=512, D=512)
        eps/iters: sinkhorn params
        cost: "sqeuclidean" or "euclidean" or "cosine"
        return_hard_match: whether to return argmax matching indices
    Returns:
        P: [B, N, M] OT plan
        match_idx (optional): [B, N] hard match (each x_i -> y_{idx})
        C: [B, N, M] cost matrix
    """
    assert x.dim() == 3 and y.dim() == 3
    assert x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2]
    B, N, D = x.shape
    _, M, _ = y.shape
    device = x.device

    if cost in ("sqeuclidean", "euclidean"):
        # torch.cdist gives euclidean distance; square if needed
        C = torch.cdist(x, y, p=2)  # [B, N, M]
        if cost == "sqeuclidean":
            C = C * C
    elif cost == "cosine":
        # C = 1 - cosine similarity
        x_n = torch.nn.functional.normalize(x, dim=-1)
        y_n = torch.nn.functional.normalize(y, dim=-1)
        C = 1.0 - torch.einsum("bnd,bmd->bnm", x_n, y_n).clamp(-1, 1)
    else:
        raise ValueError(f"Unknown cost: {cost}")

    P = sinkhorn_log_domain(C, eps=eps, iters=iters)

    match_idx = None
    if return_hard_match:
        # simple hard match: pick max mass in each row
        match_idx = P.argmax(dim=-1)  # [B, N]

    return P, match_idx, C


# ------------------ Example ------------------
if __name__ == "__main__":
    B = 32
    x = torch.randn(B, 512, 512, device="cuda", dtype=torch.float16)
    y = torch.randn(B, 512, 512, device="cuda", dtype=torch.float16)

    

    P, match_idx, C = batch_ot_plan_from_features(
        x, y, eps=0.05, iters=50, cost="sqeuclidean", return_hard_match=True
    )
    print(P.shape, match_idx.shape, C.shape)  # [B,512,512], [B,512], [B,512,512]


    idx = match_idx.unsqueeze(-1).expand(-1, -1, y.size(-1))  # [B, 512, 512]
    y_aligned = torch.gather(y, dim=1, index=idx)             # [B, 512, 512]
    pass