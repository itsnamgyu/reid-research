import torch


def compute_distances(f1: torch.Tensor, f2: torch.Tensor, metric="euclidean") -> torch.Tensor:
    assert(f1.shape[1] == f2.shape[1])
    d1, d2 = f1.shape[0], f2.shape[0]

    if metric == "euclidean":
        ff1 = f1.pow(2).sum(1, keepdim=True).expand(d1, d2)
        ff2 = f2.pow(2).sum(1, keepdim=True).t().expand(d1, d2)
        distances = (ff1 + ff2).addmm(mat1=f1, mat2=f2.t(), alpha=-2, beta=1)
    else:
        raise ValueError("{} is an invalid metric".format(metric))

    return distances

