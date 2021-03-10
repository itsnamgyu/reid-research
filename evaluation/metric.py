import torch


def compute_distances(f1: torch.Tensor, f2: torch.Tensor, metric="euclidean") -> torch.Tensor:
    assert (f1.shape[1] == f2.shape[1])
    d1, d2 = f1.shape[0], f2.shape[0]

    if metric == "euclidean":
        ff1 = f1.pow(2).sum(1, keepdim=True).expand(d1, d2)
        ff2 = f2.pow(2).sum(1, keepdim=True).t().expand(d1, d2)
        distances = (ff1 + ff2).addmm(mat1=f1, mat2=f2.t(), alpha=-2, beta=1)
    else:
        raise ValueError("{} is an invalid metric".format(metric))

    return distances


def evaluate(distmat: torch.Tensor, qpids: torch.Tensor, gpids: torch.Tensor, qcamids: torch.Tensor,
             gcamids: torch.Tensor, max_rank=50):
    """
    Evaluate match # by rank
    :param distmat:
    :param qpids:
    :param gpids:
    :param qcamids:
    :param gcamids:
    :param max_rank:
    :return:
        all_cmc: percentage of queries matched until each rank r
        ap_list: average precision for each query image (mean for mAP)
        inp_list: inp score for each query image (mean for mINP)
    """
    q, q = distmat.shape

    if q < max_rank:
        max_rank = q
        print("Note: number of gallery samples is quite small, got {}".format(q))
    indices = distmat.argsort(dim=1)
    matches = gpids[indices].eq(qpids.reshape(-1, 1)).int()

    order = indices
    remove = torch.logical_and(gpids[order].eq(qpids.reshape(-1, 1)), gcamids[order].eq(qcamids.reshape(-1, 1)))
    keep = remove.logical_not()
    kept = keep.cumsum(dim=1)

    q, g = len(qpids), len(gpids)

    valid_matches = matches * keep
    valid_query = valid_matches.sum(dim=1).gt(0)  # at least one matchable (== matched) gallery image
    assert (valid_query.all())  # reid dataset queries should all be valid
    assert (valid_matches.sum() != 0)  # error: all query identities do not appear in gallery

    final_rank_positions = (valid_matches * torch.arange(1, g + 1)).argmax(dim=1)
    final_rank_valid = kept[torch.arange(q), final_rank_positions]
    all_INP = valid_matches.sum(dim=1).float() / final_rank_valid.float()

    # `kept` is analogous to index within only-valid instances
    cum_precision = valid_matches.cumsum(dim=1).float() / kept.float()
    cum_precision[cum_precision.isnan()] = 1
    all_AP = (cum_precision * valid_matches).sum(dim=1) / valid_matches.sum(dim=1)

    # Compute CMC (need to go query-by-query) (assume that at least 10% are valid)
    buffer = 10
    keep = keep[:, :max_rank * buffer]
    matches = matches[:, :max_rank * buffer]
    all_cmc = []
    for i in range(q):
        mc = matches[i][keep[i]][:50]
        if len(mc) < max_rank:
            raise AssertionError("Not enough matching galleries. Consider higher `buffer` value.")
        cmc = mc[:max_rank].cumsum(dim=0)
        # E.g., 0 1 x x x x ... to 0 1 1 1 1 1 ...
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)

    all_cmc = torch.stack(all_cmc).float()
    all_cmc = all_cmc.sum(dim=0) / valid_query.float().sum()
    # mAP = all_AP[valid_query].mean()
    # mINP = all_INP[valid_query].mean()

    return all_cmc, all_AP, all_INP

