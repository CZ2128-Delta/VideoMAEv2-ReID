import numpy as np
from scipy.spatial.distance import cdist


def evaluate_rank(distmat, q_pids, g_pids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

    all_cmc, all_AP = [], []
    num_valid_q = 0

    for q_idx in range(num_q):
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1)
        AP = (precision * orig_cmc).sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        raise RuntimeError('No valid query for evaluation. Please ensure each identity has at least one gallery sample.')

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def compute_reid_metrics(query_features,
                         query_labels,
                         gallery_features,
                         gallery_labels):
    if query_features is None or gallery_features is None:
        return {'rank1': 0.0, 'rank5': 0.0, 'mAP': 0.0}
    if len(query_features) == 0 or len(gallery_features) == 0:
        return {'rank1': 0.0, 'rank5': 0.0, 'mAP': 0.0}

    q_labels = np.asarray(query_labels)
    g_labels = np.asarray(gallery_labels)

    distmat = cdist(query_features, gallery_features, metric='cosine')
    cmc, mAP = evaluate_rank(distmat, q_labels, g_labels)

    rank1 = float(cmc[0]) if cmc.size > 0 else 0.0
    rank5_index = min(4, cmc.size - 1) if cmc.size > 0 else 0
    rank5 = float(cmc[rank5_index]) if cmc.size > 0 else 0.0

    return {'rank1': rank1, 'rank5': rank5, 'mAP': float(mAP)}
