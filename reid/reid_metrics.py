import numpy as np
from scipy.spatial.distance import cdist


def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

    all_cmc, all_AP = [], []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
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
        raise RuntimeError('No valid query for evaluation. Please check your data splits.')

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def compute_reid_metrics(features, labels):
    labels = np.asarray(labels)
    num_samples = labels.shape[0]
    if num_samples == 0:
        raise RuntimeError('No features were provided for evaluation.')

    # group indices by label
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    query_indices = []
    gallery_indices = []
    for indices in label_to_indices.values():
        query_indices.append(indices[0])
        gallery_indices.extend(indices)

    q_features = features[query_indices]
    g_features = features[gallery_indices]

    q_labels = labels[query_indices]
    g_labels = labels[gallery_indices]

    q_camids = np.zeros(len(query_indices), dtype=int)
    g_camids = np.ones(len(gallery_indices), dtype=int)

    distmat = cdist(q_features, g_features, metric='cosine')
    cmc, mAP = evaluate_rank(distmat, q_labels, g_labels, q_camids, g_camids)

    rank1 = float(cmc[0]) if cmc.size > 0 else 0.0
    rank5_index = min(4, cmc.size - 1) if cmc.size > 0 else 0
    rank5 = float(cmc[rank5_index]) if cmc.size > 0 else 0.0

    return {'rank1': rank1, 'rank5': rank5, 'mAP': float(mAP)}
