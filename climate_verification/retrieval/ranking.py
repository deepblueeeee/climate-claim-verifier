"""Candidate fusion and evidence selection."""


def unique_union(dense_rank, bm25_rank, dense_top_k, bm25_top_k, output_top_k):
    candidates = (
        list(dense_rank[:dense_top_k]) + list(bm25_rank[:bm25_top_k]) + list(dense_rank)
    )
    output = []
    seen = set()
    for pos in candidates:
        if pos not in seen:
            output.append(pos)
            seen.add(pos)
        if len(output) >= output_top_k:
            break
    return output


def select_by_score_gap(ranked, max_evidences=5):
    if not ranked:
        return []
    if len(ranked) == 1:
        return ranked[:1]

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    scan_n = min(len(ranked), max_evidences + 1)
    scores = [float(score) for _, score in ranked[:scan_n]]
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    if not gaps:
        return ranked[:1]

    best_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    cutoff = max(1, min(best_gap_idx + 1, max_evidences))
    return ranked[:cutoff]
