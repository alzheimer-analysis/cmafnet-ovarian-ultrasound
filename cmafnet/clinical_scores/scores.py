import math


def roma_score(he4, ca125, postmenopausal):
    if postmenopausal >= 0.5:
        a = -8.09 + 0.866 * math.log2(he4 + 1e-6) + 0.843 * math.log2(ca125 + 1e-6)
    else:
        a = -12.0 + 0.785 * math.log2(he4 + 1e-6) + 0.731 * math.log2(ca125 + 1e-6)
    return 1.0 / (1.0 + math.exp(-a))


def orads_to_pathology_prior(orads_value):
    mapping = {
        2: [0.85, 0.05, 0.10],
        3: [0.55, 0.15, 0.30],
        4: [0.15, 0.45, 0.40],
        5: [0.05, 0.70, 0.25],
    }
    try:
        v = int(float(orads_value))
    except (TypeError, ValueError):
        return [1 / 3, 1 / 3, 1 / 3]
    return mapping.get(v, [1 / 3, 1 / 3, 1 / 3])


def roma_to_pathology_prior(roma, postmenopausal):
    if postmenopausal >= 0.5:
        t_mal = 0.35
    else:
        t_mal = 0.25
    if roma >= t_mal:
        return [0.10, 0.75, 0.15]
    if roma >= t_mal * 0.6:
        return [0.25, 0.35, 0.40]
    return [0.80, 0.10, 0.10]
