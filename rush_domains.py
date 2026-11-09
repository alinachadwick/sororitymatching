#!/usr/bin/env python3
"""
Cross-domain stability of aggregation rules (ranking -> output ranking) and of the DA matchings they induce.

DOMAINS (what a member reports to the aggregator; true utilities are the same in all three)
  full    every member ranks every rushee
  topk    every member reports only the top k of their true ranking
  random  every member sees a uniformly random k rushees and ranks those

AGGREGATION CONVENTIONS (n = # rushees, k = # reported/seen)
  plurality  top reported rushee gets a vote                       (same in every domain)
  borda      full:   places get n-1, ..., 0
             topk:   top-k get k, ..., 1; unranked get 0
             random: seen get n-1, ..., n-k; each unseen gets the average of the remaining
                     n-k points, (n-k-1)/2   (member's total points match the full domain)
  copeland   score = # pairwise wins + 0.5 * # ties, margins from members who compared the pair
             full:   as usual
             topk:   a ranked rushee beats every unranked one; two unranked rushees are not compared
             random: a pair is compared only by members who saw both

PROFILES  member utility = rho * (sorority's shared Dirichlet) + (1 - rho) * (own Dirichlet), unit-sum;
          rushee utilities are built the same way over sororities. rho = 0 is i.i.d. members.

Example
  python3 rush_domains.py --sororities 4 --members 6 --rushees 20 --spots 3 --k 5 --rho 0.5 --trials 200
"""
import argparse, itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from rush_distortion import da_rushee_proposing, da_sorority_proposing

DOMAINS = ["full", "topk", "random"]
RULES = ["plurality", "borda", "copeland"]

# ----------------------------------------------------------------------------------------
# Aggregation: (member views, n_rushees, domain) -> scores (higher = better)
# A member view is a tuple of rushees, best first: the full ranking, its top-k prefix, or the ranking of a sample.
# ----------------------------------------------------------------------------------------

def agg_plurality(views, n, domain):
    scores = [0.0] * n
    for v in views:
        scores[v[0]] += 1
    return scores


def agg_borda(views, n, domain):
    scores = [0.0] * n
    for v in views:
        k = len(v)
        if domain == "topk":
            for place, r in enumerate(v):
                scores[r] += k - place
        else:  # full (k = n) and random
            for place, r in enumerate(v):
                scores[r] += n - 1 - place
            unseen_avg = (n - k - 1) / 2 if k < n else 0
            for r in set(range(n)) - set(v):
                scores[r] += unseen_avg
    return scores


def agg_copeland(views, n, domain):
    margin = np.zeros((n, n))
    compared = np.zeros((n, n), dtype=bool)
    for v in views:
        for place, r in enumerate(v):
            below = list(v[place + 1:])
            if domain == "topk":
                below += [x for x in range(n) if x not in v]  # ranked beats unranked
            for r2 in below:
                margin[r][r2] += 1
                margin[r2][r] -= 1
                compared[r][r2] = compared[r2][r] = True
    wins = (margin > 0) * 1.0 + (margin == 0) * 0.5
    return list((wins * compared).sum(axis=1))


AGGS = {"plurality": agg_plurality, "borda": agg_borda, "copeland": agg_copeland}


def output_ranking(scores, tiebreak):
    """tiebreak[r] = priority of rushee r (lower wins); shared across domains so ties don't add noise."""
    return sorted(range(len(scores)), key=lambda r: (-scores[r], tiebreak[r]))

# ----------------------------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------------------------

def mixed_dirichlet(rng, shared, size, alpha, rho):
    return rho * shared + (1 - rho) * rng.dirichlet([alpha] * size)


def sample_instance(rng, n_sor, members, n, alpha, rho, k):
    """Returns member_views[domain][s] = list of views, rushee_rankings, member_utils, rushee_utils."""
    views = {d: [] for d in DOMAINS}
    member_utils = []
    for s in range(n_sor):
        shared = rng.dirichlet([alpha] * n)
        us = [mixed_dirichlet(rng, shared, n, alpha, rho) for _ in range(members[s])]
        member_utils.append(np.array(us))
        full = [tuple(np.argsort(-u, kind="stable")) for u in us]
        views["full"].append(full)
        views["topk"].append([f[:k] for f in full])
        views["random"].append([tuple(sorted(rng.choice(n, size=k, replace=False), key=lambda r: -u[r]))
                                for u in us])
    shared = rng.dirichlet([alpha] * n_sor)
    rushee_utils = np.array([mixed_dirichlet(rng, shared, n_sor, alpha, rho) for _ in range(n)])
    rushee_rankings = [tuple(np.argsort(-u, kind="stable")) for u in rushee_utils]
    return views, rushee_rankings, member_utils, rushee_utils

# ----------------------------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------------------------

def kendall_dist(a, b):
    """Normalized Kendall-tau distance between two rankings of the same items (0 = same, 1 = reversed)."""
    pos = {r: i for i, r in enumerate(b)}
    seq = [pos[r] for r in a]
    inv = sum(1 for i, j in itertools.combinations(range(len(seq)), 2) if seq[i] > seq[j])
    return inv / (len(seq) * (len(seq) - 1) / 2)


def top_overlap(a, b, q):
    """1 - |top-q(a) & top-q(b)| / q : fraction of a sorority's top-q slots that differ."""
    return 1 - len(set(a[:q]) & set(b[:q])) / q


def matching_dist(m1, m2):
    """Fraction of rushees matched to a different sorority."""
    return sum(x != y for x, y in zip(m1, m2)) / len(m1)


def welfare(matching, member_utils, rushee_utils, which):
    total = 0.0
    for r, s in enumerate(matching):
        if s < 0:
            continue
        if which in ("sor", "both"):
            total += member_utils[s][:, r].sum()
        if which in ("rush", "both"):
            total += rushee_utils[r][s]
    return total

def optimal_welfare(member_utils, rushee_utils, quota, which="both"):
    """Max welfare over all feasible matchings (assignment problem; each sorority s gets quota[s] slots)."""
    n, n_sor = rushee_utils.shape
    value = np.zeros((n, n_sor))  # value[r][s] = welfare contributed by matching rushee r to sorority s
    for s in range(n_sor):
        if which in ("sor", "both"):
            value[:, s] += member_utils[s].sum(axis=0)
        if which in ("rush", "both"):
            value[:, s] += rushee_utils[:, s]
    slot_sor = [s for s in range(n_sor) for _ in range(quota[s])]
    rows, cols = linear_sum_assignment(value[:, slot_sor], maximize=True)
    return value[:, slot_sor][rows, cols].sum()

# ----------------------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------------------

def run_trial(rng, args, members, quota):
    n, n_sor = args.rushees, args.sororities
    views, rushee_rankings, member_utils, rushee_utils = sample_instance(
        rng, n_sor, members, n, args.alpha, args.rho, args.k)
    tiebreak = list(rng.permutation(n))
    out = {}   # out[(rule, domain)] = list of per-sorority output rankings
    for rule in RULES:
        for d in DOMAINS:
            out[(rule, d)] = [output_ranking(AGGS[rule](views[d][s], n, d), tiebreak) for s in range(n_sor)]
    da = {"rushee": da_rushee_proposing, "sorority": da_sorority_proposing}
    match = {(side, rule, d): da[side](rushee_rankings, out[(rule, d)], quota, n_sor, n)
             for side in da for rule in RULES for d in DOMAINS}

    res = {}
    for rule in RULES:
        for d in ("topk", "random"):
            lists, base = out[(rule, d)], out[(rule, "full")]
            res[("tau", rule, d)] = np.mean([kendall_dist(lists[s], base[s]) for s in range(n_sor)])
            res[("top", rule, d)] = np.mean([top_overlap(lists[s], base[s], quota[s]) for s in range(n_sor)])
            for side in da:
                m, m0 = match[(side, rule, d)], match[(side, rule, "full")]
                res[("match", side, rule, d)] = matching_dist(m, m0)
                w, w0 = welfare(m, member_utils, rushee_utils, "both"), welfare(m0, member_utils, rushee_utils, "both")
                res[("welfare", side, rule, d)] = w / w0  # welfare relative to the same rule under full information
    opt = optimal_welfare(member_utils, rushee_utils, quota)
    for side in da:  # realized welfare vs the true optimum (ratio >= 1; 1 = optimal)
        for rule in RULES:
            for d in DOMAINS:
                res[("opt", side, rule, d)] = opt / welfare(match[(side, rule, d)], member_utils, rushee_utils, "both")
    for d in DOMAINS:  # rule-vs-rule within a domain
        for r1, r2 in itertools.combinations(RULES, 2):
            res[("rules", r1, r2, d)] = np.mean([kendall_dist(out[(r1, d)][s], out[(r2, d)][s]) for s in range(n_sor)])
            res[("rulematch", r1, r2, d)] = matching_dist(match[("rushee", r1, d)], match[("rushee", r2, d)])
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sororities", type=int, default=4)
    p.add_argument("--members", type=int, default=6, help="members per sorority")
    p.add_argument("--rushees", type=int, default=20)
    p.add_argument("--spots", type=int, default=3, help="quota per sorority")
    p.add_argument("--k", type=int, default=5, help="top-k size / random sample size")
    p.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration")
    p.add_argument("--rho", type=float, default=0.5, help="weight of shared component in utilities (0 = iid)")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    members, quota = [args.members] * args.sororities, [args.spots] * args.sororities
    rng = np.random.default_rng(args.seed)
    runs = [run_trial(rng, args, members, quota) for _ in range(args.trials)]
    avg = {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}
    worst = {key: float(np.max([r[key] for r in runs])) for key in runs[0] if key[0] == "opt"}

    print(f"n_sor={args.sororities} members={args.members} rushees={args.rushees} spots={args.spots} "
          f"k={args.k} alpha={args.alpha} rho={args.rho} trials={args.trials}\n")
    print("Q1  output-ranking distance to the full-information output of the SAME rule")
    print(f"{'rule':<10}{'domain':<8}{'kendall-tau':>12}{'top-quota diff':>16}")
    for rule in RULES:
        for d in ("topk", "random"):
            print(f"{rule:<10}{d:<8}{avg[('tau', rule, d)]:>12.3f}{avg[('top', rule, d)]:>16.3f}")
    print("\nQ2  DA matching vs the same rule under full information: frac of rushees matched differently"
          " | welfare ratio (both sides, 1 = same)")
    print(f"{'proposing':<10}{'rule':<10}{'domain':<8}{'match diff':>12}{'welfare ratio':>15}")
    for side in ("rushee", "sorority"):
        for rule in RULES:
            for d in ("topk", "random"):
                print(f"{side:<10}{rule:<10}{d:<8}{avg[('match', side, rule, d)]:>12.3f}"
                      f"{avg[('welfare', side, rule, d)]:>15.3f}")
    print("\nQ3  realized welfare vs the TRUE optimum (both sides): ratio opt / achieved, 1 = optimal"
          f" | mean and worst case over {args.trials} trials")
    print(f"{'proposing':<10}{'rule':<10}{'domain':<8}{'mean':>10}{'worst':>10}")
    for side in ("rushee", "sorority"):
        for rule in RULES:
            for d in DOMAINS:
                print(f"{side:<10}{rule:<10}{d:<8}{avg[('opt', side, rule, d)]:>10.3f}{worst[('opt', side, rule, d)]:>10.3f}")
    print("\nRule-vs-rule within a domain: Kendall-tau distance of output rankings | rushee-proposing matching diff")
    print(f"{'pair':<22}" + "".join(f"{d:>16}" for d in DOMAINS))
    for r1, r2 in itertools.combinations(RULES, 2):
        cells = "".join(f"{avg[('rules', r1, r2, d)]:>9.3f} |{avg[('rulematch', r1, r2, d)]:>5.2f}" for d in DOMAINS)
        print(f"{r1 + ' vs ' + r2:<22}{cells}")


if __name__ == "__main__":
    main()
