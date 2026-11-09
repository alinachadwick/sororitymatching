#!/usr/bin/env python3
"""Empirical check: with block-ordered sorority preferences, is sequential DA (block by block, residual quotas)
the same as one DA run on the full lists?  Tests lists built directly, plus lists aggregated from member rankings."""
import argparse
import numpy as np
from rush_distortion import da_rushee_proposing, da_sorority_proposing, AGGREGATORS

DA = {"rushee": da_rushee_proposing, "sorority": da_sorority_proposing}

def sequential(side, rushee_rankings, lists, quota, n_sor, blocks, mixed=False):
    match, quota = [-1] * len(rushee_rankings), list(quota)
    for bi, block in enumerate(blocks):
        idx = {r: i for i, r in enumerate(block)}
        sub_lists = [[idx[r] for r in l if r in idx] for l in lists]
        sub_rr = [rushee_rankings[r] for r in block]
        s = side if not mixed else ("rushee" if bi % 2 == 0 else "sorority")  # alternate sides (should differ sometimes)
        m = DA[s](sub_rr, sub_lists, quota, n_sor, len(block))
        for i, r in enumerate(block):
            match[r] = m[i]
            if m[i] >= 0:
                quota[m[i]] -= 1
    return tuple(match)

def trial(rng, n_sor, n, spots, n_blocks, members, agg):
    perm = rng.permutation(n)
    cuts = sorted(rng.choice(np.arange(1, n), size=n_blocks - 1, replace=False))
    blocks = [list(map(int, b)) for b in np.split(perm, cuts)]
    quota = [spots] * n_sor
    rr = [tuple(rng.permutation(n_sor)) for _ in range(n)]
    member_rankings = [[tuple(x for b in blocks for x in rng.permutation(b)) for _ in range(members)]
                       for _ in range(n_sor)]  # everyone agrees on block order, shuffles within blocks
    if agg == "direct":  # each sorority's list = one member's ranking
        lists = [list(map(int, mr[0])) for mr in member_rankings]
    else:
        tb = list(rng.permutation(n))
        lists = []
        for mr in member_rankings:
            sc = AGGREGATORS[agg]([tuple(map(int, m)) for m in mr], n)
            lists.append(sorted(range(n), key=lambda r: (-sc[r], tb[r])))
    out = {}
    for side in DA:
        full = DA[side](rr, lists, quota, n_sor, n)
        out[side] = (full == sequential(side, rr, lists, quota, n_sor, blocks),
                     full == sequential(side, rr, lists, quota, n_sor, blocks, mixed=True))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sororities", type=int, default=4); p.add_argument("--rushees", type=int, default=20)
    p.add_argument("--spots", type=int, default=3); p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--members", type=int, default=5); p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    print(f"{a.trials} trials, {a.sororities} sororities, {a.rushees} rushees, {a.spots} spots, {a.blocks} blocks")
    print(f"{'lists':<11}{'side':<10}{'same-side seq == full':>24}{'alternating sides == full':>28}")
    for agg in ["direct", "borda", "copeland", "plurality"]:
        res = [trial(rng, a.sororities, a.rushees, a.spots, a.blocks, a.members, agg) for _ in range(a.trials)]
        for side in DA:
            print(f"{agg:<11}{side:<10}{np.mean([r[side][0] for r in res]):>24.1%}{np.mean([r[side][1] for r in res]):>28.1%}")

if __name__ == "__main__":
    main()
