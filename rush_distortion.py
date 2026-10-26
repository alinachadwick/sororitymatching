#!/usr/bin/env python3
"""
Sorority rush: aggregate member rankings -> deferred acceptance, with welfare + distortion.

PIPELINE
  1. Members of each sorority hold rankings over the rushees THEY MEET (visibility mask).
  2. An aggregation rule turns member rankings into one preference list per sorority.
  3. DA runs (rushee-proposing and/or sorority-proposing) on [aggregated sorority lists,
     rushee rankings over sororities].
  4. We report, per proposing side and per welfare type (sor / rush / both):
       - realized welfare + realized ratio (opt / achieved)   [sample mode, true utilities known]
       - worst-case LP distortion over all utilities consistent with the ordinal profile,
         against two benchmarks:  'all'    = best feasible matching (ignores stability)
                                  'stable' = best matching stable w.r.t. aggregated prefs

MODES
  --search sample      draw random cardinal utilities -> rankings -> everything (default)
  --search exhaustive  enumerate ALL ordinal profiles (full visibility only; tiny instances)

NORMALISATION  unit-sum (each agent's utilities sum to 1), the usual convention in the
distortion literature. --floor EPS additionally forces every utility >= EPS (0 = allow zeros,
which makes many distortions infinite).

NOTATION (used throughout the code)
  n_sor, n_rushees      number of sororities / rushees
  members_per_sor[s]    number of members in sorority s
  quota[s]              number of spots sorority s can fill
  matching              tuple of length n_rushees; matching[r] = sorority of rushee r, -1 = unmatched
  member_rankings[s]    list of rankings, one per member of sorority s (best rushee first)
  rushee_rankings[r]    rushee r's ranking of all sororities (best first)
  sor_lists[s]          sorority s's aggregated list of acceptable rushees (best first)
  utils                 one flat vector holding every agent's utilities plus the LP scale variable
                        (see Instance for the layout)

EXTENDING
  * New aggregation rule: write f(rankings, n_rushees) -> list of n_rushees scores (higher = better)
    and add it to AGGREGATORS. `rankings` is a list of tuples, each a member's ranking of the
    rushees that member met (best first). Unmet rushees simply do not appear.
  * Partial visibility (sample mode), pick at most one of:
      --meet-prob p   each member meets each rushee independently w.p. p (at least --min-meet rushees)
      --meet-count k  each member meets exactly k rushees, chosen uniformly at random
    With neither, every member meets every rushee. Replace `draw_visibility` for structured patterns
    (e.g. each member meets a fixed block of rushees). Utilities over unmet rushees are
    unconstrained in the LP (still unit-sum).
  * --truncate controls which rushees a sorority finds unacceptable after aggregation.

Example
  python3 rush_distortion.py --sororities 2 --members 3 --rushees 2 --spots 1 \
      --proposing both --agg plurality --search exhaustive
  python3 rush_distortion.py --sororities 3 --members 4 --rushees 5 --spots 2 \
      --proposing both --agg borda --meet-prob 0.7 --samples 200 --floor 0.01
"""
import argparse, itertools, math, sys, time
import numpy as np
from scipy.optimize import linprog

INF = math.inf
WELFARE_TYPES = ["sor", "rush", "both"]

# ----------------------------------------------------------------------------------------
# Aggregation rules:  (list of member rankings, n_rushees) -> scores (higher = better)
# ----------------------------------------------------------------------------------------

def agg_plurality(rankings, n_rushees):
    scores = [0.0] * n_rushees
    for ranking in rankings:
        if ranking:
            scores[ranking[0]] += 1
    return scores


def agg_borda(rankings, n_rushees):
    """Each member gives n_met points to their top visible rushee, ..., 1 to their last; 0 if unmet."""
    scores = [0.0] * n_rushees
    for ranking in rankings:
        n_met = len(ranking)
        for place, rushee in enumerate(ranking):
            scores[rushee] += n_met - place
    return scores


def agg_copeland(rankings, n_rushees):
    """Score = (# rushees beaten head-to-head) + 0.5 * (# ties).  A head-to-head between r and r'
    uses only the members who met both; a pair no member compared counts as nothing (not a tie)."""
    margin = [[0] * n_rushees for _ in range(n_rushees)]  # margin[r][r2] = #members preferring r to r2, minus the reverse
    compared = [[False] * n_rushees for _ in range(n_rushees)]
    for ranking in rankings:
        for place, r in enumerate(ranking):
            for r2 in ranking[place + 1:]:  # this member ranks r above r2
                margin[r][r2] += 1
                margin[r2][r] -= 1
                compared[r][r2] = compared[r2][r] = True
    scores = [0.0] * n_rushees
    for r in range(n_rushees):
        for r2 in range(n_rushees):
            if r != r2 and compared[r][r2]:
                scores[r] += 1.0 if margin[r][r2] > 0 else 0.5 if margin[r][r2] == 0 else 0.0
    return scores


AGGREGATORS = {"plurality": agg_plurality, "borda": agg_borda, "copeland": agg_copeland}


def sorority_lists(member_rankings, agg, n_rushees, truncate):
    """member_rankings[s] = list of member rankings. Returns acceptable-rushee list per sorority."""
    sor_lists = []
    for rankings in member_rankings:
        scores = AGGREGATORS[agg](rankings, n_rushees)
        met = {r for ranking in rankings for r in ranking}
        order = sorted(range(n_rushees), key=lambda r: (-scores[r], r))  # ties: lowest index
        if truncate == "unseen":
            order = [r for r in order if r in met]
        elif truncate == "zero":
            order = [r for r in order if scores[r] > 0]
        sor_lists.append(order)
    return sor_lists

# ----------------------------------------------------------------------------------------
# Deferred acceptance (supports incomplete sorority lists; rushees rank all sororities)
# ----------------------------------------------------------------------------------------

def da_rushee_proposing(rushee_rankings, sor_lists, quota, n_sor, n_rushees):
    # sor_rank[s][r] = position of rushee r in sorority s's list (lower = better; absent = unacceptable)
    sor_rank = [{r: place for place, r in enumerate(lst)} for lst in sor_lists]
    next_choice = [0] * n_rushees              # index into each rushee's ranking of the next sorority to try
    held_by_sor = [[] for _ in range(n_sor)]  # rushees each sorority is currently holding
    free_rushees = list(range(n_rushees))
    while free_rushees:
        r = free_rushees.pop()
        if next_choice[r] >= n_sor:
            continue  # r has been rejected everywhere
        s = rushee_rankings[r][next_choice[r]]
        next_choice[r] += 1
        if r not in sor_rank[s] or quota[s] == 0:
            free_rushees.append(r)
            continue
        held_by_sor[s].append(r)
        if len(held_by_sor[s]) > quota[s]:
            rejected = max(held_by_sor[s], key=lambda x: sor_rank[s][x])
            held_by_sor[s].remove(rejected)
            free_rushees.append(rejected)
    matching = [-1] * n_rushees
    for s in range(n_sor):
        for r in held_by_sor[s]:
            matching[r] = s
    return tuple(matching)


def da_sorority_proposing(rushee_rankings, sor_lists, quota, n_sor, n_rushees):
    # rushee_rank[r][s] = position of sorority s in rushee r's ranking (lower = better)
    rushee_rank = [{s: place for place, s in enumerate(rushee_rankings[r])} for r in range(n_rushees)]
    next_choice = [0] * n_sor                  # index into each sorority's list of the next rushee to try
    held_by_rushee = [None] * n_rushees        # sorority currently holding each rushee
    n_held = [0] * n_sor                       # how many rushees each sorority currently holds
    queue = list(range(n_sor))                 # sororities with spots left to fill
    while queue:
        s = queue.pop()
        while n_held[s] < quota[s] and next_choice[s] < len(sor_lists[s]):
            r = sor_lists[s][next_choice[s]]
            next_choice[s] += 1
            current = held_by_rushee[r]
            if current is None:
                held_by_rushee[r] = s
                n_held[s] += 1
            elif rushee_rank[r][s] < rushee_rank[r][current]:  # r prefers s: trade up
                held_by_rushee[r] = s
                n_held[s] += 1
                n_held[current] -= 1
                queue.append(current)
    return tuple(-1 if s is None else s for s in held_by_rushee)


def is_stable(matching, rushee_rankings, sor_lists, quota, n_sor, n_rushees):
    sor_rank = [{r: place for place, r in enumerate(lst)} for lst in sor_lists]
    for r in range(n_rushees):
        if matching[r] >= 0 and r not in sor_rank[matching[r]]:
            return False  # individual rationality
    for r in range(n_rushees):
        for s in rushee_rankings[r]:
            if s == matching[r]:
                break  # only sororities r prefers to her current one can block
            if r not in sor_rank[s] or quota[s] == 0:
                continue
            admitted = [x for x in range(n_rushees) if matching[x] == s]
            if len(admitted) < quota[s]:
                return False  # s has an empty spot and r wants it
            worst_admitted = max(admitted, key=lambda x: sor_rank[s][x])
            if sor_rank[s][r] < sor_rank[s][worst_admitted]:
                return False  # s prefers r to someone it admitted
    return True

# ----------------------------------------------------------------------------------------
# Instance + LP
# ----------------------------------------------------------------------------------------

class Instance:
    """Fixed problem structure (sizes, quotas) plus the LP built on top of it.

    LP variables live in one flat vector `utils`:
        [ member utilities | rushee utilities | scale ]
      * member utilities:  for sorority s, member i, rushee r  -> member_util_idx(s, i, r)
                           (sororities' blocks are laid out one after another)
      * rushee utilities:  for rushee r, sorority s            -> rushee_util_idx(r, s)
      * scale:             last entry; the Charnes-Cooper scaling variable. Each agent's
                           utilities sum to `scale` (rather than 1), which lets the LP
                           maximize a ratio of welfares as a plain linear program.
    """

    def __init__(self, n_sor, members_per_sor, quota, n_rushees, floor=0.0):
        self.n_sor, self.members_per_sor, self.quota = n_sor, members_per_sor, quota
        self.n_rushees, self.floor = n_rushees, floor
        # member_block_start[s] = index of sorority s's first member utility
        self.member_block_start = [0]
        for s in range(n_sor):
            self.member_block_start.append(self.member_block_start[-1] + members_per_sor[s] * n_rushees)
        self.n_member_utils = self.member_block_start[-1]
        self.n_vars = self.n_member_utils + n_rushees * n_sor + 1  # + scale
        # every feasible matching: each rushee gets a sorority (or -1), no sorority over quota
        self.matchings = [m for m in itertools.product(range(-1, n_sor), repeat=n_rushees)
                          if all(m.count(s) <= quota[s] for s in range(n_sor))]

    def member_util_idx(self, s, i, r):
        """Index of member i of sorority s's utility for rushee r."""
        return self.member_block_start[s] + i * self.n_rushees + r

    def rushee_util_idx(self, r, s):
        """Index of rushee r's utility for sorority s."""
        return self.n_member_utils + r * self.n_sor + s

    def welfare_weights(self, matching, welfare_type):
        """0/1 vector w such that w @ utils = social welfare of `matching`."""
        w = np.zeros(self.n_vars)
        for r, s in enumerate(matching):
            if s < 0:
                continue
            if welfare_type in ("sor", "both"):
                for i in range(self.members_per_sor[s]):
                    w[self.member_util_idx(s, i, r)] += 1
            if welfare_type in ("rush", "both"):
                w[self.rushee_util_idx(r, s)] += 1
        return w

    def constraints(self, member_rankings, rushee_rankings):
        """Constraints that define 'utilities consistent with these rankings'.

        Returns (A_ub, A_eq): A_ub @ utils <= 0 are the ordinal (and floor) constraints,
        A_eq @ utils = 0 say every agent's utilities sum to `scale`.
        """
        rows = []
        for s in range(self.n_sor):
            for i, ranking in enumerate(member_rankings[s]):
                for j in range(len(ranking) - 1):  # u(ranking[j]) >= u(ranking[j+1]); unmet rushees unconstrained
                    row = np.zeros(self.n_vars)
                    row[self.member_util_idx(s, i, ranking[j + 1])] = 1
                    row[self.member_util_idx(s, i, ranking[j])] = -1
                    rows.append(row)
        for r in range(self.n_rushees):
            ranking = rushee_rankings[r]
            for j in range(self.n_sor - 1):
                row = np.zeros(self.n_vars)
                row[self.rushee_util_idx(r, ranking[j + 1])] = 1
                row[self.rushee_util_idx(r, ranking[j])] = -1
                rows.append(row)
        if self.floor > 0:  # utility >= floor * scale
            for v in range(self.n_vars - 1):
                row = np.zeros(self.n_vars)
                row[v], row[-1] = -1, self.floor
                rows.append(row)
        A_ub = np.array(rows) if rows else None
        eq_rows = []
        for s in range(self.n_sor):
            for i in range(self.members_per_sor[s]):
                row = np.zeros(self.n_vars)
                for r in range(self.n_rushees):
                    row[self.member_util_idx(s, i, r)] = 1
                row[-1] = -1
                eq_rows.append(row)
        for r in range(self.n_rushees):
            row = np.zeros(self.n_vars)
            for s in range(self.n_sor):
                row[self.rushee_util_idx(r, s)] = 1
            row[-1] = -1
            eq_rows.append(row)
        return A_ub, np.array(eq_rows)

    def distortion(self, A_ub, A_eq, matching, other_matching, welfare_type):
        """sup over consistent utilities of SW(other_matching)/SW(matching), floored at 1.

        Returns (distortion, witness). When the distortion is infinite, `witness` is a utility
        vector (same layout as `utils`, unit-sum) consistent with the rankings under which
        SW(matching) = 0 < SW(other_matching); otherwise it is None."""
        if matching == other_matching:
            return 1.0, None
        w_matching = self.welfare_weights(matching, welfare_type)
        w_other = self.welfare_weights(other_matching, welfare_type)
        b_ub = np.zeros(A_ub.shape[0]) if A_ub is not None else None
        n_eq_rows = A_eq.shape[0]
        if self.floor == 0:  # can SW(matching)=0 while SW(other_matching)>0 ?  -> infinite distortion
            scale_row = np.zeros(self.n_vars)
            scale_row[-1] = 1
            zero_check = linprog(-w_other, A_ub=A_ub, b_ub=b_ub,
                                 A_eq=np.vstack([A_eq, w_matching, scale_row]),
                                 b_eq=np.concatenate([np.zeros(n_eq_rows), [0.0, 1.0]]),
                                 bounds=(0, None), method="highs")
            if zero_check.status == 0 and -zero_check.fun > 1e-9:
                return INF, zero_check.x
        # Charnes-Cooper: fix SW(matching) = 1, maximize SW(other_matching)
        result = linprog(-w_other, A_ub=A_ub, b_ub=b_ub, A_eq=np.vstack([A_eq, w_matching]),
                         b_eq=np.concatenate([np.zeros(n_eq_rows), [1.0]]),
                         bounds=(0, None), method="highs")
        if result.status == 3:
            return INF, None  # unbounded LP: no single utility vector to show
        if result.status == 2:
            return 1.0, None
        return max(1.0, -result.fun), None

# ----------------------------------------------------------------------------------------
# Profile generation
# ----------------------------------------------------------------------------------------

def draw_visibility(rng, n_rushees, meet_prob, min_meet, meet_count=None):
    """Set of rushees one member meets. Replace for structured visibility patterns.

    meet_count given: exactly that many rushees, chosen uniformly at random.
    Otherwise: each rushee independently w.p. meet_prob (at least min_meet rushees)."""
    if meet_count is not None:
        return [int(r) for r in rng.choice(n_rushees, size=meet_count, replace=False)]
    if meet_prob >= 1:
        return list(range(n_rushees))
    met = [r for r in range(n_rushees) if rng.random() < meet_prob]
    while len(met) < min(min_meet, n_rushees):
        r = int(rng.integers(n_rushees))
        if r not in met:
            met.append(r)
    return met


def sample_profile(rng, inst, alpha, meet_prob, min_meet, meet_count=None):
    """Draw true utilities, then derive the rankings they induce.

    Returns (member_rankings, rushee_rankings, utils)."""
    n_sor, n_rushees = inst.n_sor, inst.n_rushees
    utils = np.zeros(inst.n_vars)
    utils[-1] = 1  # scale
    member_rankings = []
    for s in range(n_sor):
        rankings = []
        for i in range(inst.members_per_sor[s]):
            member_utils = rng.dirichlet([alpha] * n_rushees)
            for r in range(n_rushees):
                utils[inst.member_util_idx(s, i, r)] = member_utils[r]
            met = draw_visibility(rng, n_rushees, meet_prob, min_meet, meet_count)
            rankings.append(tuple(sorted(met, key=lambda r: -member_utils[r])))
        member_rankings.append(rankings)
    rushee_rankings = []
    for r in range(n_rushees):
        rushee_utils = rng.dirichlet([alpha] * n_sor)
        for s in range(n_sor):
            utils[inst.rushee_util_idx(r, s)] = rushee_utils[s]
        rushee_rankings.append(tuple(sorted(range(n_sor), key=lambda s: -rushee_utils[s])))
    return member_rankings, rushee_rankings, utils


def exhaustive_profiles(inst):
    """Yield every ordinal profile (full visibility). Members of a sorority are interchangeable
    for aggregation, so their rankings are enumerated as multisets."""
    rushee_orders = list(itertools.permutations(range(inst.n_rushees)))
    sor_orders = list(itertools.permutations(range(inst.n_sor)))
    per_sor = [list(itertools.combinations_with_replacement(rushee_orders, k)) for k in inst.members_per_sor]
    for member_rankings in itertools.product(*per_sor):
        for rushee_rankings in itertools.product(sor_orders, repeat=inst.n_rushees):
            yield [list(g) for g in member_rankings], list(rushee_rankings), None

# ----------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------

def ratio(opt, got):
    if got > 1e-12:
        return max(1.0, opt / got)
    return 1.0 if opt <= 1e-12 else INF


def evaluate(inst, member_rankings, rushee_rankings, utils, sides, agg, truncate, do_lp):
    n_sor, n_rushees, quota = inst.n_sor, inst.n_rushees, inst.quota
    sor_lists = sorority_lists(member_rankings, agg, n_rushees, truncate)
    da_algorithms = {"rushee": da_rushee_proposing, "sorority": da_sorority_proposing}
    outcome = {side: da_algorithms[side](rushee_rankings, sor_lists, quota, n_sor, n_rushees)
               for side in sides}
    stable = [m for m in inst.matchings
              if is_stable(m, rushee_rankings, sor_lists, quota, n_sor, n_rushees)]
    for side in sides:
        assert outcome[side] in stable, f"{side}-proposing DA output failed stability check"
    if do_lp:
        A_ub, A_eq = inst.constraints(member_rankings, rushee_rankings)
    cache = {}

    def cached_distortion(matching, other_matching, welfare_type):
        key = (matching, other_matching, welfare_type)
        if key not in cache:
            cache[key] = inst.distortion(A_ub, A_eq, matching, other_matching, welfare_type)
        return cache[key]

    def worst_benchmark(matching, benchmark, welfare_type):
        """Max distortion of `matching` against every matching in `benchmark`.
        Returns (distortion, (benchmark matching attaining it, witness utilities or None))."""
        best_value, best_matching, best_witness = -1.0, None, None
        for other_matching in benchmark:
            value, witness = cached_distortion(matching, other_matching, welfare_type)
            if value > best_value:
                best_value, best_matching, best_witness = value, other_matching, witness
        return best_value, (best_matching, best_witness)

    result = {"lists": sor_lists, "stable": stable, "outcome": outcome}
    for side in sides:
        matching = outcome[side]
        for welfare_type in WELFARE_TYPES:
            cell = {}
            if utils is not None:
                got = float(inst.welfare_weights(matching, welfare_type) @ utils)
                cell["welfare"] = got
                cell["real_all"] = ratio(
                    max(float(inst.welfare_weights(m, welfare_type) @ utils) for m in inst.matchings), got)
                cell["real_stable"] = ratio(
                    max(float(inst.welfare_weights(m, welfare_type) @ utils) for m in stable), got)
            if do_lp:
                cell["lp_all"], cell["witness_all"] = worst_benchmark(matching, inst.matchings, welfare_type)
                cell["lp_stable"], cell["witness_stable"] = worst_benchmark(matching, stable, welfare_type)
            result[(side, welfare_type)] = cell
    return result

# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def broadcast(text, n_sor, flag_name):
    """Parse an int or comma list; a single value applies to every sorority."""
    values = [int(v) for v in text.split(",")]
    if len(values) == 1:
        values *= n_sor
    if len(values) != n_sor:
        sys.exit(f"--{flag_name} needs 1 value or {n_sor} comma-separated values")
    return values


def fmt(v):
    return "inf" if v == INF else f"{v:.3f}"


def print_unbounded_case(inst, side, welfare_type, benchmark, profile):
    """Show a concrete profile + utilities for which the distortion is infinite."""
    other_matching, witness = profile["witness"]
    matching = profile["outcome"]
    nums = lambda vals: "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"
    print(f"\n[{side}-proposing, {welfare_type} welfare, vs '{benchmark}' benchmark]  distortion=inf")
    print(f"  member rankings per sorority: {profile['members']}")
    print(f"  rushee rankings over sororities: {profile['rushees']}")
    print(f"  aggregated sorority lists: {profile['lists']}")
    print(f"  algorithm's matching (rushee -> sorority, -1 = unmatched): {matching}")
    print(f"  better matching: {other_matching}")
    if witness is None:
        print("  (LP is unbounded; no single utility vector extracted)")
        return
    print("  utilities consistent with the rankings (ties allowed):")
    for s in range(inst.n_sor):
        for i in range(inst.members_per_sor[s]):
            vals = [witness[inst.member_util_idx(s, i, r)] for r in range(inst.n_rushees)]
            print(f"    sorority {s} member {i}, utility for rushees 0..{inst.n_rushees - 1}: {nums(vals)}")
    for r in range(inst.n_rushees):
        vals = [witness[inst.rushee_util_idx(r, s)] for s in range(inst.n_sor)]
        print(f"    rushee {r}, utility for sororities 0..{inst.n_sor - 1}: {nums(vals)}")
    print(f"  welfare under these utilities: algorithm's matching = "
          f"{float(inst.welfare_weights(matching, welfare_type) @ witness):.3f}, "
          f"better matching = {float(inst.welfare_weights(other_matching, welfare_type) @ witness):.3f}")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(values):
    """'mean±sd' of the finite values (one per seed), plus how many seeds were infinite."""
    finite = [v for v in values if v != INF]
    n_inf = len(values) - len(finite)
    if not finite:
        return "inf"
    text = f"{np.mean(finite):.3f}"
    if len(finite) > 1:
        text += f"±{np.std(finite, ddof=1):.3f}"
    if n_inf:
        text += f" ({n_inf}/{len(values)} inf)"
    return text


def run_seed(inst, args, sides, cells, seed):
    """Evaluate every profile for one seed. Returns counts plus the per-cell results."""
    rng = np.random.default_rng(seed)
    if args.search == "exhaustive":
        profiles = exhaustive_profiles(inst)
    else:
        profiles = (sample_profile(rng, inst, args.alpha, args.meet_prob, args.min_meet, args.meet_count)
                    for _ in range(args.samples))

    realized = {c: {"welfare": [], "real_all": [], "real_stable": []} for c in cells}  # per-profile values
    worst = {c: {} for c in cells}  # worst[cell]["lp_all" / "lp_stable"] = (distortion, profile that attains it)
    n_profiles = n_multi_stable = n_differ = 0
    for member_rankings, rushee_rankings, utils in profiles:
        n_profiles += 1
        res = evaluate(inst, member_rankings, rushee_rankings, utils, sides, args.agg, args.truncate,
                       not args.no_lp)
        n_multi_stable += len(res["stable"]) > 1
        if len(sides) == 2:
            n_differ += res["outcome"]["rushee"] != res["outcome"]["sorority"]
        for cell in cells:
            for name, value in res[cell].items():
                if name in realized[cell]:
                    realized[cell][name].append(value)
                if name.startswith("lp") and (name not in worst[cell] or value > worst[cell][name][0]):
                    worst[cell][name] = (value, dict(members=member_rankings, rushees=rushee_rankings,
                                                     lists=res["lists"], outcome=res["outcome"][cell[0]],
                                                     stable=res["stable"],
                                                     witness=res[cell]["witness_" + name[len("lp_"):]]))
    return dict(n_profiles=n_profiles, n_multi_stable=n_multi_stable, n_differ=n_differ,
                realized=realized, worst=worst)


def report_single_seed(args, inst, sides, cells, run, elapsed):
    n_profiles, n_multi_stable, n_differ = run["n_profiles"], run["n_multi_stable"], run["n_differ"]
    realized, worst = run["realized"], run["worst"]
    print(f"{args.search}: {n_profiles} profiles in {elapsed:.1f}s | >1 stable matching: {n_multi_stable}"
          + (f" | rushee-DA != sorority-DA: {n_differ}" if len(sides) == 2 else ""))
    print("-" * 92)
    has_utils = args.search == "sample"
    header = f"{'proposing':<9} {'welfare':<6}"
    if has_utils:
        header += f" {'mean W':>8} {'mean ratio(all)':>16} {'mean ratio(stab)':>17}"
    if not args.no_lp:
        header += f" {'max LP dist(all)':>17} {'max LP dist(stab)':>18}"
    print(header)
    for side, welfare_type in cells:
        line = f"{side:<9} {welfare_type:<6}"
        if has_utils:
            vals = realized[(side, welfare_type)]
            line += (f" {mean(vals['welfare']):8.3f} {fmt(mean(vals['real_all'])):>16}"
                     f" {fmt(mean(vals['real_stable'])):>17}")
        if not args.no_lp:
            w = worst[(side, welfare_type)]
            line += f" {fmt(w['lp_all'][0]):>17} {fmt(w['lp_stable'][0]):>18}"
        print(line)
    if not args.no_lp:
        unbounded = [(cell, bench) for cell in cells for bench in ("all", "stable")
                     if worst[cell]["lp_" + bench][0] == INF]
        if unbounded:
            print("-" * 92 + "\nUnbounded distortion: example profile and utilities for each case")
        for (side, welfare_type), bench in unbounded:
            print_unbounded_case(inst, side, welfare_type, bench, worst[(side, welfare_type)]["lp_" + bench][1])
    if args.verbose and not args.no_lp:
        print("-" * 92 + "\nWorst-case profiles (LP distortion vs 'stable' benchmark):")
        for cell in cells:
            value, profile = worst[cell]["lp_stable"]
            print(f"\n[{cell[0]}-proposing, {cell[1]} welfare]  distortion={fmt(value)}")
            print(f"  member rankings per sorority: {profile['members']}")
            print(f"  rushee rankings over sororities: {profile['rushees']}")
            print(f"  aggregated sorority lists: {profile['lists']}")
            print(f"  outcome (rushee -> sorority, -1 = unmatched): {profile['outcome']}")
            print(f"  stable matchings: {profile['stable']}")


def report_multi_seed(args, sides, cells, seeds, runs, elapsed):
    multi = [r["n_multi_stable"] for r in runs]
    differ = [r["n_differ"] for r in runs]
    print(f"sample: {len(seeds)} seeds ({seeds[0]}..{seeds[-1]}) x {args.samples} profiles in {elapsed:.1f}s")
    print(f"per seed, profiles with >1 stable matching: mean {np.mean(multi):.1f} (min {min(multi)}, max {max(multi)})"
          + (f" | rushee-DA != sorority-DA: mean {np.mean(differ):.1f}" if len(sides) == 2 else ""))
    print("-" * 92)
    header = ["proposing", "welfare", "mean W", "mean ratio(all)", "mean ratio(stab)"]
    if not args.no_lp:
        header += ["max LP dist(all)", "max LP dist(stab)", "worst seed (all)", "worst seed (stab)"]
    rows = [header]
    for cell in cells:
        def per_seed(name):
            return [mean(r["realized"][cell][name]) for r in runs]
        row = [cell[0], cell[1], summarize(per_seed("welfare")),
               summarize(per_seed("real_all")), summarize(per_seed("real_stable"))]
        if not args.no_lp:
            lp_all = [r["worst"][cell]["lp_all"][0] for r in runs]
            lp_stable = [r["worst"][cell]["lp_stable"][0] for r in runs]
            row += [summarize(lp_all), summarize(lp_stable), fmt(max(lp_all)), fmt(max(lp_stable))]
        rows.append(row)
    widths = [max(len(row[c]) for row in rows) for c in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    print("-" * 92)
    print("Each entry is mean±sd across seeds of that seed's statistic (the LP columns use each seed's max\n"
          "over its profiles); '(k/N inf)' = k seeds were infinite and are left out of the mean.\n"
          "'worst seed' = the largest value over all seeds. --verbose / unbounded-case details need --num-seeds 1.")



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sororities", type=int, default=2)
    ap.add_argument("--members", default="3", help="members per sorority (int or comma list)")
    ap.add_argument("--rushees", type=int, default=2)
    ap.add_argument("--spots", default="1", help="quota per sorority (int or comma list)")
    ap.add_argument("--proposing", choices=["rushee", "sorority", "both"], default="both")
    ap.add_argument("--agg", choices=list(AGGREGATORS), default="plurality")
    ap.add_argument("--truncate", choices=["none", "unseen", "zero"], default="none",
                    help="which rushees a sorority finds unacceptable after aggregation")
    ap.add_argument("--search", choices=["sample", "exhaustive"], default="sample")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration for sampled utilities")
    visibility = ap.add_mutually_exclusive_group()  # neither flag = every member meets every rushee
    visibility.add_argument("--meet-prob", type=float, default=1.0,
                            help="prob. a member meets a given rushee (independently)")
    visibility.add_argument("--meet-count", type=int, default=None,
                            help="every member meets exactly this many rushees, chosen at random")
    ap.add_argument("--min-meet", type=int, default=1, help="with --meet-prob: minimum rushees a member meets")
    ap.add_argument("--floor", type=float, default=0.0, help="utility floor (unit-sum); 0 allows zeros")
    ap.add_argument("--no-lp", action="store_true", help="skip distortion LPs (realized welfare only)")
    ap.add_argument("--seed", type=int, default=0, help="first random seed")
    ap.add_argument("--num-seeds", type=int, default=1,
                    help="run seeds --seed .. --seed+N-1 and report mean/sd across seeds (sample mode)")
    ap.add_argument("--verbose", action="store_true", help="print worst-case profile per cell")
    args = ap.parse_args()

    n_sor, n_rushees = args.sororities, args.rushees
    members_per_sor = broadcast(args.members, n_sor, "members")
    quota = broadcast(args.spots, n_sor, "spots")
    sides = ["rushee", "sorority"] if args.proposing == "both" else [args.proposing]
    inst = Instance(n_sor, members_per_sor, quota, n_rushees, args.floor)
    if args.meet_count is not None and not 1 <= args.meet_count <= n_rushees:
        sys.exit(f"--meet-count must be between 1 and the number of rushees ({n_rushees})")
    partial_visibility = args.meet_prob < 1 or (args.meet_count is not None and args.meet_count < n_rushees)
    if args.search == "exhaustive" and partial_visibility:
        sys.exit("exhaustive search supports full visibility only (use --search sample)")
    if args.search == "exhaustive" and args.no_lp:
        sys.exit("exhaustive search needs the LPs")

    if args.num_seeds < 1:
        sys.exit("--num-seeds must be at least 1")
    if args.search == "exhaustive" and args.num_seeds > 1:
        sys.exit("exhaustive search does not depend on the seed (use --search sample with --num-seeds)")

    cells = [(side, welfare_type) for side in sides for welfare_type in WELFARE_TYPES]
    seeds = [args.seed + i for i in range(args.num_seeds)]
    t0 = time.time()
    runs = [run_seed(inst, args, sides, cells, seed) for seed in seeds]
    elapsed = time.time() - t0

    print("=" * 92)
    print(f"S={n_sor} sororities  members={members_per_sor}  rushees={n_rushees}  spots={quota}  "
          f"agg={args.agg}  truncate={args.truncate}  floor={args.floor}  "
          + (f"meet_count={args.meet_count}" if args.meet_count is not None else f"meet_prob={args.meet_prob}"))
    if len(seeds) == 1:
        report_single_seed(args, inst, sides, cells, runs[0], elapsed)
    else:
        report_multi_seed(args, sides, cells, seeds, runs, elapsed)


if __name__ == "__main__":
    main()
