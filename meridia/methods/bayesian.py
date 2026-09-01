"""Strong method B: the Bayesian hierarchical line.

Sources: hierarchical Bayesian small-area models after Rao and Molina (2015, ch. 10)
and Gelman et al. (2013); capture-recapture reasoning after Chandrasekar and Deming
(1949); cohort-component projection after Preston, Heuveline, and Guillot (2001).
See docs/INDEPENDENCE.md.

The same packet, a different statistical philosophy. Instead of ratio adjustments and a
bootstrap, every county quantity is a latent variable with a posterior:

1. Coverage. The deduplicated register count in county c is a binomial draw from the
   true population with coverage p_c. The survey's nonresponse-adjusted direct estimate
   is a noisy measurement of the same true population. Coverage rates share a Beta
   prior within each state and size class (small counties are covered worse), whose
   hyperparameters are learned from the data. A Gibbs sampler alternates population,
   coverage, and hyperparameters; posterior draws give intervals that already carry
   both sampling error and between-county model error.
2. Income. County log-mean incomes follow a normal hierarchical model shrunk toward a
   state mean plus a slope on the income source's county mean; sampling variances come
   from the survey. Selective nonresponse is corrected by the same development-world
   calibration channel every participant has, fitted for this method.
3. Projection. Cohort-component on each posterior draw of the county age cube, with
   mortality and fertility drawn from the public ranges, so the projection's interval
   integrates over reconstruction and demographic uncertainty together.
4. Allocation proportional to posterior-mean projected elders; detailed table with
   primary suppression and no totals.

The method reads participant files only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..release import AGE_BANDS, ESTIMAND_IDS
from . import design_based as A
from .common import (COUNT_ITEMS, INCOME_ITEMS, apply_calibration, calibrate_income,
                     calibration_half_widths, income_dispersion, load_factors, load_packet,
                     rows_from_draws, write_submission)


@dataclass(frozen=True)
class MethodParams:
    sweeps: int = 400
    burn_in: int = 100
    seed: int = 20260902
    suppression_multiplier: float = 2.0
    calibration_path: str | None = None


def _beta_from_moments(mean: float, var: float, floor: float = 2.0) -> tuple[float, float]:
    mean = min(max(mean, 0.05), 0.995)
    var = max(var, 1e-6)
    common = mean * (1.0 - mean) / var - 1.0
    common = max(common, floor)
    return mean * common, (1.0 - mean) * common


def sample_population(register: np.ndarray, direct: np.ndarray, direct_var: np.ndarray,
                      county_state: np.ndarray, small: np.ndarray, rng: np.random.Generator,
                      sweeps: int, burn_in: int, n_psu: np.ndarray | None = None,
                      concentration: float = 12.0) -> np.ndarray:
    """Gibbs draws of true county populations.

    Coverage p_c has a Beta prior per state and size class whose mean is the pooled
    survey-to-register ratio of that group and whose concentration is fixed (25,
    roughly a plus or minus ten percent spread of county coverages). The prior is not
    re-estimated from the draws, which would collapse it; the group mean carries the
    survey's information and the concentration carries the modeller's.
    """
    n_counties = len(register)
    n_states = int(county_state.max()) + 1
    groups = county_state * 2 + small.astype(np.int64)
    have = np.isfinite(direct_var) & (direct_var > 0) & (direct > 0)
    if n_psu is None:
        n_psu = np.where(direct > 0, 4.0, 0.0)
    # Group prior means: pooled register over pooled direct where the survey has
    # sampled the group; the small-county class is pooled nationally.
    # Group prior means shrink toward the national ratio in proportion to the sampling
    # units behind them, and stay within the public mechanism bounds.
    national = float(register[direct > 0].sum() / max(direct[direct > 0].sum(), 1e-9)) if (direct > 0).any() else 0.9
    prior_mean = np.full(2 * n_states, national)
    for g in range(2 * n_states):
        members = (groups == g) & (direct > 0)
        if g % 2 == 1:
            members = small & (direct > 0)
        if members.sum() >= 1 and direct[members].sum() > 0:
            ratio = register[members].sum() / direct[members].sum()
            units = float(n_psu[members].sum())
            w = units / (units + A.COVERAGE_PRIOR_UNITS)
            prior_mean[g] = float(np.clip(w * ratio + (1.0 - w) * national, A.COVERAGE_BOUNDS[0], A.COVERAGE_BOUNDS[1]))
    # The group mean itself is estimated from the survey; its sampling error widens
    # the effective prior so the posterior carries it.
    prior_se = np.zeros(2 * n_states)
    for g in range(2 * n_states):
        members = (groups == g) & have if g % 2 == 0 else small & have
        if members.sum() >= 1 and direct[members].sum() > 0:
            prior_se[g] = float(np.sqrt(direct_var[members].sum()) / direct[members].sum() * prior_mean[g])
        else:
            prior_se[g] = 0.08
    var_eff = prior_mean * (1.0 - prior_mean) / (concentration + 1.0) + prior_se ** 2
    kappa = np.maximum(prior_mean * (1.0 - prior_mean) / np.maximum(var_eff, 1e-9) - 1.0, 2.0)
    a = prior_mean * kappa
    b = (1.0 - prior_mean) * kappa
    # Marginal sampling on a grid. Register and true population are tied along the
    # curve N p = r; a Gibbs chain along that ridge mixes too slowly to trust. Instead,
    # for each county the posterior of p under the Beta prior and, where the survey
    # sampled the county, the direct measurement's normal likelihood of N = r / p, is
    # evaluated on a grid and sampled exactly; N then adds the binomial noise.
    from scipy.stats import beta as beta_dist
    grid = np.linspace(0.30, 0.999, 400)
    n_draws = sweeps - burn_in
    draws = np.zeros((n_draws, n_counties))
    for c in range(n_counties):
        g = groups[c]
        log_w = beta_dist.logpdf(grid, a[g], b[g])
        r = float(register[c])
        if have[c]:
            # A floor on the direct measurement's variance: two or three sampling
            # units cannot certify a county more tightly than eight percent.
            var_c = max(direct_var[c], (0.08 * direct[c]) ** 2)
            log_w = log_w - 0.5 * (direct[c] - r / grid) ** 2 / var_c
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        p = rng.choice(grid, size=n_draws, p=w)
        draws[:, c] = np.maximum(r / p + rng.normal(0.0, np.sqrt(max(r, 1.0) * (1.0 - p)) / p), r)
    # Counties in a group share the error of their group's coverage estimate: one
    # common factor per group per draw, so state and national intervals carry it.
    for g in range(2 * n_states):
        members = groups == g
        if members.any() and prior_se[g] > 0:
            shift = rng.normal(0.0, prior_se[g] / prior_mean[g], size=n_draws)
            draws[:, members] *= np.exp(-shift)[:, None]
    return draws


def sample_income(frame, register_frame, income, county_state: np.ndarray,
                  rng: np.random.Generator, sweeps: int, burn_in: int) -> dict:
    """Normal hierarchical draws of county income quantities from the survey with the
    income source as a covariate; state and nation from person-weighted aggregation."""
    n_counties = len(county_state)
    n_states = int(county_state.max()) + 1
    ratios = A.income_source_ratios(income, county_state, A.survey_statistics(frame, county_state)
                                    ["median_household_income"]["nation"])
    out = {}
    w = frame["weight"].to_numpy(dtype=np.float64)
    county = frame["county"].to_numpy(dtype=np.int64)
    adults = frame["age"].to_numpy() >= 16
    income = frame["income"].to_numpy(dtype=np.float64)
    # Direct county estimates: the log of the weighted arithmetic mean of adult income
    # (not the mean of logs, which would understate a skewed mean), with a
    # between-unit variance proxy on the same scale.
    direct = np.full(n_counties, np.nan)
    var = np.full(n_counties, np.nan)
    for c in range(n_counties):
        mask = (county == c) & adults
        if mask.sum() >= 5:
            direct[c] = float(np.log(max((w[mask] * income[mask]).sum() / w[mask].sum(), 1.0)))
            psus = frame["psu"].to_numpy()[mask]
            per = [float(np.log(max((w[mask][psus == u] * income[mask][psus == u]).sum() / w[mask][psus == u].sum(), 1.0)))
                   for u in np.unique(psus)]
            var[c] = float(np.var(per, ddof=1) / len(per)) if len(per) >= 2 else np.nan
    covariate = np.log(np.maximum(ratios["mean_income_adults"], 1e-3))
    have = np.isfinite(direct) & np.isfinite(var) & (var > 0)
    theta = np.where(have, direct, np.nan)
    mu = np.zeros(n_states)
    beta, tau2 = 0.5, 0.05
    draws = np.zeros((sweeps - burn_in, n_counties))
    state_of = county_state
    for sweep in range(sweeps):
        # State means given county effects.
        for s in range(n_states):
            members = (state_of == s) & np.isfinite(theta)
            if members.sum():
                resid = theta[members] - beta * covariate[members]
                mu[s] = rng.normal(resid.mean(), np.sqrt(tau2 / members.sum()))
        prior_mean = mu[state_of] + beta * covariate
        # County effects: shrink direct estimates toward the prior mean.
        post_var = np.where(have, 1.0 / (1.0 / tau2 + 1.0 / np.where(have, var, 1.0)), tau2)
        post_mean = np.where(have, post_var * (prior_mean / tau2 + np.where(have, direct, 0.0) / np.where(have, var, 1.0)), prior_mean)
        theta = rng.normal(post_mean, np.sqrt(post_var))
        # Slope and variance by moments.
        resid = theta - mu[state_of]
        cov_c = covariate - covariate.mean()
        if (cov_c ** 2).sum() > 1e-9:
            beta = float((cov_c * (resid - resid.mean())).sum() / (cov_c ** 2).sum())
        tau2 = float(max(np.var(theta - prior_mean), 1e-4))
        if sweep >= burn_in:
            draws[sweep - burn_in] = theta
    out["mean_income_adults"] = np.exp(draws)
    # Median and low-income share: state survey estimates scaled by county ratios, with
    # the county mean's posterior relative spread as the uncertainty carrier.
    stats = A.survey_statistics(frame, county_state)
    rel = np.exp(draws) / np.maximum(np.exp(draws).mean(axis=0), 1e-9)
    # The household median tracks the county mean through one survey-estimated
    # national ratio (median household income over mean adult income), so its
    # posterior inherits the mean's; the low-income share moves against the mean.
    national_ratio = stats["median_household_income"]["nation"] / max(stats["mean_income_adults"]["nation"], 1e-9)
    out["median_household_income"] = out["mean_income_adults"] * national_ratio
    state_values = np.asarray([stats["low_income_household_share"][s] for s in range(n_states)])
    base = state_values[county_state] * ratios["low_income_household_share"]
    out["low_income_household_share"] = np.clip(base[None, :] * (2.0 - rel), 0.0, 1.0)
    out["_state_stats"] = stats
    return out


def run(packet_dir: Path, out_dir: Path, params: MethodParams = MethodParams()) -> dict:
    data = load_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    n_counties = len(county_state)
    tick = int(contract["ticks"]["revised"])
    horizon_months = int(contract["ticks"]["horizon"]) - tick
    rng = np.random.default_rng(params.seed)

    register_frame = A.deduplicate_population(data["population"], tick)
    register = A.register_counts(register_frame, n_counties)
    mortality = A.estimate_mortality(data["population_preliminary"], data["population"],
                                     int(contract["ticks"]["preliminary"]), tick)
    survey = A.impute_income(A.rake_to_register(A.adjusted_survey(data["survey"]), register_frame, county_state))
    dispersion = income_dispersion(survey)
    factors = load_factors(params.calibration_path)

    direct, direct_var = A._direct_county_persons(survey, n_counties)
    reg_persons = np.asarray(register["persons"], dtype=np.float64)
    small = reg_persons <= np.quantile(reg_persons[reg_persons > 0], 0.25) if (reg_persons > 0).sum() >= 8 \
        else np.zeros(n_counties, bool)
    n_draws = params.sweeps - params.burn_in
    n_psu_county = survey.groupby("county")["psu"].nunique().reindex(range(n_counties), fill_value=0).to_numpy(dtype=np.float64)
    persons_draws = sample_population(reg_persons, direct, direct_var, county_state, small, rng,
                                      params.sweeps, params.burn_in, n_psu=n_psu_county)
    ratio_draws = persons_draws / np.maximum(reg_persons, 1.0)[None, :]
    count_draws = {"persons": persons_draws}
    for e in ("households", "children_under_16", "elders_65_plus"):
        count_draws[e] = np.asarray(register[e], dtype=np.float64)[None, :] * ratio_draws
    income_draws = sample_income(survey, register_frame, data["income"], county_state, rng,
                                 params.sweeps, params.burn_in)
    stats = income_draws.pop("_state_stats")

    def aggregate_b(values: dict, persons: np.ndarray) -> dict:
        """Every level from county draws: counts add; income items and shares are
        person-weighted means of county values, medians included, so the posterior
        spread reaches state and nation."""
        out = A.aggregate(values, county_state, stats, persons)
        n_states = int(county_state.max()) + 1
        for e in INCOME_ITEMS:
            v = np.nan_to_num(values[e])
            state_v = np.bincount(county_state, weights=v * persons, minlength=n_states) / \
                np.maximum(np.bincount(county_state, weights=persons, minlength=n_states), 1e-9)
            for s_ in range(n_states):
                out[(e, "state", s_)] = float(state_v[s_])
            out[(e, "nation", 0)] = float((v * persons).sum() / max(persons.sum(), 1e-9))
        return out

    def county_values(k: int) -> dict:
        values = {e: count_draws[e][k] for e in COUNT_ITEMS}
        values.update({e: income_draws[e][k] for e in INCOME_ITEMS})
        values["tertiary_share_25_plus"] = register["tertiary_share_25_plus"]
        return values

    draws_now: dict[tuple, list] = {}
    draws_future: dict[tuple, list] = {}
    cube = np.asarray(register["cube"], dtype=np.float64)
    age_sex = np.asarray(register["age_sex"], dtype=np.float64)
    for k in range(n_draws):
        values = county_values(k)
        agg = apply_calibration(aggregate_b(values, values["persons"]), factors, dispersion)
        for key, v in agg.items():
            draws_now.setdefault(key, []).append(v)
        future = A.project(values, age_sex * ratio_draws[k][:, None, None], horizon_months, rng, mortality)
        agg_f = apply_calibration(aggregate_b(future, future["persons"]), factors, dispersion)
        for key, v in agg_f.items():
            draws_future.setdefault(key, []).append(v)
    point_now = {key: float(np.nanmean(v)) for key, v in draws_now.items()}
    point_future = {key: float(np.nanmean(v)) for key, v in draws_future.items()}
    # Counts must add exactly: rebuild state and nation points from county points.
    for point in (point_now, point_future):
        for e in COUNT_ITEMS:
            county_points = np.asarray([point[(e, "county", c)] for c in range(n_counties)])
            state_points = np.bincount(county_state, weights=county_points, minlength=int(county_state.max()) + 1)
            for s, v in enumerate(state_points):
                point[(e, "state", s)] = float(v)
            point[(e, "nation", 0)] = float(state_points.sum())

    extra_now = calibration_half_widths(point_now, factors)
    extra_future = calibration_half_widths(point_future, factors)
    for extra in (extra_now, extra_future):
        for c in range(n_counties):
            v = point_now[("tertiary_share_25_plus", "county", c)]
            n_base = float(register["tertiary_n"][c])
            extra[("tertiary_share_25_plus", "county", c)] = float(np.sqrt((1.645 * np.sqrt(max(v * (1 - v), 1e-6) / max(n_base, 1.0))) ** 2 + 0.02 ** 2)) if np.isfinite(v) else 0.0
        for s_ in range(int(county_state.max()) + 1):
            v = point_now[("tertiary_share_25_plus", "state", s_)]
            n_base = float(register["tertiary_n"][county_state == s_].sum())
            extra[("tertiary_share_25_plus", "state", s_)] = float(np.sqrt((1.645 * np.sqrt(max(v * (1 - v), 1e-6) / max(n_base, 1.0))) ** 2 + 0.02 ** 2))
        v = point_now[("tertiary_share_25_plus", "nation", 0)]
        extra[("tertiary_share_25_plus", "nation", 0)] = float(np.sqrt((1.645 * np.sqrt(max(v * (1 - v), 1e-6) / max(float(register["tertiary_n"].sum()), 1.0))) ** 2 + 0.02 ** 2))
    # Model-error allowances the posterior does not carry: ten percent relative on
    # county quantities (synthetic components), and eight percent on every projected
    # count for the coarseness of band-level cohort aging.
    def widen(extra: dict, point: dict, projection: bool) -> None:
        for key, v in point.items():
            if not np.isfinite(v):
                continue
            e, level, _ = key
            add = 0.0
            if level == "county" and e != "tertiary_share_25_plus":
                add = 1.645 * 0.10 * (abs(v) if not e.endswith("share") else 0.5)
            if projection and e in COUNT_ITEMS:
                add = float(np.sqrt(add ** 2 + (1.645 * 0.03 * np.sqrt(horizon_months / 12.0) * abs(v)) ** 2))
            if projection and e in INCOME_ITEMS:
                # Income items are carried forward; five years of drift is not in the posterior.
                drift = 1.645 * 0.05 * np.sqrt(horizon_months / 12.0)
                add = float(np.sqrt(add ** 2 + (drift * (abs(v) if e != "low_income_household_share" else 0.5)) ** 2))
            if projection and e == "tertiary_share_25_plus":
                add = float(np.sqrt(add ** 2 + (0.03 * horizon_months / 60.0) ** 2))
            if add > 0:
                extra[key] = float(np.sqrt(extra.get(key, 0.0) ** 2 + add ** 2))
    widen(extra_now, point_now, False)
    widen(extra_future, point_future, True)
    release_rows = rows_from_draws(point_now, draws_now, extra_now)
    projection_rows = rows_from_draws(point_future, draws_future, extra_future)
    elders = np.maximum(np.asarray([point_future[("elders_65_plus", "county", c)] for c in range(n_counties)]), 0.0)
    budget = float(contract["allocation"]["budget"])
    allocation = np.floor(elders / max(elders.sum(), 1e-9) * budget * 1e6) / 1e6
    cube_point = cube * ratio_draws.mean(axis=0)[:, None, None]
    write_submission(out_dir, release_rows, projection_rows, cube_point,
                     params.suppression_multiplier * int(contract["disclosure_threshold"]), allocation)
    return {"release": release_rows, "projection": projection_rows, "dispersion": dispersion}


def calibrate(dev_packet_dirs, calibration_path: Path, params: MethodParams = MethodParams()) -> dict:
    quick = MethodParams(sweeps=150, burn_in=50, seed=params.seed)
    return calibrate_income(lambda d, o: run(d, o, quick), dev_packet_dirs, calibration_path)
