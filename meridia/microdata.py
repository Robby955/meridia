"""Microdata layer v1: persons and households consistent with the population grid.

The population grid gives an exact integer count per cell; this layer turns those counts
into person records grouped into households, cell by cell, so the microdata aggregates
back to the population grid exactly. Attributes are generated coherently: household structure
(one or two adults, then children and elders), ages by role, education shifted by how
urban the cell is (settlement pull), log-normal income driven by education, age, and
urbanity, and a latent frailty that carries each person's baseline health burden. Everything is determined by (seed, population inputs); the same inputs yield the
same tables.

This is the sampling frame for every survey product built on the world: a survey is a
draw from these tables, and any estimate can be checked against them exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDUCATION_LEVELS = 4  # 0 none/primary, 1 secondary, 2 tertiary, 3 advanced


@dataclass(frozen=True)
class MicrodataParams:
    household_size_probs: tuple = (0.28, 0.32, 0.16, 0.13, 0.07, 0.04)  # sizes 1..6
    two_adult_prob: float = 0.62         # households of size >= 2 with a second adult
    elder_prob: float = 0.14             # chance a non-head adult slot is an elder
    extra_child_prob: float = 0.62       # extra slots: child, else young adult or elder
    extra_adult_prob: float = 0.24
    education_urban_shift: float = 0.9   # logit shift toward higher education in cities
    income_base: float = 9.6             # log-income intercept
    income_per_education: float = 0.38
    income_age_peak: float = 47.0
    income_age_scale: float = 22.0
    income_urban_bonus: float = 0.35
    income_sigma: float = 0.55
    frailty_sigma: float = 0.45          # spread of latent health burden (log scale)
    frailty_age_slope: float = 0.30      # per 40 years above 45
    frailty_urban_slope: float = -0.20   # urban cells carry a lighter baseline burden


def _household_sizes_for_count(count: int, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Household sizes summing to ``count`` exactly: draw, then trim the last one."""
    sizes = []
    remaining = count
    max_size = len(probs)
    while remaining > 0:
        s = int(rng.choice(max_size, p=probs)) + 1
        if s >= remaining:
            sizes.append(remaining)
            remaining = 0
        else:
            sizes.append(s)
            remaining -= s
    return np.asarray(sizes, dtype=np.int64)


def build_microdata(population: np.ndarray, habitability: np.ndarray,
                    sites: list[tuple[int, int]], seed: int,
                    params: MicrodataParams = MicrodataParams()) -> dict:
    """Return person and household arrays whose cell totals equal the population grid exactly."""
    height, width = population.shape
    probs = np.asarray(params.household_size_probs, dtype=np.float64)
    probs = probs / probs.sum()

    # Urbanity in [0, 1]: settlement pull normalized, reusing the population layer's rule.
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    pull = np.zeros((height, width))
    for rank, (r, c) in enumerate(sites, start=1):
        distance = np.maximum(np.abs(rows - r), np.abs(cols - c))
        pull += (rank ** -1.0) * np.exp(-distance / 18.0)
    urbanity = pull / max(pull.max(), 1e-12)

    total = int(population.sum())
    person_household = np.empty(total, dtype=np.int64)
    person_cell = np.empty(total, dtype=np.int64)
    person_age = np.empty(total, dtype=np.int16)
    person_sex = np.empty(total, dtype=np.int8)
    person_role = np.empty(total, dtype=np.int8)  # 0 head, 1 partner, 2 child, 3 elder
    household_cell: list[int] = []

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x51D0]))
    p_idx = 0
    h_idx = 0
    for flat in np.flatnonzero(population):
        r, c = divmod(int(flat), width)
        count = int(population[r, c])
        sizes = _household_sizes_for_count(count, probs, rng)
        for size in sizes:
            household_cell.append(flat)
            members = []
            head_age = int(np.clip(round(rng.normal(46.0, 15.0)), 20, 89))
            members.append((0, head_age))
            slots = size - 1
            if slots > 0 and rng.random() < params.two_adult_prob:
                if rng.random() < params.elder_prob:
                    members.append((3, int(rng.integers(65, 95))))  # elder relative
                else:
                    partner_age = int(np.clip(round(head_age + rng.normal(0.0, 4.0)), 20, 89))
                    members.append((1, partner_age))                # partner, age-correlated
                slots -= 1
            for _ in range(slots):
                u = rng.random()
                if u < params.extra_child_prob:
                    members.append((2, int(rng.integers(0, 18))))   # child
                elif u < params.extra_child_prob + params.extra_adult_prob:
                    members.append((1, int(rng.integers(18, 40))))  # young adult
                else:
                    members.append((3, int(rng.integers(65, 95))))  # elder
            for role, age in members:
                person_household[p_idx] = h_idx
                person_cell[p_idx] = flat
                person_age[p_idx] = age
                person_sex[p_idx] = int(rng.random() < 0.5)
                person_role[p_idx] = role
                p_idx += 1
            h_idx += 1
    assert p_idx == total

    # Education: ordinal by a logit shifted with urbanity; children capped by age.
    urb_person = urbanity.flatten()[person_cell]
    logit = rng.normal(0.0, 1.0, size=total) + params.education_urban_shift * urb_person
    education = np.clip(np.digitize(logit, [-0.6, 0.4, 1.4]), 0, EDUCATION_LEVELS - 1).astype(np.int8)
    education[person_age < 16] = 0
    education[(person_age >= 16) & (person_age < 22)] = np.minimum(
        education[(person_age >= 16) & (person_age < 22)], 1)

    # Income: log-normal in education, an age hump, and urbanity; zero for children.
    age_term = -((person_age.astype(np.float64) - params.income_age_peak)
                 / params.income_age_scale) ** 2
    mu = (params.income_base + params.income_per_education * education
          + 0.5 * age_term + params.income_urban_bonus * urb_person)
    income = np.exp(mu + params.income_sigma * rng.normal(0.0, 1.0, size=total))
    income[person_age < 16] = 0.0
    income = np.round(income, 2)

    # Latent frailty: the per-person health burden that drives mortality, hospital
    # incidence, and health-source inclusion.  It is never published; the survey's
    # hospitalization item is the anchor that makes it estimable.  Mean one on the log
    # scale, heavier with age, lighter in urban cells, so baseline health burden is
    # observable structure rather than a world constant.
    frailty_mu = (params.frailty_age_slope * (person_age.astype(np.float64) - 45.0) / 40.0
                  + params.frailty_urban_slope * (urb_person - 0.5)
                  - 0.5 * params.frailty_sigma ** 2)
    frailty = np.exp(frailty_mu + params.frailty_sigma * rng.normal(0.0, 1.0, size=total))
    frailty = np.clip(frailty, 0.15, 6.0)

    return {
        "person": {
            "household": person_household, "cell": person_cell, "age": person_age,
            "sex": person_sex, "role": person_role, "education": education,
            "income": income, "frailty": frailty,
        },
        "household_cell": np.asarray(household_cell, dtype=np.int64),
        "urbanity": urbanity,
        "n_persons": total,
        "n_households": h_idx,
    }
