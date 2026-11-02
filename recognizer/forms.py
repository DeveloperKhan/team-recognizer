"""Forms that look the same on the overlay, scored as one species.

Gourgeist and Pumpkaboo come in four sizes, but the overlay scales every sprite to fit its tile,
so the sizes are indistinguishable; Genesect's drives differ only by the colour of a tiny
cassette. Scored separately, the classifier spreads its belief evenly over the forms (about a
quarter each for Gourgeist), so any lookalike that reaches a little more than that wins even
when the species as a whole is near-certain. Each group's probabilities are therefore summed
into one representative form: the one Dracoviz uses by default (the other sizes are tagged
`teambuilderexclude` in its gamemaster, and the site maps a plain "Gourgeist" to the super size).
"""

from __future__ import annotations

import numpy as np

FORM_GROUPS: dict[str, tuple[str, ...]] = {
    "gourgeist_super": ("gourgeist_average", "gourgeist_small", "gourgeist_large", "gourgeist_super"),
    "pumpkaboo_super": ("pumpkaboo_average", "pumpkaboo_small", "pumpkaboo_large", "pumpkaboo_super"),
    "genesect": ("genesect", "genesect_burn", "genesect_chill", "genesect_douse", "genesect_shock"),
}

CANONICAL = {member: rep for rep, members in FORM_GROUPS.items() for member in members}


def canonical(species_id: str) -> str:
    """The form reported for a speciesId (unchanged unless it belongs to a group)."""
    return CANONICAL.get(species_id, species_id)


def merge_plan(class_species: list[str]) -> list[tuple[int, list[int]]]:
    """For each group present among the classifier's classes: (representative column, all
    member columns)."""
    column = {species: i for i, species in enumerate(class_species)}
    plan = []
    for rep, members in FORM_GROUPS.items():
        cols = [column[m] for m in members if m in column]
        if rep in column and len(cols) > 1:
            plan.append((column[rep], cols))
    return plan


def merge_probabilities(probs: np.ndarray, plan: list[tuple[int, list[int]]]) -> np.ndarray:
    """Move each group's total probability onto its representative column; the other members
    get zero, so they never appear as a separate guess."""
    out = probs.copy()
    for rep, cols in plan:
        total = probs[:, cols].sum(axis=1)
        out[:, cols] = 0.0
        out[:, rep] = total
    return out
