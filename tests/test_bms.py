import numpy as np
import pytest

from cbm.model_selection import bms
from cbm.bms_group import (
    bms_group,
    bms_group_btw_conds,
    bms_group_btw_groups,
)


def test_standard_bms_favors_better_model():
    L = np.array([
        [0.0, -10.0],
        [0.0, -10.0],
        [0.0, -10.0],
        [0.0, -10.0],
        [0.0, -10.0],
    ])

    result = bms(
        L,
        Nsamp=20_000,
        random_state=1,
    )

    assert result.model_frequency[0] > result.model_frequency[1]
    assert result.exceedance_prob[0] > 0.95
    assert result.protected_exceedance_prob[0] > 0.5
    np.testing.assert_allclose(
        result.g.sum(axis=1),
        1.0,
        atol=1e-12,
    )


def test_bms_seed_is_exactly_reproducible():
    L = np.array([
        [0.0, -1.0],
        [-0.5, 0.0],
        [0.2, -0.2],
    ])

    a = bms(
        L,
        Nsamp=5000,
        random_state=42,
    )
    b = bms(
        L,
        Nsamp=5000,
        random_state=42,
    )

    np.testing.assert_array_equal(
        a.exceedance_prob,
        b.exceedance_prob,
    )
    np.testing.assert_array_equal(
        a.protected_exceedance_prob,
        b.protected_exceedance_prob,
    )


def test_null_evidence_gives_symmetric_model_frequencies():
    L = np.zeros((10, 3))

    result = bms(
        L,
        Nsamp=10_000,
        random_state=4,
    )

    np.testing.assert_allclose(
        result.model_frequency,
        np.ones(3) / 3.0,
        atol=1e-3,
    )


def test_group_bms_display_is_numerically_invariant():
    L = np.array([
        [0.0, -2.0],
        [-1.0, 0.0],
        [0.2, -0.5],
    ])

    a = bms_group(
        L,
        n_samples=5000,
        random_state=123,
        verbose=False,
        display=False,
    )

    b = bms_group(
        L,
        n_samples=5000,
        random_state=123,
        verbose=False,
        display=True,
    )

    np.testing.assert_array_equal(
        a.model_frequency,
        b.model_frequency,
    )
    np.testing.assert_array_equal(
        a.exceedance_prob,
        b.exceedance_prob,
    )
    np.testing.assert_array_equal(
        a.protected_exceedance_prob,
        b.protected_exceedance_prob,
    )


def test_family_bms_favors_better_family():
    L = np.array([
        [0.0, -0.3, -8.0, -9.0],
        [-0.2, 0.0, -8.0, -9.0],
        [0.0, -0.1, -7.0, -8.0],
        [-0.4, 0.0, -8.0, -9.0],
    ])

    result = bms_group(
        L,
        families=[[0, 1], [2, 3]],
        family_names=["A", "B"],
        n_samples=10_000,
        random_state=2,
        verbose=False,
        display=False,
    )

    assert result.families.family_frequency[0] > result.families.family_frequency[1]
    assert result.families.exceedance_prob[0] > result.families.exceedance_prob[1]


@pytest.mark.parametrize(
    "families",
    [
        [[0, 1], [1, 2]],     # overlap
        [[0], [2]],           # missing model
        [[0, 1], [2, 99]],    # invalid index
    ],
)
def test_invalid_family_partitions_raise(families):
    L = np.zeros((4, 3))

    with pytest.raises(ValueError):
        bms_group(
            L,
            families=families,
            n_samples=1000,
            random_state=1,
            verbose=False,
            display=False,
        )


def test_between_conditions_detects_different_winners():
    # Same subjects, model 1 favored in condition 1,
    # model 2 favored in condition 2.
    L1 = np.tile(np.array([0.0, -8.0]), (8, 1))
    L2 = np.tile(np.array([-8.0, 0.0]), (8, 1))
    L = np.stack([L1, L2], axis=2)

    result = bms_group_btw_conds(
        L,
        n_samples=10_000,
        random_state=3,
        verbose=False,
        display=False,
    )

    assert result.pxp < 0.5


def test_between_conditions_detects_same_winner():
    L1 = np.tile(np.array([0.0, -8.0]), (8, 1))
    L2 = np.tile(np.array([0.0, -8.0]), (8, 1))
    L = np.stack([L1, L2], axis=2)

    result = bms_group_btw_conds(
        L,
        n_samples=10_000,
        random_state=3,
        verbose=False,
        display=False,
    )

    assert result.pxp > 0.5


def test_between_groups_detects_equal_profiles():
    L = np.tile(
        np.array([0.0, -5.0]),
        (8, 1),
    )

    result = bms_group_btw_groups(
        [L[:4], L[4:]],
        n_samples=5000,
        random_state=6,
        verbose=False,
        display=False,
    )

    assert result.p_equal > 0.5


def test_between_groups_detects_different_profiles():
    g1 = np.tile(
        np.array([0.0, -8.0]),
        (8, 1),
    )
    g2 = np.tile(
        np.array([-8.0, 0.0]),
        (8, 1),
    )

    result = bms_group_btw_groups(
        [g1, g2],
        n_samples=5000,
        random_state=6,
        verbose=False,
        display=False,
    )

    assert result.p_equal < 0.5
