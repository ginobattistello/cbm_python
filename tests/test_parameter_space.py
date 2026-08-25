import numpy as np
import pytest

from cbm.parameter_space import ParameterSpace


def test_all_parameters_free():
    space = ParameterSpace.from_prior(
        prior_mean=np.array([1.0, 2.0]),
        prior_variance=np.array([4.0, 9.0]),
    )

    np.testing.assert_array_equal(
        space.free_mask,
        np.array([True, True]),
    )
    np.testing.assert_array_equal(
        space.fixed_mask,
        np.array([False, False]),
    )
    assert space.d_full == 2
    assert space.d_free == 2


def test_zero_variance_fixes_parameter_exactly():
    space = ParameterSpace.from_prior(
        prior_mean=np.array([1.0, 2.0, 3.0]),
        prior_variance=np.array([4.0, 0.0, 9.0]),
    )

    np.testing.assert_array_equal(
        space.free_indices,
        np.array([0, 2]),
    )
    np.testing.assert_array_equal(
        space.fixed_indices,
        np.array([1]),
    )
    assert space.fixed_values[0] == 2.0
    assert space.d_free == 2


def test_multiple_fixed_parameters():
    space = ParameterSpace.from_prior(
        prior_mean=np.array([1.0, 2.0, 3.0]),
        prior_variance=np.array([0.0, 5.0, 0.0]),
    )

    np.testing.assert_array_equal(
        space.fixed_values,
        np.array([1.0, 3.0]),
    )
    assert space.d_free == 1


def test_all_parameters_fixed():
    space = ParameterSpace.from_prior(
        prior_mean=np.array([1.0, 2.0]),
        prior_variance=np.array([0.0, 0.0]),
    )

    assert space.d_free == 0
    assert space.free_covariance.shape == (0, 0)
    assert space.free_precision.shape == (0, 0)


def test_free_covariance_is_correct_submatrix():
    cov = np.array([
        [4.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 3.0],
    ])

    space = ParameterSpace.from_prior(
        prior_mean=np.zeros(3),
        prior_variance=cov,
    )

    expected = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ])

    np.testing.assert_allclose(
        space.free_covariance,
        expected,
    )
    np.testing.assert_allclose(
        space.free_precision,
        np.linalg.inv(expected),
    )


def test_negative_prior_variance_raises():
    with pytest.raises(ValueError, match="non-negative"):
        ParameterSpace.from_prior(
            prior_mean=np.zeros(2),
            prior_variance=np.array([1.0, -1.0]),
        )


def test_wrong_prior_variance_length_raises():
    with pytest.raises(ValueError):
        ParameterSpace.from_prior(
            prior_mean=np.zeros(2),
            prior_variance=np.ones(3),
        )


def test_fixed_parameter_cannot_have_nonzero_covariance():
    cov = np.array([
        [0.0, 0.2],
        [0.2, 1.0],
    ])

    with pytest.raises(ValueError, match="zero prior variance"):
        ParameterSpace.from_prior(
            prior_mean=np.zeros(2),
            prior_variance=cov,
        )


def test_all_free_constructor_preserves_precision():
    precision = np.array([
        [2.0, 0.2],
        [0.2, 1.5],
    ])

    space = ParameterSpace.all_free(
        prior_mean=np.array([0.5, -0.5]),
        prior_precision=precision,
    )

    np.testing.assert_allclose(
        space.free_precision,
        precision,
    )
    np.testing.assert_allclose(
        space.full_precision,
        precision,
    )
