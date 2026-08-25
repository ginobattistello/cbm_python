import numpy as np
import pytest

from cbm.optimization import (
    BFGSOptimizer,
    Config,
    ConvergenceStatus,
)


def test_lbfgsb_recovers_known_optimum():
    def objective(theta):
        theta = np.asarray(theta)
        return (
            (theta[0] - 1.0) ** 2
            + (theta[1] + 2.0) ** 2
        )

    result = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=4,
            random_state=11,
            verbose=False,
        ),
    ).optimize(objective)

    np.testing.assert_allclose(
        result.x,
        np.array([1.0, -2.0]),
        atol=1e-5,
    )

    assert result.success
    assert result.lbfgsb_status is not None
    assert isinstance(result.lbfgsb_message, str)
    assert result.lbfgsb_message


def test_random_state_reproduces_initializations():
    def objective(theta):
        return np.sum((np.asarray(theta) - 0.2) ** 2)

    cfg1 = Config(
        d=2,
        num_init=5,
        random_state=123,
        verbose=False,
    )
    cfg2 = Config(
        d=2,
        num_init=5,
        random_state=123,
        verbose=False,
    )

    opt1 = BFGSOptimizer(2, cfg1)
    opt2 = BFGSOptimizer(2, cfg2)

    opt1.optimize(objective)
    opt2.optimize(objective)

    starts1 = np.vstack([
        r.x_init
        for r in opt1.get_all_results()
    ])
    starts2 = np.vstack([
        r.x_init
        for r in opt2.get_all_results()
    ])

    np.testing.assert_array_equal(
        starts1,
        starts2,
    )


def test_different_seeds_change_random_starts():
    def objective(theta):
        return np.sum(np.asarray(theta) ** 2)

    opt1 = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=4,
            random_state=1,
            verbose=False,
        ),
    )
    opt2 = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=4,
            random_state=2,
            verbose=False,
        ),
    )

    opt1.optimize(objective)
    opt2.optimize(objective)

    starts1 = np.vstack([
        r.x_init
        for r in opt1.get_all_results()
    ])
    starts2 = np.vstack([
        r.x_init
        for r in opt2.get_all_results()
    ])

    assert not np.array_equal(
        starts1,
        starts2,
    )


def test_scalar_model_skips_gn():
    def objective(theta):
        return np.sum((np.asarray(theta) - 1.0) ** 2)

    result = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=2,
            random_state=1,
            verbose=False,
        ),
    ).optimize(objective)

    assert (
        result.convergence_status
        == ConvergenceStatus.SKIPPED_NO_TRIAL_FUNC
    )
    assert result.gn_condition_number is None
    assert result.gn_is_positive_definite is None
    assert result.gn_ill_conditioned is None


def test_trialwise_model_activates_gn():
    target = np.array([0.5, -0.25])

    def objective(theta):
        theta = np.asarray(theta)
        return np.sum((theta - target) ** 2)

    def trial_func(theta):
        theta = np.asarray(theta)
        return np.array([
            theta[0] - target[0],
            theta[1] - target[1],
            0.5 * (theta[0] + theta[1]),
        ])

    result = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=2,
            random_state=2,
            verbose=False,
        ),
    ).optimize(
        objective,
        trial_func=trial_func,
        prior_precision=np.eye(2),
    )

    assert (
        result.convergence_status
        != ConvergenceStatus.SKIPPED_NO_TRIAL_FUNC
    )


def test_gn_ill_conditioning_is_detected():
    def objective(theta):
        return 0.5 * np.sum(
            (np.asarray(theta) - 1.0) ** 2
        )

    def trial_func(theta):
        theta = np.asarray(theta)

        # J.T @ J has eigenvalues approximately
        # [1, 1e-16], hence condition number ~1e16.
        return np.array([
            theta[0],
            1e-8 * theta[1],
        ])

    optimizer = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=1,
            condition_number_warn=1e10,
            random_state=1,
            verbose=False,
        ),
    )

    (
        x,
        f,
        status,
        n_steps,
        gn_diag,
    ) = optimizer._newton_polish(
        objective,
        np.array([0.0, 0.0]),
        trial_func=trial_func,
        prior_precision=np.zeros((2, 2)),
    )

    assert (
        status
        == ConvergenceStatus.ILL_CONDITIONED_CURVATURE
    )

    assert gn_diag is not None

    # Complete GN diagnostic state.
    assert gn_diag["is_positive_definite"] is True
    assert gn_diag["ill_conditioned"] is True

    assert gn_diag["min_eig"] > 0
    assert gn_diag["max_eig"] > gn_diag["min_eig"]

    assert (
        gn_diag["condition_number"]
        > 1e10
    )

    # GN correctly refuses the polish step.
    assert n_steps == 0


def test_invalid_objective_evaluations_are_recorded():
    def objective(theta):
        theta = np.asarray(theta)
        if theta[0] < 0:
            raise ValueError("outside domain")
        return (
            (theta[0] - 0.5) ** 2
            + theta[1] ** 2
        )

    result = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=2,
            inits=np.array([
                [-0.5, 0.0],
                [0.5, 0.0],
            ]),
            random_state=1,
            verbose=False,
        ),
    ).optimize(objective)

    assert result.n_invalid_evaluations > 0
    assert result.first_invalid_evaluation is not None
    assert "outside domain" in result.first_invalid_evaluation
    assert result.flag == 0.5


def test_unsuccessful_lbfgsb_is_explicitly_retained():
    def objective(theta):
        return np.sum((np.asarray(theta) - 1.0) ** 2)

    result = BFGSOptimizer(
        2,
        Config(
            d=2,
            num_init=1,
            inits=np.array([[5.0, 5.0]]),
            max_iter=0,
            random_state=1,
            verbose=False,
        ),
    ).optimize(objective)

    assert result.lbfgsb_status is not None
    assert isinstance(result.lbfgsb_message, str)
    assert result.lbfgsb_message
    assert result.flag <= 0.5


def test_display_tracking_does_not_change_numerical_result():
    def objective(theta):
        return np.sum((np.asarray(theta) - 0.3) ** 2)

    base = dict(
        d=2,
        num_init=3,
        random_state=123,
        verbose=False,
    )

    a = BFGSOptimizer(
        2,
        Config(**base, display=False),
    ).optimize(objective)

    b = BFGSOptimizer(
        2,
        Config(**base, display=True),
    ).optimize(objective)

    np.testing.assert_allclose(a.x, b.x, atol=1e-12)
    np.testing.assert_allclose(a.f, b.f, atol=1e-12)
    np.testing.assert_allclose(a.hess, b.hess, atol=1e-12)

    assert a.is_hess_pos == b.is_hess_pos
    assert a.hess_ill_conditioned == b.hess_ill_conditioned
    assert a.laplace_valid == b.laplace_valid
    assert a.laplace_fragile == b.laplace_fragile
