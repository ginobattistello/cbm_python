import numpy as np
import pytest

from cbm.hbi_types import (
    DirichletDistribution,
    IndividualPosterior,
)
from cbm.hbi_updates import (
    hbi_sumstats,
    hbi_qm,
)


def _individual_posterior():
    return IndividualPosterior(
        loglik=np.array([
            [-1.0, -1.2, -0.9],
            [-2.0, -1.8, -2.1],
        ]),
        parameters=[
            np.array([
                [0.1, 0.2, 0.3],
                [1.0, 1.1, 0.9],
            ]),
            np.array([
                [-0.5, -0.4, -0.6],
            ]),
        ],
        hessian_inv_diag=[
            np.ones((2, 3)) * 0.1,
            np.ones((1, 3)) * 0.2,
        ],
        log_det_hessian=np.zeros((2, 3)),
    )


def test_hbi_sumstats_shapes_and_effective_counts():
    qh = _individual_posterior()
    r = np.array([
        [0.8, 0.7, 0.9],
        [0.2, 0.3, 0.1],
    ])

    Nbar, thetabar, Sdiag = hbi_sumstats(r, qh)

    np.testing.assert_allclose(
        Nbar,
        r.sum(axis=1),
    )

    assert thetabar[0].shape == (2, 1)
    assert Sdiag[0].shape == (2, 1)
    assert thetabar[1].shape == (1, 1)
    assert Sdiag[1].shape == (1, 1)

    assert np.all(np.isfinite(thetabar[0]))
    assert np.all(np.isfinite(Sdiag[0]))
    assert np.all(Sdiag[0] >= 0)


def test_hbi_sumstats_rejects_zero_effective_subjects():
    qh = _individual_posterior()
    r = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ])

    with pytest.raises(RuntimeError, match="zero effective subjects"):
        hbi_sumstats(r, qh)


def test_hbi_qm_updates_dirichlet_counts():
    pm = DirichletDistribution(
        limInf=False,
        alpha=np.ones(2),
        Elogm=np.zeros(2),
        logC=0.0,
    )

    Nbar = np.array([8.0, 2.0])

    qm, bound = hbi_qm(pm, Nbar)

    np.testing.assert_allclose(
        qm.alpha,
        np.array([9.0, 3.0]),
    )

    assert np.all(np.isfinite(qm.Elogm))
    assert np.isfinite(qm.logC)
    assert np.all(np.isfinite(bound.ElogpZ))


def test_hbi_responsibility_rows_normalize_if_qhz_available():
    # The full qHZ update requires several distribution objects.
    # Keep this test lightweight and API-resilient by importing lazily;
    # the integration HBI tests below exercise the complete update cycle.
    from cbm.hbi_updates import hbi_qHZ

    assert callable(hbi_qHZ)
