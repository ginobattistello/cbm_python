import matplotlib.figure
import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config


def _model(theta, data):
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["X"]["x"], dtype=float)
    mu = theta[0] + theta[1] * x
    return -0.5 * (y - mu) ** 2


def _observation(theta, data):
    x = np.asarray(data["X"]["x"], dtype=float)
    return theta[0] + theta[1] * x


def _data():
    return [{
        "y": np.array([-1.0, 0.0, 1.0]),
        "X": {"x": np.array([-1.0, 0.0, 1.0])},
    }]


def test_subject_plot_returns_figure():
    fit = individual_fit(
        _data(),
        _model,
        observation=_observation,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 10,
        config=Config(
            d=2,
            num_init=2,
            random_state=1,
            verbose=False,
            display=True,
        ),
    )

    fig = fit.plot(
        subject=0,
        display=False,
    )

    assert isinstance(
        fig,
        matplotlib.figure.Figure,
    )


def test_reporting_summary_does_not_mutate_fit():
    fit = individual_fit(
        _data(),
        _model,
        prior_mean=np.zeros(2),
        prior_variance=np.ones(2) * 10,
        config=Config(
            d=2,
            num_init=2,
            random_state=1,
            verbose=False,
            display=False,
        ),
    )

    theta_before = fit.output.parameters.copy()
    evidence_before = fit.output.log_evidence.copy()

    _ = fit.summary()
    _ = fit.table()

    np.testing.assert_array_equal(
        theta_before,
        fit.output.parameters,
    )
    np.testing.assert_array_equal(
        evidence_before,
        fit.output.log_evidence,
    )
