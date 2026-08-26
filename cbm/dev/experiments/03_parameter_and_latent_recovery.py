"""Joint parameter and latent-variable recovery experiment.

Question
--------
When data are generated from a reinforcement-learning model, can CBM recover:
1. the subject-level parameters; and
2. the trialwise latent variables reconstructed at the fitted MAP?

The experiment stores the true Q values and prediction errors used during
simulation, then compares them with evolution(theta_MAP, data).

Run
---
python cbm/dev/03_parameter_and_latent_recovery.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from cbm.individual_fit import individual_fit
from cbm.optimization import Config


VERBOSE = True
DISPLAY = True

N_SUBJECTS = 20
N_TRIALS = 200
SEED = 7


def softmax(x):
    z = np.asarray(x) - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def simulate_subject(rng, theta, n_trials=N_TRIALS):
    """Binary RW simulation returning both observed data and latent truth."""
    alpha, beta = theta
    reward_probability = np.array([0.75, 0.25])

    q = np.zeros(2)

    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials, dtype=float)

    q_pre = np.zeros((n_trials, 2), dtype=float)
    prediction_error = np.zeros(n_trials, dtype=float)

    for t in range(n_trials):
        q_pre[t] = q

        p = softmax(beta * q)
        choice = rng.choice(2, p=p)
        reward = float(
            rng.random()
            < reward_probability[choice]
        )

        pe = reward - q[choice]

        choices[t] = choice
        rewards[t] = reward
        prediction_error[t] = pe

        q[choice] += alpha * pe

    data = {
        "y": choices,
        "X": {
            "reward": rewards,
        },
    }

    truth = {
        "Q0": q_pre[:, 0],
        "Q1": q_pre[:, 1],
        "prediction_error": prediction_error,
    }

    return data, truth


def model(theta, data):
    """Per-trial binary RW log-likelihood."""
    alpha, beta = theta
    choices = np.asarray(
        data["y"],
        dtype=int,
    )
    rewards = np.asarray(
        data["X"]["reward"],
        dtype=float,
    )

    q = np.zeros(2)
    logp = np.zeros(len(choices))

    for t, choice in enumerate(choices):
        p = softmax(beta * q)
        logp[t] = np.log(
            p[choice] + 1e-12
        )

        pe = rewards[t] - q[choice]
        q[choice] += alpha * pe

    return logp


def observation(theta, data):
    """Return P(choice=1) on each trial."""
    alpha, beta = theta
    choices = np.asarray(
        data["y"],
        dtype=int,
    )
    rewards = np.asarray(
        data["X"]["reward"],
        dtype=float,
    )

    q = np.zeros(2)
    p1 = np.zeros(len(choices))

    for t, choice in enumerate(choices):
        p = softmax(beta * q)
        p1[t] = p[1]

        pe = rewards[t] - q[choice]
        q[choice] += alpha * pe

    return p1


def evolution(theta, data):
    """Reconstruct trialwise latent variables at supplied theta."""
    alpha, _ = theta
    choices = np.asarray(
        data["y"],
        dtype=int,
    )
    rewards = np.asarray(
        data["X"]["reward"],
        dtype=float,
    )

    q = np.zeros(2)

    q_pre = np.zeros((len(choices), 2))
    prediction_error = np.zeros(len(choices))

    for t, choice in enumerate(choices):
        q_pre[t] = q

        pe = rewards[t] - q[choice]
        prediction_error[t] = pe

        q[choice] += alpha * pe

    return {
        "Q0": q_pre[:, 0],
        "Q1": q_pre[:, 1],
        "prediction_error": prediction_error,
    }


def recovery_stats(true, estimated):
    true = np.asarray(true, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    r = np.corrcoef(
        true,
        estimated,
    )[0, 1]

    bias = np.mean(
        estimated - true
    )

    rmse = np.sqrt(
        np.mean(
            (estimated - true) ** 2
        )
    )

    return r, bias, rmse


def header(title):
    print("=" * 82)
    print(title)
    print("=" * 82)


def main():
    rng = np.random.default_rng(SEED)

    true_theta = np.column_stack([
        rng.uniform(
            0.15,
            0.75,
            N_SUBJECTS,
        ),
        rng.uniform(
            1.5,
            5.0,
            N_SUBJECTS,
        ),
    ])

    data = []
    latent_truth = []

    for theta in true_theta:
        dat, latent = simulate_subject(
            rng,
            theta,
        )
        data.append(dat)
        latent_truth.append(latent)

    header("PARAMETER + LATENT VARIABLE RECOVERY")
    print(f"Subjects: {N_SUBJECTS}")
    print(f"Trials per subject: {N_TRIALS}")
    print("Model: binary Rescorla-Wagner + softmax")
    print("-" * 82)

    fit = individual_fit(
        data=data,
        model=model,
        observation=observation,
        evolution=evolution,
        prior_mean=np.array([
            0.5,
            2.5,
        ]),
        prior_variance=np.array([
            4.0,
            64.0,
        ]),
        config=Config(
            d=2,
            range_bounds=np.array([
                [0.05, 0.25],
                [0.95, 8.00],
            ]),
            hard_bounds=np.array([
                [0.001, 0.01],
                [0.999, 20.0],
            ]),
            num_init=5,
            random_state=SEED,
            hessian_method="central_fd",
            verbose=False,
            # Needed to retain individual display diagnostics.
            display=True,
        ),
    )

    estimated_theta = fit.output.parameters

    alpha_stats = recovery_stats(
        true_theta[:, 0],
        estimated_theta[:, 0],
    )
    beta_stats = recovery_stats(
        true_theta[:, 1],
        estimated_theta[:, 1],
    )

    true_q = []
    estimated_q = []

    true_pe = []
    estimated_pe = []

    for n in range(N_SUBJECTS):
        latent_hat = fit.output.latent[n]

        q_true_n = np.column_stack([
            latent_truth[n]["Q0"],
            latent_truth[n]["Q1"],
        ])

        q_hat_n = np.column_stack([
            latent_hat["Q0"],
            latent_hat["Q1"],
        ])

        true_q.append(q_true_n.reshape(-1))
        estimated_q.append(q_hat_n.reshape(-1))

        true_pe.append(
            latent_truth[n]["prediction_error"]
        )
        estimated_pe.append(
            latent_hat["prediction_error"]
        )

    true_q = np.concatenate(true_q)
    estimated_q = np.concatenate(estimated_q)

    true_pe = np.concatenate(true_pe)
    estimated_pe = np.concatenate(estimated_pe)

    q_stats = recovery_stats(
        true_q,
        estimated_q,
    )

    pe_stats = recovery_stats(
        true_pe,
        estimated_pe,
    )

    if VERBOSE:
        print(
            f"{'quantity':<20} {'r':>10} {'bias':>12} {'RMSE':>12}"
        )
        print("-" * 58)
        print(
            f"{'alpha':<20} "
            f"{alpha_stats[0]:10.4f} "
            f"{alpha_stats[1]:12.4f} "
            f"{alpha_stats[2]:12.4f}"
        )
        print(
            f"{'beta':<20} "
            f"{beta_stats[0]:10.4f} "
            f"{beta_stats[1]:12.4f} "
            f"{beta_stats[2]:12.4f}"
        )
        print(
            f"{'Q values':<20} "
            f"{q_stats[0]:10.4f} "
            f"{q_stats[1]:12.4f} "
            f"{q_stats[2]:12.4f}"
        )
        print(
            f"{'prediction error':<20} "
            f"{pe_stats[0]:10.4f} "
            f"{pe_stats[1]:12.4f} "
            f"{pe_stats[2]:12.4f}"
        )
        print("-" * 82)
        print(
            f"Valid Laplace fits: "
            f"{sum(d.laplace_valid for d in fit.math.diagnostics)}"
            f"/{N_SUBJECTS}"
        )

    if DISPLAY:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12.0, 3.8),
        )

        ax = axes[0]
        ax.scatter(
            true_theta[:, 0],
            estimated_theta[:, 0],
        )
        lo = min(
            true_theta[:, 0].min(),
            estimated_theta[:, 0].min(),
        )
        hi = max(
            true_theta[:, 0].max(),
            estimated_theta[:, 0].max(),
        )
        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle=":",
        )
        ax.set_title(
            f"A  alpha recovery · r={alpha_stats[0]:.2f}"
        )
        ax.set_xlabel("true alpha")
        ax.set_ylabel("MAP alpha")

        ax = axes[1]
        ax.scatter(
            true_theta[:, 1],
            estimated_theta[:, 1],
        )
        lo = min(
            true_theta[:, 1].min(),
            estimated_theta[:, 1].min(),
        )
        hi = max(
            true_theta[:, 1].max(),
            estimated_theta[:, 1].max(),
        )
        ax.plot(
            [lo, hi],
            [lo, hi],
            linestyle=":",
        )
        ax.set_title(
            f"B  beta recovery · r={beta_stats[0]:.2f}"
        )
        ax.set_xlabel("true beta")
        ax.set_ylabel("MAP beta")

        n = 0
        trial = np.arange(N_TRIALS)

        ax = axes[2]
        ax.plot(
            trial,
            latent_truth[n]["Q0"],
            label="true Q0",
        )
        ax.plot(
            trial,
            fit.output.latent[n]["Q0"],
            linestyle="--",
            label="MAP Q0",
        )
        ax.plot(
            trial,
            latent_truth[n]["Q1"],
            label="true Q1",
        )
        ax.plot(
            trial,
            fit.output.latent[n]["Q1"],
            linestyle="--",
            label="MAP Q1",
        )
        ax.set_title(
            f"C  latent recovery · pooled r={q_stats[0]:.3f}"
        )
        ax.set_xlabel("trial")
        ax.set_ylabel("Q value")
        ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()

        # Also show the toolbox's own subject diagnostic display.
        fit.plot(
            subject=0,
            display=True,
        )

        plt.show()


if __name__ == "__main__":
    main()
