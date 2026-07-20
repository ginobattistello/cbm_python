import numpy as np
import pickle
from pathlib import Path

from code.methods.utils.cbm_python.cbm.hbi import hbi_main, hbi_null
from code.methods.utils.cbm_python.cbm.individual_fit import individual_fit
from code.methods.utils.cbm_python.cbm.model_selection import bms

# Paths
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Define three competing models
# ============================================================================

def linear_model_with_intercept(parameters, data):
    """
    Model 1: Linear regression with intercept
    y ~ N(X*slope + intercept, sigma^2)
    parameters = [slope, intercept, log(sigma)]
    """
    X, y = data
    slope, intercept, log_sigma = parameters

    y_pred = X * slope + intercept
    sigma = np.exp(log_sigma)

    log_lik = -0.5 * np.sum((y - y_pred) ** 2 / sigma ** 2) - len(y) * np.log(sigma * np.sqrt(2 * np.pi))
    return log_lik


def linear_model_no_intercept(parameters, data):
    """
    Model 2: Linear regression without intercept (through origin)
    y ~ N(X*slope, sigma^2)
    parameters = [slope, log(sigma)]
    """
    X, y = data
    slope, log_sigma = parameters

    y_pred = X * slope
    sigma = np.exp(log_sigma)

    log_lik = -0.5 * np.sum((y - y_pred) ** 2 / sigma ** 2) - len(y) * np.log(sigma * np.sqrt(2 * np.pi))
    return log_lik


def linear_model_no_sigma(parameters, data):
    """
    Model 3: Linear regression with intercept and fixed sigma (sigma=exp(0))
    parameters = [slope, intercept]
    """
    X, y = data
    slope, intercept = parameters

    y_pred = X * slope + intercept
    sigma = np.exp(0)

    log_lik = -0.5 * np.sum((y - y_pred) ** 2 / sigma ** 2) - len(y) * np.log(sigma * np.sqrt(2 * np.pi))
    return log_lik

# ============================================================================
# Generate synthetic data
# ============================================================================

np.random.seed(42)
N_subjects = 5  # number of subjects per group
all_data = []
true_parameters = []

X = np.linspace(0, 10, 50)
noise = np.random.randn(len(X))

# Group 1: with intercept
for _ in range(N_subjects):
    true_slope = 2.0 + np.random.randn() * 0.5
    true_intercept = 1.0 + np.random.randn() * 0.5
    true_log_sigma = np.log(0.5)
    y = true_slope * X + true_intercept + noise * np.exp(true_log_sigma)
    all_data.append((X, y))
    true_parameters.append([true_slope, true_intercept, true_log_sigma])

# Group 2: no intercept (intercept ~ 0)
for _ in range(N_subjects):
    true_slope = 4.0 + np.random.randn() * 0.5
    true_intercept = 0.0
    true_log_sigma = np.log(0.5)
    y = true_slope * X + true_intercept + noise * np.exp(true_log_sigma)
    all_data.append((X, y))
    true_parameters.append([true_slope, true_intercept, true_log_sigma])

true_parameters = np.array(true_parameters)

# # ============================================================================
# # Fit models
# # ============================================================================
print("\n" + "+" * 70)
print("Running individual model fits")

prior_mean_m1 = np.array([0, 0, 0])
prior_variance_m1 = 10
cbm1 = individual_fit(all_data, linear_model_with_intercept, prior_mean_m1, prior_variance_m1)
with open(OUT_DIR / "cbm1.pkl", "wb") as f:
    pickle.dump(cbm1, f)

prior_mean_m2 = np.array([0, 0])
prior_variance_m2 = 10
cbm2 = individual_fit(all_data, linear_model_no_intercept, prior_mean_m2, prior_variance_m2)
with open(OUT_DIR / "cbm2.pkl", "wb") as f:
    pickle.dump(cbm2, f)

prior_mean_m3 = np.array([0, 0])
prior_variance_m3 = 10
cbm3 = individual_fit(all_data, linear_model_no_sigma, prior_mean_m3, prior_variance_m3)
with open(OUT_DIR / "cbm3.pkl", "wb") as f:
    pickle.dump(cbm3, f)

# # ============================================================================
# # Bayesian Model Selection
# # ============================================================================

print("\n" + "+" * 70)
print("Bayesian Model Selection based on individual fits")

lme = np.column_stack([cbm1.output.log_evidence, cbm2.output.log_evidence, cbm3.output.log_evidence])
bms_result = bms(lme)

print(f"\nExpected model frequencies:")
print(f"  Model 1 (with intercept):    {bms_result.model_frequency[0]:.4f}")
print(f"  Model 2 (without intercept): {bms_result.model_frequency[1]:.4f}")
print(f"  Model 3 (without sigma):     {bms_result.model_frequency[2]:.4f}")

print(f"\nExceedance probabilities:")
print(f"  Model 1 (with intercept):    {bms_result.exceedance_prob[0]:.4f}")
print(f"  Model 2 (without intercept): {bms_result.exceedance_prob[1]:.4f}")
print(f"  Model 3 (without sigma):     {bms_result.exceedance_prob[2]:.4f}")

print(f"\nProtected exceedance probabilities:")
print(f"  Model 1 (with intercept):    {bms_result.protected_exceedance_prob[0]:.4f}")
print(f"  Model 2 (without intercept): {bms_result.protected_exceedance_prob[1]:.4f}")
print(f"  Model 3 (without sigma):     {bms_result.protected_exceedance_prob[2]:.4f}")

# ============================================================================
# HBI
# ============================================================================

cbm_maps = [str(OUT_DIR / 'cbm1.pkl'), str(OUT_DIR / 'cbm2.pkl'), str(OUT_DIR / 'cbm3.pkl')]
config = {
    "save_prog": False,    
}
models = [linear_model_with_intercept, linear_model_no_intercept, linear_model_no_sigma]

cbm = hbi_main(all_data, models, cbm_maps, fname=str(OUT_DIR / "hbi.pkl"), config=config)

print("\n" + "+" * 70)
print("HBI Results")
print("+" * 70)

print(f"\nModel frequencies:")
for k, freq in enumerate(cbm.output.model_frequency):
    print(f"  Model {k+1}: {freq:.4f}")

print(f"\nExceedance probabilities:")
for k, xp in enumerate(cbm.output.exceedance_prob):
    print(f"  Model {k+1}: {xp:.8f}")

print(f"\nGroup-level parameter means:")
for k, mean in enumerate(cbm.output.group_mean):
    print(f"  Model {k+1}: {mean}")

print(f"\nGroup-level hierarchical error bars:")
for k, he in enumerate(cbm.output.group_hierarchical_errorbar):
    print(f"  Model {k+1}: {he}")

# ============================================================================
# HBI Null model for protected exceedance probabilities
# ============================================================================
cbm = hbi_null(all_data, str(OUT_DIR / 'hbi.pkl'))

print(f"\nProtected exceedance probabilities:")
for k, pxp in enumerate(cbm.output.protected_exceedance_prob):
    print(f"  Model {k+1}: {pxp:.8f}")