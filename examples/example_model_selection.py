import numpy as np
from code.methods.utils.cbm_python.cbm.individual_fit import individual_fit
from code.methods.utils.cbm_python.cbm.model_selection import bms
import pickle
from pathlib import Path

# Save outputs to examples/output
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

    y_pred = X * slope  # No intercept
    sigma = np.exp(log_sigma)

    log_lik = -0.5 * np.sum((y - y_pred) ** 2 / sigma ** 2) - len(y) * np.log(sigma * np.sqrt(2 * np.pi))

    return log_lik

def linear_model_no_sigma(parameters, data):
    """
    Model 1: Linear regression with intercept
    y ~ N(X*slope + intercept, sigma^2)
    parameters = [slope, intercept, log(sigma)]
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
N_subjects = 5  # More subjects for better model comparison
data = []
true_parameters = []

X = np.linspace(0, 10, 50)
noise = np.random.randn(len(X))

for i in range(N_subjects):
    # Generate data WITH intercept (so Model 1 should win)
    true_slope = 2.0 + np.random.randn() * 0.5
    true_intercept = 1.0 + np.random.randn() * 0.5
    true_log_sigma = np.log(0.5)

    y = true_slope * X + true_intercept + noise * np.exp(true_log_sigma)

    data.append((X, y))
    true_parameters.append([true_slope, true_intercept, true_log_sigma])

for i in range(N_subjects):
    # Generate data WITH intercept (so Model 1 should win)
    true_slope = 4 + np.random.randn() * 0.5
    true_intercept = np.random.randn() * 0.5
    true_log_sigma = np.log(0.5)

    y = true_slope * X + true_intercept + noise * np.exp(true_log_sigma)

    data.append((X, y))
    true_parameters.append([true_slope, true_intercept, true_log_sigma])

true_parameters = np.array(true_parameters)


# ============================================================================
# Fit Model 1: With intercept
# ============================================================================

print("+" * 70)
print("Fitting Model 1: Linear regression WITH intercept")
print("+" * 70)

prior_mean_m1 = np.array([0, 0, 0])  # slope, intercept, log(sigma)
prior_variance_m1 = 10

cbm1 = individual_fit(data, linear_model_with_intercept,
                      prior_mean_m1, prior_variance_m1)

print(f"\nModel 1 - Log model evidence:\n{cbm1.output.log_evidence}")

# ============================================================================
# Fit Model 2: No intercept
# ============================================================================

print("\n" + "+" * 70)
print("Fitting Model 2: Linear regression WITHOUT intercept")
print("+" * 70)

prior_mean_m2 = np.array([0, 0])  # slope, log(sigma)
prior_variance_m2 = 10

cbm2 = individual_fit(data, linear_model_no_intercept,
                      prior_mean_m2, prior_variance_m2)

print(f"\nModel 2 - Log model evidence:\n{cbm2.output.log_evidence}")

# ============================================================================
# Fit Model 3: No sigma
# ============================================================================

print("\n" + "+" * 70)
print("Fitting Model 3: Linear regression WITHOUT intercept")
print("+" * 70)

prior_mean_m3 = np.array([0, 0])  # slope, intercept
prior_variance_m3 = 10

cbm3 = individual_fit(data, linear_model_no_sigma,
                      prior_mean_m3, prior_variance_m3)

print(f"\nModel 3 - Log model evidence:\n{cbm3.output.log_evidence}")

# ============================================================================
# Bayesian Model Selection
# ============================================================================

print("\n" + "+" * 70)
print("Bayesian Model Selection")
print("+" * 70)

# Create log model evidence matrix: subjects × models
lme = np.column_stack([cbm1.output.log_evidence, cbm2.output.log_evidence, cbm3.output.log_evidence])

# Run BMS
bms_result = bms(lme)

print(f"\nExpected model frequencies:")
print(f"  Model 1 (with intercept):    {bms_result.model_frequency[0]:.4f}")
print(f"  Model 2 (without intercept): {bms_result.model_frequency[1]:.4f}")
print(f"  Model 3 (without sigma): {bms_result.model_frequency[2]:.4f}")

print(f"\nExceedance probabilities:")
print(f"  Model 1 (with intercept):    {bms_result.exceedance_prob[0]:.4f}")
print(f"  Model 2 (without intercept): {bms_result.exceedance_prob[1]:.4f}")
print(f"  Model 3 (without sigma): {bms_result.exceedance_prob[2]:.4f}")

print(f"\nProtected exceedance probabilities:")
print(f"  Model 1 (with intercept):    {bms_result.protected_exceedance_prob[0]:.4f}")
print(f"  Model 2 (without intercept): {bms_result.protected_exceedance_prob[1]:.4f}")
print(f"  Model 3 (without sigma): {bms_result.protected_exceedance_prob[2]:.4f}")


