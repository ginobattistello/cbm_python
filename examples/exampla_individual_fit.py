import numpy as np
from code.methods.utils.cbm_python.cbm.individual_fit import individual_fit
import pickle
from pathlib import Path

def linear_model(parameters, data):
    """
    This current model is a simple linear regresion model: y ~ N(X*parameters, sigma^2)
    parameters = [slope, intercept, log(sigma)]
    all parameters live in the real range (of Gaussian distribition), so you should properly transform them inside
    your model

    replace this with your model, but the format should be the same in terms of input and output. data is Any format.
    """
    X, y = data
    slope, intercept, log_sigma = parameters

    y_pred = X * slope + intercept
    sigma = np.exp(log_sigma)  # sigma is positive

    # Log-likelihood of Gaussian
    log_lik = -0.5 * np.sum((y - y_pred) ** 2 / sigma ** 2) - len(y) * np.log(sigma * np.sqrt(2 * np.pi))

    return log_lik


# Generate synthetic data for 5 subjects
np.random.seed(42)
N_subjects = 5
data = []
true_parameters = []

X = np.linspace(0, 10, 50)
noise = np.random.randn(len(X))

for i in range(N_subjects):
    true_slope = 2.0 + np.random.randn() * 0.5
    true_intercept = 1.0 + np.random.randn() * 0.5
    true_log_sigma = np.log(0.5)

    y = true_slope * X + true_intercept + noise * np.exp(true_log_sigma)
    data.append((X, y))
    true_parameters.append([true_slope, true_intercept, true_log_sigma])

true_parameters = np.array(true_parameters)

# define prior_mean (usually 0), and prior_variance. prior_mean size should be the same as number of parameters
prior_mean = np.array([0, 0, 0])
prior_variance = 10

# run individual_fit and (optionally) save it to a pickle file
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_file = OUT_DIR / 'example_individual_fit.pkl'
cbm = individual_fit(data, linear_model, prior_mean, prior_variance, fname=str(out_file))

with open(out_file, 'rb') as f:
    cbm = pickle.load(f)

print(f"\nTrue parameters:")
print(true_parameters)
print(f"\nFitted parameters:")
print(cbm.output.parameters)
print(f"\nLog model evidence:\n{cbm.output.log_evidence}")


