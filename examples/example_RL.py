import numpy as np
import pickle
from pathlib import Path

from code.methods.utils.cbm_python.cbm.hbi import hbi_main, hbi_null
from code.methods.utils.cbm_python.cbm.individual_fit import individual_fit
from code.methods.utils.cbm_python.cbm.model_selection import bms
from code.methods.utils.cbm_python.cbm.optimization import Config

# Paths
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Define two competing reinforcement learning models
# ============================================================================

def RL_model(parameters, data):
    """
    Model 1: Standard Rescorla-Wagner model with single learning rate
    Q-values updated with: Q(a) = Q(a) + alpha * (r - Q(a))
    parameters = [alpha, beta] where:
        alpha: learning rate (0-1)
        beta: inverse temperature (softmax parameter)
    """
    choices, rewards = data
    alpha = 1 / (1 + np.exp(-parameters[0]))  # sigmoid transform to (0,1)
    beta = np.exp(parameters[1])  # exp transform to ensure positive
    
    n_trials = len(choices)
    n_options = 2
    Q = np.zeros(n_options)
    
    log_lik = 0.0
    for t in range(n_trials):
        # Softmax choice probability
        action = int(choices[t])
        arg_exp = beta * Q
        max_arg = np.max(arg_exp)
        exp_values = np.exp(arg_exp - max_arg)  # for numerical stability
        p = exp_values / np.sum(exp_values)
        log_lik += np.log(p[action] + 1e-10) # add small constant for numerical stability
        
        # Update Q-value
        reward = rewards[t]
        delta = reward - Q[action]
        Q[action] = Q[action] + alpha * delta

    return log_lik


def RL_model_trials(parameters, data):
    """
    Same model as RL_model, but returns the per-trial log-likelihood
    array instead of the summed scalar (sum(...) == RL_model(...)).
    Enables the VBA-style Gauss-Newton curvature (optimization.py Mod 5)
    instead of the finite-difference/eigenvalue-clip fallback (Mod 2).
    """
    choices, rewards = data
    alpha = 1 / (1 + np.exp(-parameters[0]))
    beta = np.exp(parameters[1])

    n_trials = len(choices)
    n_options = 2
    Q = np.zeros(n_options)

    log_liks = np.zeros(n_trials)
    for t in range(n_trials):
        action = int(choices[t])
        arg_exp = beta * Q
        max_arg = np.max(arg_exp)
        exp_values = np.exp(arg_exp - max_arg)
        p = exp_values / np.sum(exp_values)
        log_liks[t] = np.log(p[action] + 1e-10)

        reward = rewards[t]
        delta = reward - Q[action]
        Q[action] = Q[action] + alpha * delta

    return log_liks


def RL2_model(parameters, data):
    """
    Model 2: RL model with separate learning rates for positive and negative PEs
    Q-values updated with: Q(a) = Q(a) + alpha_pos * PE  if PE >= 0
                                   Q(a) + alpha_neg * PE  if PE < 0
    parameters = [alpha_pos, alpha_neg, beta] where:
        alpha_pos: learning rate for positive prediction errors (0-1)
        alpha_neg: learning rate for negative prediction errors (0-1)
        beta: inverse temperature (softmax parameter)
    """
    choices, rewards = data
    alpha_pos = 1 / (1 + np.exp(-parameters[0]))  # sigmoid to (0,1)
    alpha_neg = 1 / (1 + np.exp(-parameters[1]))  # sigmoid to (0,1)
    beta = np.exp(parameters[2])  # exp to ensure positive
    
    n_trials = len(choices)
    n_options = 2
    Q = np.zeros(n_options)
    
    log_lik = 0.0
    for t in range(n_trials):
        # Softmax choice probability
        action = int(choices[t])
        arg_exp = beta * Q
        max_arg = np.max(arg_exp)
        exp_values = np.exp(arg_exp - max_arg)  # for numerical stability
        p = exp_values / np.sum(exp_values)
        log_lik += np.log(p[action] + 1e-10) # add small constant for numerical stability
        
        # Update Q-value with appropriate learning rate
        reward = rewards[t]
        delta = reward - Q[action]
        if delta >= 0:
            Q[action] = Q[action] + alpha_pos * delta
        else:
            Q[action] = Q[action] + alpha_neg * delta

    return log_lik


def RL2_model_trials(parameters, data):
    """
    Same model as RL2_model, but returns the per-trial log-likelihood
    array instead of the summed scalar (sum(...) == RL2_model(...)).
    """
    choices, rewards = data
    alpha_pos = 1 / (1 + np.exp(-parameters[0]))
    alpha_neg = 1 / (1 + np.exp(-parameters[1]))
    beta = np.exp(parameters[2])

    n_trials = len(choices)
    n_options = 2
    Q = np.zeros(n_options)

    log_liks = np.zeros(n_trials)
    for t in range(n_trials):
        action = int(choices[t])
        arg_exp = beta * Q
        max_arg = np.max(arg_exp)
        exp_values = np.exp(arg_exp - max_arg)
        p = exp_values / np.sum(exp_values)
        log_liks[t] = np.log(p[action] + 1e-10)

        reward = rewards[t]
        delta = reward - Q[action]
        if delta >= 0:
            Q[action] = Q[action] + alpha_pos * delta
        else:
            Q[action] = Q[action] + alpha_neg * delta

    return log_liks


# ============================================================================
# Generate synthetic data
# ============================================================================

def generate_data(n_trials, alpha_pos, alpha_neg, beta, reward_probs):
    """Generate data"""
    n_options = len(reward_probs)
    Q = np.zeros(n_options)
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    
    for t in range(n_trials):
        # Softmax choice
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
        action = np.random.choice(n_options, p=p)
        choices[t] = action
        
        # Sample reward
        reward = np.random.binomial(1, reward_probs[action])
        rewards[t] = reward
        
        # Update Q-value with appropriate learning rate
        prediction_error = reward - Q[action]
        if prediction_error > 0:
            Q[action] = Q[action] + alpha_pos * prediction_error
        else:
            Q[action] = Q[action] + alpha_neg * prediction_error
    
    return choices, rewards


np.random.seed(42)
n_trials = 100
reward_probs = [0.7, 0.3]  # Option 0 is better

all_data = []

print("Generating data...")
print(f"  30 subjects from RL2 (dual learning rates)")
for i in range(30):
    alpha_pos = 0.8 + np.random.rand() * 0.05
    alpha_neg = 0.4 + np.random.rand() * 0.05
    beta = 3.0 + np.random.rand() * 0.5
    
    choices, rewards = generate_data(n_trials, alpha_pos, alpha_neg, beta, reward_probs)
    all_data.append((choices, rewards))

# Generate 10 subjects from RL model
print(f"  10 subjects from RL (single learning rate)")
for i in range(10):
    alpha = 0.1 + np.random.rand() * 0.05
    beta = 1.0 + np.random.rand() * 0.5
    
    choices, rewards = generate_data(n_trials, alpha, alpha, beta, reward_probs)
    all_data.append((choices, rewards))

# ============================================================================
# Fit models
# ============================================================================

print("\n" + "+" * 70)
print("Running individual model fits")
print("+" * 70)

prior_mean_rl = np.array([0, 0])  # [alpha, beta] in unconstrained space
prior_variance_rl = 10
config = {"num_init": 1} # Use single initialization for speed
cbm_rl1 = individual_fit(all_data, RL_model, prior_mean_rl, prior_variance_rl, config=config,
                          model_trials=RL_model_trials)
with open(OUT_DIR / "cbm_rl1.pkl", "wb") as f:
    pickle.dump(cbm_rl1, f)

prior_mean_rl2 = np.array([0, 0, 0])  # [alpha_pos, alpha_neg, beta] in unconstrained space
prior_variance_rl2 = 10
config = {"num_init": 1} # Use single initialization for speed
cbm_rl2 = individual_fit(all_data, RL2_model, prior_mean_rl2, prior_variance_rl2, config=config,
                          model_trials=RL2_model_trials)
with open(OUT_DIR / "cbm_rl2.pkl", "wb") as f:
    pickle.dump(cbm_rl2, f)

# ============================================================================
# Bayesian Model Selection
# ============================================================================

print("\n" + "+" * 70)
print("Bayesian Model Selection based on individual fits")
print("+" * 70)

lme = np.column_stack([cbm_rl1.output.log_evidence, cbm_rl2.output.log_evidence])
bms_result = bms(lme)

print(f"\nExpected model frequencies:")
print(f"  RL (single alpha):     {bms_result.model_frequency[0]:.4f}")
print(f"  RL (dual alphas):     {bms_result.model_frequency[1]:.4f}")

print(f"\nExceedance probabilities:")
print(f"  RL (single alpha):     {bms_result.exceedance_prob[0]:.4f}")
print(f"  RL (dual alphas):     {bms_result.exceedance_prob[1]:.4f}")

print(f"\nProtected exceedance probabilities:")
print(f"  RL (single alpha):     {bms_result.protected_exceedance_prob[0]:.4f}")
print(f"  RL (dual alphas):     {bms_result.protected_exceedance_prob[1]:.4f}")

# ============================================================================
# HBI
# ============================================================================

print("\n" + "+" * 70)
print("Hierarchical Bayesian Inference")
print("+" * 70)

cbm_maps = [str(OUT_DIR / 'cbm_rl1.pkl'), str(OUT_DIR / 'cbm_rl2.pkl')]
config = {
    "save_prog": False,    
    'tolx': 0.05, # looser tolx for speed
}
models = [RL_model, RL2_model]

cbm = hbi_main(all_data, models, cbm_maps, fname=str(OUT_DIR / "hbi_rl.pkl"), config=config)

print("\n" + "+" * 70)
print("HBI Results")
print("+" * 70)

print(f"\nModel frequencies:")
for k, freq in enumerate(cbm.output.model_frequency):
    model_name = "RL" if k == 0 else "RL2"
    print(f"  {model_name}: {freq:.4f}")

print(f"\nExceedance probabilities:")
for k, xp in enumerate(cbm.output.exceedance_prob):
    model_name = "RL" if k == 0 else "RL2"
    print(f"  {model_name}: {xp:.8f}")

print(f"\nGroup-level parameter means:")
for k, mean in enumerate(cbm.output.group_mean):
    model_name = "RL" if k == 0 else "RL2"
    print(f"  {model_name}: {mean}")

print(f"\nGroup-level hierarchical error bars:")
for k, he in enumerate(cbm.output.group_hierarchical_errorbar):
    model_name = "RL" if k == 0 else "RL2"
    print(f"  {model_name}: {he}")

print("\n" + "+" * 70)
print("Run hbi_null for computing protected exceedance probabilities")
print("+" * 70)

cbm = hbi_null(all_data, str(OUT_DIR / 'hbi_rl.pkl'))

print(f"\nProtected exceedance probabilities:")
for k, pxp in enumerate(cbm.output.protected_exceedance_prob):
    model_name = "RL" if k == 0 else "RL2"
    print(f"  {model_name}: {pxp:.8f}")
