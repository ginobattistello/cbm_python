"""Bayesian model selection examples.

This script demonstrates:

1. standard group BMS;
2. between-condition BMS;
3. between-group BMS.

Two candidate binary Rescorla-Wagner models are used:

M1
    alpha and beta are both estimated.

M2
    alpha is fixed at 0.5 by setting its prior variance to zero;
    beta remains estimated.

All BMS functions use their integrated CBM-style ``verbose=True`` output.
"""

import numpy as np

from cbm.bms_group import (
    bms_group,
    bms_group_btw_conds,
    bms_group_btw_groups,
)
from cbm.individual_fit import individual_fit
from cbm.optimization import Config

from models import binary_model
from simulate import binary_subject


# =====================================================================
# Shared fitting configuration
# =====================================================================

config = Config(
    d=2,
    range_bounds=np.array([
        [0.02, 0.10],
        [0.98, 8.00],
    ]),
    hard_bounds=np.array([
        [0.001, 0.01],
        [0.999, 20.0],
    ]),
    num_init=5,
    verbose=False,
    display=False,
)


# =====================================================================
# 1. Standard group BMS
# =====================================================================

rng = np.random.default_rng(10)

data = [
    binary_subject(
        rng,
        theta=(0.25, 3.0),
    )
    for _ in range(20)
]

fit_free = individual_fit(
    data=data,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

fit_fixed = individual_fit(
    data=data,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([0.0, 16.0]),
    config=config,
)

L = np.column_stack([
    fit_free.output.log_evidence,
    fit_fixed.output.log_evidence,
])

result = bms_group(
    L,
    n_samples=100_000,
    verbose=True,
)


# =====================================================================
# 2. Between-condition BMS
# =====================================================================
#
# Same subjects, two repeated conditions.
#
# Condition 1: alpha = 0.50
# Condition 2: alpha = 0.20
#
# Input shape:
#     subjects x models x conditions
#

rng = np.random.default_rng(20)
n_subjects = 16

data_condition_1 = [
    binary_subject(
        rng,
        theta=(0.50, 3.0),
    )
    for _ in range(n_subjects)
]

fit_free_condition_1 = individual_fit(
    data=data_condition_1,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

fit_fixed_condition_1 = individual_fit(
    data=data_condition_1,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([0.0, 16.0]),
    config=config,
)

L_condition_1 = np.column_stack([
    fit_free_condition_1.output.log_evidence,
    fit_fixed_condition_1.output.log_evidence,
])


data_condition_2 = [
    binary_subject(
        rng,
        theta=(0.20, 3.0),
    )
    for _ in range(n_subjects)
]

fit_free_condition_2 = individual_fit(
    data=data_condition_2,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

fit_fixed_condition_2 = individual_fit(
    data=data_condition_2,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([0.0, 16.0]),
    config=config,
)

L_condition_2 = np.column_stack([
    fit_free_condition_2.output.log_evidence,
    fit_fixed_condition_2.output.log_evidence,
])

L_conditions = np.stack(
    [
        L_condition_1,
        L_condition_2,
    ],
    axis=2,
)

result_conditions = bms_group_btw_conds(
    L_conditions,
    n_samples=100_000,
    verbose=True,
)


# =====================================================================
# 3. Between-group BMS
# =====================================================================
#
# Different subjects in two independent groups.
#
# Group 1: alpha = 0.50
# Group 2: alpha = 0.20
#
# Input:
#     [subjects_group1 x models,
#      subjects_group2 x models]
#

rng = np.random.default_rng(30)

data_group_1 = [
    binary_subject(
        rng,
        theta=(0.50, 3.0),
    )
    for _ in range(16)
]

fit_free_group_1 = individual_fit(
    data=data_group_1,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

fit_fixed_group_1 = individual_fit(
    data=data_group_1,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([0.0, 16.0]),
    config=config,
)

L_group_1 = np.column_stack([
    fit_free_group_1.output.log_evidence,
    fit_fixed_group_1.output.log_evidence,
])


data_group_2 = [
    binary_subject(
        rng,
        theta=(0.20, 3.0),
    )
    for _ in range(16)
]

fit_free_group_2 = individual_fit(
    data=data_group_2,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([1.0, 16.0]),
    config=config,
)

fit_fixed_group_2 = individual_fit(
    data=data_group_2,
    model=binary_model,
    prior_mean=np.array([0.5, 2.0]),
    prior_variance=np.array([0.0, 16.0]),
    config=config,
)

L_group_2 = np.column_stack([
    fit_free_group_2.output.log_evidence,
    fit_fixed_group_2.output.log_evidence,
])

result_groups = bms_group_btw_groups(
    [
        L_group_1,
        L_group_2,
    ],
    n_samples=100_000,
    verbose=True,
)
