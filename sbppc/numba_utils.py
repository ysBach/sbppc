"""Numba-optimized versions of PPC models and minimization routines.

This module dynamically compiles JIT versions of models defined in `sbppc.functions`
to ensure consistency and avoid code duplication.
"""

import inspect
import numpy as np
from numba import njit
from sbppc import functions

# Define JIT-compiled helpers that override the ones in functions.py


def _get_jitted_model(func_name):
    """Dynamically get source from functions.py and JIT compile it."""
    func = getattr(functions, func_name)
    src = inspect.getsource(func)

    # Create a namespace matching sbppc.functions but with JIT-ed helpers
    ns = vars(functions).copy()

    ns["np"] = np # Ensure numpy is available
    ns["njit"] = njit

    # Clean up decorators if any? (PPC models usually don't have decorators)
    # Execute definition
    exec(src, ns)

    # Retrieve and JIT
    py_func = ns[func_name]
    return njit(py_func)

# Generate JIT-compiled models
ppc_le_jit = _get_jitted_model("ppc_le")
ppc_sh3_jit = _get_jitted_model("ppc_sh3")
ppc_sh5_jit = _get_jitted_model("ppc_sh5")
ppc_lm_jit = _get_jitted_model("ppc_lm")

# Map names to functions for dispatch
# ID mapping: 0=le, 1=sh3, 2=lm, 3=sh5
_FUNC_MAP = {
    0: ppc_le_jit,
    1: ppc_sh3_jit,
    2: ppc_lm_jit,
    3: ppc_sh5_jit
}

@njit
def _eval_func(func_id, x, theta):
    # Dispatcher manually expanded because Numba doesn't support dict[int, func] dispatch easily
    # in nopython mode for typical closures without overhead.
    if func_id == 0: # LE
        return ppc_le_jit(x, theta[0], theta[1], theta[2])
    elif func_id == 1: # SH3
        return ppc_sh3_jit(x, theta[0], theta[1], theta[2])
    elif func_id == 2: # LM (b or f)
        return ppc_lm_jit(x, theta[0], theta[1], theta[2], theta[3])
    elif func_id == 3: # SH5
        return ppc_sh5_jit(x, theta[0], theta[1], theta[2], theta[3], theta[4])
    return 0.0

@njit
def minimize_golden(func_id, theta, ax, bx, tol=1e-5, maxiter=50):
    """Golden Section Search for minimum."""
    R = 0.61803399
    C = 1.0 - R

    x0 = ax
    x3 = bx
    x1 = x0 + C * (x3 - x0)
    x2 = x0 + R * (x3 - x0)

    f1 = _eval_func(func_id, x1, theta)
    f2 = _eval_func(func_id, x2, theta)

    for _ in range(maxiter):
        if abs(x3 - x0) < tol * (abs(x1) + abs(x2) + 1e-9):
            break

        if f2 < f1:
            x0 = x1
            x1 = x2
            x2 = R * x1 + C * x3
            f1 = f2
            f2 = _eval_func(func_id, x2, theta)
        else:
            x3 = x2
            x2 = x1
            x1 = R * x2 + C * x0
            f2 = f1
            f1 = _eval_func(func_id, x1, theta)

    if f1 < f2:
        return x1, f1
    else:
        return x2, f2

@njit
def minimize_golden_max(func_id, theta, ax, bx, tol=1e-5, maxiter=50):
    """Golden Section Search for MAXIMUM (minimize negative)."""
    R = 0.61803399
    C = 1.0 - R
    x0 = ax
    x3 = bx
    x1 = x0 + C * (x3 - x0)
    x2 = x0 + R * (x3 - x0)

    # Eval NEGATIVE func
    f1 = -_eval_func(func_id, x1, theta)
    f2 = -_eval_func(func_id, x2, theta)

    for _ in range(maxiter):
        if abs(x3 - x0) < tol * (abs(x1) + abs(x2) + 1e-9):
            break
        if f2 < f1:
            x0 = x1
            x1 = x2
            x2 = R * x1 + C * x3
            f1 = f2
            f2 = -_eval_func(func_id, x2, theta)
        else:
            x3 = x2
            x2 = x1
            x1 = R * x2 + C * x0
            f2 = f1
            f1 = -_eval_func(func_id, x1, theta)

    if f1 < f2:
        return x1, f1 # Result of min(-f) -> f max
    else:
        return x2, f2

@njit
def calculate_derived_numba(samples, func_id):
    """Calculate derived parameters for many samples using Numba loop.

    samples: (N, npars) array
    """
    N = len(samples)
    alpha_min = np.empty(N)
    p_min = np.empty(N)
    alpha_max = np.empty(N)
    p_max = np.empty(N)

    for i in range(N):
        theta = samples[i]

        # Calculate min (assumed in 0-60)
        amin, pmin = minimize_golden(func_id, theta, 0.0, 60.0)
        alpha_min[i] = amin
        p_min[i] = pmin

        # Calculate max (assumed in 60-180)
        amax, neg_pmax = minimize_golden_max(func_id, theta, 60.0, 180.0)
        alpha_max[i] = amax
        p_max[i] = -neg_pmax # Negate back

    return alpha_min, p_min, alpha_max, p_max
