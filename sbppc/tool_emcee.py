"""Visualization and calculation utilities for emcee MCMC sampling results."""

import multiprocessing
from multiprocessing import Pool

import numpy as np
import emcee
import corner

__all__ = ["trace_plot", "run_emcee", "calc_derived_samples", "corner_plot", "summarize_samples"]


def run_emcee(model, x, y, yerr=None, p0=None, nwalkers=32, nsteps=2000,
              burnin=500, parallel=True, progress=True, **kwargs):
    """Run MCMC sampling using emcee.

    Parameters
    ----------
    model : PPCModel
        The initialized PPCModel instance.
    x : array-like
        Phase angles.
    y : array-like
        Observed polarization.
    yerr : array-like, optional
        Uncertainties. If None, assumes unit variance.
    p0 : array-like, optional
        Initial guess for parameters. If None, it first runs `solve_lsq`
        to find a starting point.
    nwalkers : int, optional
        Number of walkers. Default 32.
    nsteps : int, optional
        Number of steps per walker. Default 2000.
    burnin : int, optional
        Number of burn-in steps to discard. Default 500.
    parallel : bool, optional
        If True, use multiprocessing.Pool for parallelization.
    progress : bool, optional
        If True, show progress bar.
    **kwargs
        Passed to `emcee.EnsembleSampler`.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        The sampler object containing the chains.
    """
    # 1. Setup initial position
    if p0 is None:
        if not hasattr(model, 'theta_lsq'):
            model.solve_lsq(x, y, yerr=yerr)
        p0_center = model.theta_lsq
    else:
        p0_center = np.atleast_1d(p0)

    # Initialize walkers in a tiny ball around the best fit
    ndim = len(p0_center)
    pos = p0_center + 1e-4 * np.random.randn(nwalkers, ndim)

    # 2. Run MCMC
    def _run_sampler(pool_obj=None):
        sampler_obj = emcee.EnsembleSampler(
            nwalkers, ndim, model.log_prob_fn, args=(x, y, yerr), pool=pool_obj,
            **kwargs
        )

        # Burn-in
        if burnin > 0:
            p_curr, _, _ = sampler_obj.run_mcmc(pos, burnin, progress=progress)
            # sampler_obj.reset()  <-- removed to allow get_chain(discard=...)
        else:
            p_curr = pos

        # Production run
        sampler_obj.run_mcmc(p_curr, nsteps, progress=progress)
        return sampler_obj

    if parallel:
        # Use context manager to ensure pool is closed
        with Pool() as pool:
            sampler = _run_sampler(pool)
    else:
        sampler = _run_sampler(None)

    return sampler



try:
    from sbppc.numba_utils import calculate_derived_numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def calc_derived_samples(model, samples):
    """Calculate derived parameters (Pmin, αmin, Pmax, αmax) from samples.

    Parameters
    ----------
    model : PPCModel
        The model instance containing `amin_pmin_fn` and `amax_pmax_fn`.
    samples : ndarray
        Flattened MCMC samples of shape (nsamples, npars).

    Returns
    -------
    derived : dict of ndarray
        Dictionary containing samples for:
        - 'amin': Phase angle of minimum polarization
        - 'pmin': Minimum polarization value
        - 'amax': Phase angle of maximum polarization
        - 'pmax': Maximum polarization value
    """
    # Numba optimization path
    func_id = -1
    if HAS_NUMBA:
        name = getattr(model, "fun_name", "")
        if name == "le":
            func_id = 0
        elif name == "sh3":
            func_id = 1
        elif name in ["lm_b", "lm_f"]:
            func_id = 2
        elif name == "sh5":
            func_id = 3

    if func_id != -1:
        # Use Numba-optimized routine
        # Bounds logic: Default for now (0, 60) and (60, 180)
        # We could extract from model if needed, but min/max usually fall here.
        amin, pmin, amax, pmax = calculate_derived_numba(samples, func_id)

        return {
            "amin": amin,
            "pmin": pmin,
            "amax": amax,
            "pmax": pmax,
        }

    # Fallback to Python loop
    nsamples = len(samples)
    amin = np.zeros(nsamples)
    pmin = np.zeros(nsamples)
    amax = np.zeros(nsamples)
    pmax = np.zeros(nsamples)

    for i in range(nsamples):
        theta = samples[i]
        # model.amin_pmin_fn returns (amin, pmin)
        amin, pmin = model.amin_pmin_fn(theta)
        # model.amax_pmax_fn returns (amax, pmax)
        amax, pmax = model.amax_pmax_fn(theta)

        amin[i] = amin
        pmin[i] = pmin
        amax[i] = amax
        pmax[i] = pmax

    return {
        "amin": amin,
        "pmin": pmin,
        "amax": amax,
        "pmax": pmax,
    }


def summarize_samples(model, flat_samples, percentiles=(16, 50, 84),
                      include_max=False, dropna=True):
    """Calculate derived parameters, stack with samples, and compute percentiles.

    Parameters
    ----------
    model : PPCModel
        The model instance used for MCMC.
    flat_samples : ndarray
        Flattened MCMC samples of shape (nsamples, npars).
    percentiles : tuple of float, optional
        Percentiles to compute. Default (16, 50, 84).
    include_max : bool, optional
        If True, include amax and pmax in output. Default False.
    dropna : bool, optional
        If True, drop samples where any derived parameter is NaN. Default True.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'samples': ndarray of stacked samples (nsamples, npars + n_derived)
        - 'labels': list of parameter labels
        - 'percentiles': dict mapping label -> array of percentile values
        - 'lsq': dict of LSQ values if available (or None)
    """
    # Calculate derived parameters
    derived = calc_derived_samples(model, flat_samples)

    # Build stacked samples and labels
    samples_list = [flat_samples, derived["amin"][:, None], derived["pmin"][:, None]]
    labels = list(getattr(model, 'par_names', [f'p{i}' for i in range(flat_samples.shape[1])]))
    labels.extend(["amin", "pmin"])

    if include_max:
        samples_list.extend([derived["amax"][:, None], derived["pmax"][:, None]])
        labels.extend(["amax", "pmax"])

    samples_all = np.hstack(samples_list)

    # Drop NaN rows if requested
    if dropna:
        mask = ~np.isnan(samples_all).any(axis=1)
        samples_all = samples_all[mask]

    # Calculate percentiles
    pct_values = np.percentile(samples_all, percentiles, axis=0)
    pct_dict = {}
    for i, label in enumerate(labels):
        pct_dict[label] = {p: pct_values[j, i] for j, p in enumerate(percentiles)}

    # Get LSQ values if available
    # lsq = None
    # if hasattr(model, "theta_lsq"):
    #     lsq = {
    #         label: val for label, val in
    #         zip(labels[:flat_samples.shape[1]], model.theta_lsq)
    #     }
    #     if hasattr(model, "amin_lsq"):
    #         lsq["amin"] = model.amin_lsq
    #     if hasattr(model, "pmin_lsq"):
    #         lsq["pmin"] = model.pmin_lsq
    #     if include_max:
    #         if hasattr(model, "amax_lsq"):
    #             lsq["amax"] = model.amax_lsq
    #         if hasattr(model, "pmax_lsq"):
    #             lsq["pmax"] = model.pmax_lsq

    return {
        "samples": samples_all,
        "labels": labels,
        "percentiles": pct_dict,
        # "lsq": lsq,
    }


def corner_plot(samples, labels=None, truths=None, **kwargs):
    """Wrapper for corner.corner plot.

    Parameters
    ----------
    samples : ndarray
        Flattened samples (nsamples, npars).
    labels : list of str, optional
        Parameter labels.
    truths : list of float, optional
        True values to mark.
    **kwargs
        Passed to corner.corner().
    """
    return corner.corner(samples, labels=labels, truths=truths, **kwargs)


def trace_plot(axes, samples, labels=None, alpha=0.3, **kwargs):
    """Plot MCMC trace (chain) for each parameter.

    Parameters
    ----------
    axes : array-like of Axes
        Matplotlib axes to plot on. Must have one axis per parameter.
    samples : ndarray
        MCMC samples with shape (nsamples, npars) for flattened chains,
        or (nsteps, nwalkers, npars) for full chains.
    labels : list of str, optional
        Parameter labels for y-axis.
    alpha : float, optional
        Line transparency, default 0.3.
    **kwargs
        Passed to ax.plot().

    Raises
    ------
    ValueError
        If samples.ndim is not 2 or 3, or if axes/labels size mismatch.
    """
    axes = np.atleast_1d(axes)
    if samples.ndim == 2:  # get_chain(flat=True) case
        nsamples, npars = samples.shape
    elif samples.ndim == 3:  # get_chain() case
        nsamples, nchain, npars = samples.shape
    else:
        raise ValueError(f"{samples.ndim = } is not 2 or 3")

    if axes.size != npars:
        raise ValueError(f"{axes.size = } is different from {samples.shape[-1] = }")

    if labels is not None and len(labels) != npars:
        raise ValueError(f"{len(labels) = } is different from {samples.shape[-1] = }")

    for i, ax in enumerate(axes):
        if samples.ndim == 3:
            ax.plot(samples[:, :, i], "k", alpha=alpha, **kwargs)
        else:
            ax.plot(samples[:, i], "k", alpha=alpha, **kwargs)
        if labels is not None:
            ax.set_ylabel(labels[i])
        ax.set_xlim(0, nsamples)
