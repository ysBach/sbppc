"""Small bodies polarimetric phase curve (PPC) fitting tools.

This module provides various polarimetric phase curve models commonly used
for small solar system bodies (asteroids, comets, etc.), including:

- Linear-Exponential (LE)
- Lumme-Muinonen (LM)
- Shestopalov 3-parameter (SH3)
- Shestopalov 5-parameter (SH5)

All angles are in degrees and polarization values are in percent.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar


__all__ = [
    "ppc_pars", "ppc_p0s", "ppc_bounds",
    "ppc_le", "ppc_lm", "ppc_sh3", "ppc_sh5",
    "alpha_min_le",
    "log_likelihood_simple", "log_prior_uniform", "xy_minimum",
    "PPCModel"
]


class Param:
    """A model parameter with bounds and default value.

    Parameters
    ----------
    name : str
        The parameter name.
    low : float
        The lower bound.
    upp : float
        The upper bound.
    p0 : float
        The default (initial) value.
    """

    def __init__(self, name, low, upp, p0):
        self.name = str(name)
        self.low = low
        self.upp = upp
        self.p0 = p0

    def __str__(self):
        return "Parameter {} in [{}, {}]; default {}".format(
            self.name, self.low, self.upp, self.p0
        )


_pars_lm_b = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    c1=Param('c1', 1.e-6, 1.e+1, 1),
    c2=Param('c2', 1.e-6, 1.e+1, 1)
)

_pars_lm_f = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    c1=Param('c1', -1.e+1, 1.e+1, 1),
    c2=Param('c2', -1.e+1, 1.e+1, 1)
)

_pars_le = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    k=Param("k", 1.e-5, 100, 10.)
)

_pars_sh5 = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    k1=Param('k1', -1., 1., 0.1),
    k2=Param('k2', -10., 10., 1.e-5),
    k0=Param('k0', -10., 10., 1.e-5)
)

_pars_sh3 = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    k1=Param('k1', -1, 1., 0.001)
)

_pars_sgbip = dict(
    h=Param('h', 1.e-2, 1.e+1, 0.1),
    a0=Param('a0', 10, 30, 20),
    kn=Param('kn', -1., 1., 0.001),
    kp=Param('kp', 1.e-8, 1., 0.001)
)

_p0_lm_b = tuple([p.p0 for p in _pars_lm_b.values()])
_p0_lm_f = tuple([p.p0 for p in _pars_lm_f.values()])
_p0_le = tuple([p.p0 for p in _pars_le.values()])
_p0_sh5 = tuple([p.p0 for p in _pars_sh5.values()])
_p0_sh3 = tuple([p.p0 for p in _pars_sh3.values()])
_p0_sgbip = tuple([p.p0 for p in _pars_sgbip.values()])

_bounds_lm_b = (tuple([(p.low, p.upp) for p in _pars_lm_b.values()]))
_bounds_lm_f = (tuple([(p.low, p.upp) for p in _pars_lm_f.values()]))
_bounds_le = (tuple([(p.low, p.upp) for p in _pars_le.values()]))
_bounds_sh5 = (tuple([(p.low, p.upp) for p in _pars_sh5.values()]))
_bounds_sh3 = (tuple([(p.low, p.upp) for p in _pars_sh3.values()]))

ppc_pars = dict(lm_b=_pars_lm_b, lm_f=_pars_lm_f, le=_pars_le,
                sh5=_pars_sh5, sh3=_pars_sh3)
ppc_p0s = dict(lm_b=_p0_lm_b, lm_f=_p0_lm_f, le=_p0_le,
               sh5=_p0_sh5, sh3=_p0_sh3, sgbip=_p0_sgbip)
ppc_bounds = dict(lm_b=_bounds_lm_b, lm_f=_bounds_lm_f, le=_bounds_le,
                  sh5=_bounds_sh5, sh3=_bounds_sh3)





def ppc_le(x, h=_pars_le["h"].p0, a0=_pars_le["a0"].p0, k=_pars_le["k"].p0):
    """ The 3-parameter Linear-Exponential function

    Parameters
    ----------
    x : array-like
        The phase angle in degrees.

    h, a0 : float
        The slope parameter and the inversion angle in the units of %/deg
        and deg.

    k : float
        The exponential coefficient in the units of deg.

    Returns
    -------
    Pr : array-like
        The calculated polarization value in per-cent
    """
    e0 = np.exp(-a0/k)
    # numer = a0*(np.exp(-a*x) + a*np.exp(-a*a0)*x - 1)
    numer = (1 - e0)*x - (1 - np.exp(-x/k))*a0
    denom = 1 - (1 + a0/k)*e0
    return h*(numer/denom)


def alpha_min_le(h=None, a0=_pars_le["a0"].p0, k=_pars_le["k"].p0):
    """The minimum phase angle for the 3-parameter Linear-Exponential.

    Parameters
    ----------
    h : None
        The slope parameter in the units of %/deg. It has no effect on the
        minimum phase angle, but added for consistency with other functions.
        (so that the user can call it by ``alpha_min_le(*theta)``)

    a0, k : float
        The inversion angle and the exponential coefficient in the units of
        deg and deg, respectively.

    Returns
    -------
    alpha_min : float
        The phase angle at which the minimum polarization occurs (degrees).
    """
    return -k*np.log(k/a0*(1 - np.exp(-a0/k)))


def ppc_lm(x, h=_pars_lm_b["h"].p0, a0=_pars_lm_b["a0"].p0,
           c1=_pars_lm_b["c1"].p0, c2=_pars_lm_b["c2"].p0):
    """ The 4-parameter Lumme-Muinonen function

    Parameters
    ----------
    x : array-like
        The phase angle in degrees.

    h, a0 : float
        The slope parameter and the inversion angle in the units of %/deg
        and deg.

    c1, c2: float
        The dimensionless powers for the sine and cosine terms.

    Returns
    -------
    Pr : array-like
        The calculated polarization value in per-cent
    """
    sx = np.sin(np.deg2rad(x))
    sa0 = np.sin(np.deg2rad(a0))
    cx2 = np.cos(np.deg2rad(x/2))
    ca02 = np.cos(np.deg2rad(a0/2))
    sxma0 = np.sin(np.deg2rad(x - a0))

    term1 = (sx / sa0)**c1
    term2 = (cx2 / ca02)**c2
    return h * term1 * term2 * sxma0


def ppc_sh5(x, h=_pars_sh5["h"].p0, a0=_pars_sh5["a0"].p0,
            k1=_pars_sh5["k1"].p0, k2=_pars_sh5["k2"].p0, k0=_pars_sh5["k0"].p0):
    """ The 5-parameter Shestopalov function

    Parameters
    ----------
    x : array-like
        The phase angle in degrees.

    h, a0 : float
        The slope parameter and the inversion angle in the units of %/deg
        and deg.

    k1, k2, k0 : float
        The exponential coefficients in the units of 1/deg.

    Returns
    -------
    Pr : array-like
        The calculated polarization value in per-cent
    """
    term1 = (1 - np.exp(-k1*x)) / (1 - np.exp(-k1*a0))
    term2 = (1 - np.exp(-k0*(x - a0))) / k0
    term3 = (1 - np.exp(-k2*(x - 180))) / (1 - np.exp(-k2*(a0 - 180)))
    return h * term1 * term2 * term3


def ppc_sh3(x, h=_pars_sh3["h"].p0, a0=_pars_sh3["a0"].p0, k1=_pars_sh3["k1"].p0):
    """The 3-parameter Shestopalov function.

    Parameters
    ----------
    x : array-like
        The phase angle in degrees.

    h, a0 : float
        The slope parameter and the inversion angle in the units of %/deg
        and deg.

    k1 : float
        The first exponential coefficient in the units of 1/deg.

    Returns
    -------
    Pr : array-like
        The calculated polarization value in per-cent.
    """
    term1 = (1 - np.exp(-k1*x)) / (1 - np.exp(-k1*a0))
    term2 = (x - 180)/(a0-180)
    return h*(x-a0)*term1*term2


_ppcs = {"le": ppc_le, "sh3": ppc_sh3, "sh5": ppc_sh5, "lm_b": ppc_lm, "lm_f": ppc_lm}


def log_likelihood_simple(fun, x, y, yerr, theta):
    """Compute log-likelihood assuming Gaussian errors.

    Parameters
    ----------
    fun : callable
        The model function.
    x : array-like
        Independent variable (phase angles).
    y : array-like
        Observed values (polarization).
    yerr : array-like or None
        Uncertainties on y. If None, assumes unit variance.
    theta : tuple
        Model parameters.

    Returns
    -------
    ll : float
        The log-likelihood value.
    """
    model = fun(x, *theta)
    sigsq = 1 if yerr is None else yerr**2
    return -0.5*(np.sum((y-model)**2/sigsq))


# def log_likelihood_const(fun, x, y, yerr, theta):
#     model = fun(x, *theta)
#     sigsq = yerr**2
#     return -0.5*(np.sum((y-model)**2/sigsq) + np.sum(np.log(2*np.pi*sigsq)))


# def log_likelihood_const_sigint(fun, x, y, yerr, theta, sig_int):
#     model = fun(x, *theta)
#     sigsq = yerr**2 + sig_int**2
#     return -0.5*(np.sum((y-model)**2/sigsq) + np.sum(2*np.pi*np.log(sigsq)))


# def log_likelihood_sigint(fun, x, y, yerr, theta, sig_int):
#     model = fun(x, *theta)
#     sigsq = yerr**2 + sig_int**2
#     return -0.5*(np.sum((y-model)**2/sigsq))


# def log_likelihood(fun, x, y, yerr, theta, sig_int=None, include_const_term=False):
#     model = fun(x, *theta)
#     if sig_int is None:
#         sigsq = yerr**2
#         if include_const_term:
#             return -0.5*(np.sum((y-model)**2/sigsq) + np.sum(np.log(2*np.pi*sigsq)))
#         else:
#             return -0.5*(np.sum((y-model)**2/sigsq))
#     else:
#         sigsq = yerr**2 + sig_int**2
#         if include_const_term:
#             return -0.5*(np.sum((y-model)**2/sigsq) + np.sum(2*np.pi*np.log(sigsq)))
#         else:
#             return -0.5*(np.sum((y-model)**2/sigsq) + np.sum(np.log(sigsq)))


def log_prior_uniform(theta, bounds):
    """Uniform prior: 0 if theta within bounds, -inf otherwise.

    Parameters
    ----------
    theta : array-like
        Model parameters.
    bounds : tuple of tuples
        (lower_bounds, upper_bounds) for each parameter.

    Returns
    -------
    lp : float
        Log-prior probability (0.0 or -inf).
    """
    if not (np.all(bounds[0] < theta) and np.all(theta < bounds[1])):
        return -np.inf
    else:
        return 0.0


def log_probability(x, y, yerr, theta, log_prior_fn, log_likelihood_fn):
    """Compute log-posterior probability.

    Parameters
    ----------
    x, y, yerr : array-like
        Data arrays (phase angles, polarization, uncertainties).
    theta : array-like
        Model parameters.
    log_prior_fn : callable
        Function returning log-prior given theta.
    log_likelihood_fn : callable
        Function returning log-likelihood given (theta, x, y, yerr).

    Returns
    -------
    lp : float
        Log-posterior probability.
    """
    lp = log_prior_fn(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_fn(theta, x, y, yerr)


def xy_minimum(fun, theta, xmin_fn=None, **kwargs):
    """Find the x-value and function value at the minimum.

    Parameters
    ----------
    fun : callable
        Function to minimize, called as fun(x, *theta).
    theta : tuple
        Additional parameters passed to fun.
    xmin_fn : callable or None
        If provided, use this to compute xmin analytically.
    **kwargs
        Passed to scipy.optimize.minimize_scalar.

    Returns
    -------
    xmin : float
        The x-value at the minimum.
    fmin : float
        The function value at the minimum.
    """
    if xmin_fn is None:
        mini = minimize_scalar(fun, args=tuple(np.atleast_1d(theta)), **kwargs)
        # Using bracket is a bit faster than using bounds.
        return mini.x, mini.fun
    else:
        xmin = xmin_fn(*theta)
        return xmin, fun(xmin, *theta)


class PPCModel:
    def __init__(self, fun, p0="default", log_likelihood_fn=None,
                 log_prior_fn=None, log_prob_fn=None,
                 bounds="default", amin_pmin_fn=None, amax_pmax_fn=None) -> None:
        """Initialize a polarimetric phase curve model.

        Parameters
        ----------
        fun : {"le", "sh3", "sh5", "lm_b", "lm_f"} or callable
            The model name or function. If str, must be one of
            {"le", "sh3", "sh5", "lm_b", "lm_f"} for linear-exponential,
            Shestopalov 3-parameter, Shestopalov 5-parameter, Lumme-Muinonen
            3-parameter trigonometric function (power parameters bounded to
            positive), and the same with free powers models, respectively. For
            these strings, the model functions (`log_likelihood_fn`,
            `log_prior_fn`, `log_prob_fn`, `amin_pmin_fn`, `amax_pmax_fn`) are
            set automatically.
            If given as a callable, it MUST have the following arguments: The
            first argument is the phase angle in degrees, and all other
            following arguments are the model parameters. I cannot guarantee
            any function with other kwargs to work properly. The model function
            MUST return the polarization value in per-cent.

        log_likelihood_fn : callable, optional
            Custom log-likelihood function with signature (theta, x, y, yerr).

        log_prior_fn : callable, optional
            Custom log-prior function with signature (theta,).

        log_prob_fn : callable, optional
            Custom log-posterior function with signature (theta, x, y, yerr).

        amin_pmin_fn : callable, optional
            Function to compute (alpha_min, P_min) given theta.

        amax_pmax_fn : callable, optional
            Function to compute (alpha_max, P_max) given theta.
        """
        if isinstance(fun, str):
            self.fun_name = fun
            self.fun = _ppcs[fun]
            if isinstance(bounds, str):
                if bounds == "default":
                    self.bounds = ppc_bounds[fun]
                else:
                    raise ValueError("`bounds` string is not understood.")
            else:
                self.bounds = None if bounds is None else tuple(bounds)

            if isinstance(p0, str):
                if p0 == "default":
                    self.p0 = ppc_p0s[fun]
                else:
                    raise ValueError("`p0` string is not understood.")
            else:
                self.p0 = None if p0 is None else tuple(p0)

        else:  # `model` is a functional object
            self.fun = fun
            self.fun_name = getattr(fun, "__name__", "custom").replace("ppc_", "")
            self.bounds = bounds
            self.p0 = p0

        self.npars = self.fun.__code__.co_argcount - 1

        # Set parameter names
        if isinstance(fun, str) and fun in ppc_pars:
            self.par_names = list(ppc_pars[fun].keys())
        else:
            # For custom functions, extract from signature
            import inspect
            sig = inspect.signature(self.fun)
            self.par_names = list(sig.parameters.keys())[1:]  # Skip 'x'

        if amin_pmin_fn is None:
            self.amin_pmin_fn = self._default_amin_pmin
        else:
            self.amin_pmin_fn = amin_pmin_fn

        if amax_pmax_fn is None:
            self.amax_pmax_fn = self._default_amax_pmax
        else:
            self.amax_pmax_fn = amax_pmax_fn

        # Log-likelihood setup
        if log_likelihood_fn is None:
            self.log_likelihood_fn = self._default_log_likelihood
        else:
            self.log_likelihood_fn = log_likelihood_fn

        # Log-prior setup
        if log_prior_fn is None:
            if self.bounds is None:
                self.bounds_lohi = None
                self.log_prior_fn = self._prior_flat
            else:
                self.bounds_lohi = (tuple([b[0] for b in self.bounds]),
                                    tuple([b[1] for b in self.bounds]))
                # ^^^ same as bounds but for easy use for some APIs...
                self.log_prior_fn = self._default_log_prior
        else:
            self.log_prior_fn = log_prior_fn

        # Log-probability setup
        if log_prob_fn is None:
            self.log_prob_fn = self._default_log_prob
        else:
            self.log_prob_fn = log_prob_fn

        self.nll = self._nll_calc

    def _prior_flat(self, theta):
        """Flat prior (always 0.0) when no bounds are given."""
        return 0.0

    def _nll_calc(self, *args):
        """Negative log-likelihood function (wrapper for minimization)."""
        return -self.log_likelihood_fn(*args)

    def _default_amin_pmin(self, theta):
        """Default calculation of minimum polarization."""
        xmin_fn = None
        if getattr(self.fun, "__name__", "") == "ppc_le":
             xmin_fn = alpha_min_le

        return xy_minimum(
            self.fun, theta, bounds=(0, 180), method="bounded", tol=0.001,
            xmin_fn=xmin_fn
        )

    def _default_amax_pmax(self, theta):
        """Default calculation of maximum polarization."""
        xmin_fn = None
        if getattr(self.fun, "__name__", "") == "ppc_le":
             xmin_fn = lambda *args: 180

        return xy_minimum(
            self._neg_fun_eval, theta, bounds=(60, 180), method="bounded", tol=0.001,
            xmin_fn=xmin_fn
        ) * np.array([1, -1])

    def _neg_fun_eval(self, *args):
        """Negative of model function for maximization."""
        return -self.fun(*args)

    def _default_log_likelihood(self, theta, x, y, yerr):
        """Default log-likelihood using the model function."""
        return log_likelihood_simple(self.fun, x, y, yerr, theta)

    def _default_log_prior(self, theta):
        """Default uniform prior based on bounds."""
        return log_prior_uniform(theta, self.bounds_lohi)

    def _default_log_prob(self, theta, x, y, yerr):
        """Default log-probability function."""
        return log_probability(
            x, y, yerr, theta, self.log_prior_fn, self.log_likelihood_fn)

    def solve_lsq(self, x, y, yerr=None, p0="default", bounds="default", calc_stats=True, **kwargs):
        """Fit the model to data using least-squares minimization.

        Parameters
        ----------
        x : array-like
            Phase angles (degrees).
        y : array-like
            Observed polarization (percent).
        yerr : array-like, optional
            Uncertainties on y.
        p0 : str or tuple, optional
            Initial parameters. Use "default" for instance defaults.
        bounds : str or tuple, optional
            Parameter bounds. Use "default" for instance defaults.
        calc_stats : bool, optional
            Whether to calculate statistics (chi2, chi2_red,BIC, AIC). Default is True.
            (chi2 = sum((y_obs - y_model)^2 / y_err^2),
            chi2_red = chi2 / (n_data - n_pars),
            BIC = chi2 + n_pars * log(n_data),
            AIC = chi2 + 2 * n_pars).
        **kwargs
            Passed to scipy.optimize.minimize.

        Returns
        -------
        theta_lsq : ndarray
            Best-fit parameters.

        Notes
        -----
        Also sets `self.theta_lsq`, `self.amin_lsq`, `self.pmin_lsq`,
        `self.amax_lsq`, `self.pmax_lsq`. If `calc_stats` is True, also sets
        `self.chi2_lsq`, `self.chi2_red_lsq`, `self.bic_lsq`, `self.aic_lsq`.
        """
        if p0 == "default":
            p0 = self.p0
        if bounds == "default":
            bounds = self.bounds
        # if yerr is None:
        #     yerr = np.zeros_like(y)
        self.lsq = minimize(self.nll, p0, bounds=bounds, args=(x, y, yerr), **kwargs)
        self.theta_lsq = self.lsq.x
        self.amin_lsq, self.pmin_lsq = self.amin_pmin_fn(self.theta_lsq)
        self.amax_lsq, self.pmax_lsq = self.amax_pmax_fn(self.theta_lsq)

        if calc_stats:
            # Calculate statistics
            # We need residuals.
            model_y = self.fun(x, *self.theta_lsq)
            resid = y - model_y
            if yerr is None:
                # Assume unit weight? Or estimate from residuals?
                # Standard practice with None is usually sigma=1.
                self.chi2_lsq = np.sum(resid**2)
            else:
                self.chi2_lsq = np.sum((resid / yerr)**2)

            n_data = len(x)
            self.chi2_red_lsq = self.chi2_lsq / (n_data - self.npars)
            self.bic_lsq = self.chi2_lsq + self.npars * np.log(n_data)
            self.aic_lsq = self.chi2_lsq + 2 * self.npars

        return self.theta_lsq

    def fun_lsq(self, xvals):
        """Evaluate the fitted model at given phase angles.

        Parameters
        ----------
        xvals : array-like
            Phase angles (degrees).

        Returns
        -------
        Pr : array-like
            Predicted polarization (percent) using theta_lsq.
        """
        return self.fun(xvals, *self.theta_lsq)
