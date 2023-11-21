"""
All angles must be in degrees unit
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
               sh5=_p0_sh5, appsp=_p0_sh3, sgbip=_p0_sgbip)
ppc_bounds = dict(lm_b=_bounds_lm_b, lm_f=_bounds_lm_f, le=_bounds_le,
                  sh5=_bounds_sh5, sh3=_bounds_sh3)


def cos(x): return np.cos(np.deg2rad(x))
def sin(x): return np.sin(np.deg2rad(x))
def tan(x): return np.tan(np.deg2rad(x))


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
    """ The minimum phase angle for the 3-parameter Linear-Exponential

    Parameters
    ----------
    h : None
        The slope parameter in the units of %/deg. It has no effect on the
        minimum phase angle, but added for consistency with other functions.
        (so that the user can call it by ``alpha_min_le(*theta)``)

    a0, k : float
        The inversion angle and the exponential coefficient in the units of
        deg and deg, respectively.
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
    term1 = (sin(x) / sin(a0))**c1
    term2 = (cos(x/2) / cos(a0/2))**c2
    term3 = sin(x - a0)
    return h * term1 * term2 * term3


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
    """ The 3-parameter Shestopalov function

    Parameters
    ----------
    x : array-like
        The phase angle in degrees.

    h, a0 : float
        The slope parameter and the inversion angle in the units of %/deg
        and deg.

    k : float
        The first exponential coefficient in the units of deg.

    Returns
    -------
    Pr : array-like
        The calculated polarization value in per-cent
    """
    term1 = (1 - np.exp(-k1*x)) / (1 - np.exp(-k1*a0))
    term2 = (x - 180)/(a0-180)
    return h*(x-a0)*term1*term2


_ppcs = {"le": ppc_le, "sh3": ppc_sh3, "sh5": ppc_sh5, "lm_b": ppc_lm, "lm_f": ppc_lm}


def log_likelihood_simple(fun, x, y, yerr, theta):
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


# Uniform distribution for (p1, p2) and ((lo1, hi1), (lo2, hi2)):
def log_prior_uniform(theta, bounds):
    if not (np.all(bounds[0] < theta) and np.all(theta < bounds[1])):
        return -np.inf
    else:
        return 0.0


def log_probability(x, y, yerr, theta, log_prior_fn, log_likelihood_fn):
    lp = log_prior_fn(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_fn(theta, x, y, yerr)


def xy_minimum(fun, theta, xmin_fn=None, **kwargs):
    if xmin_fn is None:
        mini = minimize_scalar(fun, args=theta, **kwargs)
        # Using bracket is a bit faster than using bounds.
        return mini.x, mini.fun
    else:
        xmin = xmin_fn(*theta)
        return xmin, fun(xmin, *theta)


class PPCModel:
    def __init__(self, fun, p0="default", log_likelihood_fn=None,
                 log_prior_fn=None, log_prob_fn=None,
                 bounds="default", amin_pmin_fn=None, amax_pmax_fn=None) -> None:
        """ The PPCModel class

        Parameters
        ----------
        fun : str or callable
            The model name or the model function. If str, it shoule be one of
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

        log_prior_fn: callable
            If given as a function, it must take only the model parameters as
            input and return the log of the prior probability.

        log_likelihood_fn: callable
            If given as a function, it must first take the x, y, yerr and model
            parameters as the following parameter (`theta`). It must return the
            log of the likelihood probability.
        """
        if isinstance(fun, str):
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
            self.bounds = bounds
            self.p0 = p0

        self.npars = self.fun.__code__.co_argcount - 1

        if amin_pmin_fn is not None:
            self.amin_pmin_fn = amin_pmin_fn
        else:
            def _amin_pmin(theta):
                """ Returns the phase angle of the minimum polarization and the
                minimum polarization degree (degree and %, respectively)
                """
                return xy_minimum(
                    self.fun, theta, bracket=(1, 20), method="Brent", tol=0.001,
                    xmin_fn=alpha_min_le if fun == "le" else None
                )
            self.amin_pmin_fn = _amin_pmin
            # The bracket/method/tol are used only if xmin_fun is given (i.e. for "le" case)

        if amin_pmin_fn is not None:
            self.amax_pmax_fn = amax_pmax_fn
        else:
            def _amax_pmax(theta):
                """ Returns the phase angle of the maximum polarization and the
                maximum polarization degree (degree and %, respectively)
                """
                def __neg_fun(*theta):
                    return -self.fun(*theta)

                return xy_minimum(
                    __neg_fun, theta, bracket=(60, 180), method="Brent", tol=0.001,
                    xmin_fn=lambda *args: 180 if fun == "le" else None
                ) * np.array([1, -1])  # The minimum is found for the negative function
            self.amax_pmax_fn = _amax_pmax
            # The bracket/method/tol are used only if xmin_fun is given (i.e. for "le" case)

        if log_likelihood_fn is None:
            def _ll(theta, x, y, yerr):
                """ Log-likelihood function for the given model, data, and parameters
                """
                return log_likelihood_simple(self.fun, x, y, yerr, theta)
            self.log_likelihood_fn = _ll
        else:
            self.log_likelihood_fn = log_likelihood_fn

        if log_prior_fn is None:
            if self.bounds is None:
                self.bounds_lohi = None
                self.log_prior_fn = lambda _: 0.0
            else:
                self.bounds_lohi = (tuple([b[0] for b in self.bounds]),
                                    tuple([b[1] for b in self.bounds]))
                # ^^^ same as bounds but for easy use for some APIs...

                def _lp_unif(theta):
                    """The Uniform distribution of priors based on the given bounds.
                    """
                    return log_prior_uniform(theta, self.bounds_lohi)
                self.log_prior_fn = _lp_unif
        else:
            self.log_prior_fn = log_prior_fn

        if log_prob_fn is None:
            def _lprob(theta, x, y, yerr):
                """ Log-probability function for the given model, data, and
                parameters, prior, and likelihood functions.
                """
                return log_probability(
                    x, y, yerr, theta, self.log_prior_fn, self.log_likelihood_fn)
            self.log_prob_fn = _lprob
        else:
            self.log_prob_fn = log_prob_fn

        self.nll = lambda *args: -self.log_likelihood_fn(*args)

    def solve_lsq(self, x, y, yerr=None, p0="default", bounds="default", **kwargs):
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
        return self.theta_lsq

    def fun_lsq(self, xvals):
        return self.fun(xvals, *self.theta_lsq)
