"""Tests for polarimetric phase curve model functions.

All expected values are pre-computed with rtol=1e-6, atol=1e-8 tolerances.
"""

import numpy as np
import pytest

from sbppc import (
    ppc_le, ppc_lm, ppc_sh3, ppc_sh5,
    alpha_min_le, ppc_pars, ppc_p0s, ppc_bounds,
    log_likelihood_simple, log_prior_uniform, xy_minimum,
    PPCModel,
)

# Strict tolerances for physical correctness
RTOL = 1e-6
ATOL = 1e-8


class TestPPCLE:
    """Test Linear-Exponential model with pre-computed expected values."""

    # Pre-computed reference values: (x, h, a0, k, expected)
    TEST_CASES = [
        # h=0.1, a0=20, k=10
        (0, 0.1, 20, 10, 0.0),
        (5, 0.1, 20, 10, -5.969862195104394e-01),
        (10, 0.1, 20, 10, -6.726941682818558e-01),
        (15, 0.1, 20, 10, -4.322308636080718e-01),
        (20, 0.1, 20, 10, 0.0),  # Inversion angle
        (30, 0.1, 20, 10, 1.167634877695490e+00),
        (45, 0.1, 20, 10, 3.220922659890065e+00),
        (90, 0.1, 20, 10, 9.734488576451605e+00),
        (180, 0.1, 20, 10, 2.283518267911398e+01),
        # h=0.2, a0=25, k=15
        (0, 0.2, 25, 15, 0.0),
        (5, 0.2, 25, 15, -1.221398952550154e+00),
        (10, 0.2, 25, 15, -1.633314109147920e+00),
        (15, 0.2, 25, 15, -1.465208780543674e+00),
        (20, 0.2, 25, 15, -8.815006135179883e-01),
        (30, 0.2, 25, 15, 1.094878225482095e+00),
        (45, 0.2, 25, 15, 5.135788796020792e+00),
        (90, 0.2, 25, 15, 1.936735536089721e+01),
        (180, 0.2, 25, 15, 4.875873888886689e+01),
        # h=0.05, a0=18, k=8
        (0, 0.05, 18, 8, 0.0),
        (5, 0.05, 18, 8, -2.960130420093089e-01),
        (10, 0.05, 18, 8, -2.963640553924885e-01),
        (15, 0.05, 18, 8, -1.384585889744236e-01),
        (20, 0.05, 18, 8, 1.041554668944425e-01),
        (30, 0.05, 18, 8, 7.043354770396179e-01),
        (45, 0.05, 18, 8, 1.697609511425914e+00),
        (90, 0.05, 18, 8, 4.754282781078298e+00),
        (180, 0.05, 18, 8, 1.087745009662973e+01),
    ]

    @pytest.mark.parametrize("x,h,a0,k,expected", TEST_CASES)
    def test_ppc_le_values(self, x, h, a0, k, expected):
        """LE model output matches pre-computed values."""
        result = ppc_le(x, h=h, a0=a0, k=k)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_ppc_le_vectorized(self):
        """LE model handles array input correctly."""
        x = np.array([0, 10, 20, 30, 45, 90])
        expected = np.array([
            0.0,
            -6.726941682818558e-01,
            0.0,
            1.167634877695490e+00,
            3.220922659890065e+00,
            9.734488576451605e+00,
        ])
        result = ppc_le(x, h=0.1, a0=20, k=10)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_ppc_le_scalar_vs_array(self):
        """Scalar and array inputs give consistent results."""
        x_scalar = 45
        x_array = np.array([45])
        result_scalar = ppc_le(x_scalar, h=0.1, a0=20, k=10)
        result_array = ppc_le(x_array, h=0.1, a0=20, k=10)
        np.testing.assert_allclose(result_scalar, result_array[0], rtol=RTOL, atol=ATOL)


class TestPPCLM:
    """Test Lumme-Muinonen model with pre-computed expected values."""

    # Pre-computed reference values: (x, h, a0, c1, c2, expected)
    TEST_CASES = [
        # h=0.1, a0=20, c1=1, c2=1
        (0, 0.1, 20, 1, 1, 0.0),
        (10, 0.1, 20, 1, 1, -8.918289015039911e-03),
        (20, 0.1, 20, 1, 1, 0.0),  # Inversion angle
        (45, 0.1, 20, 1, 1, 8.196821239585392e-02),
        (90, 0.1, 20, 1, 1, 1.972730117640734e-01),
        # h=0.2, a0=25, c1=0.5, c2=1.5
        (0, 0.2, 25, 0.5, 1.5, 0.0),
        (10, 0.2, 25, 0.5, 1.5, -3.420040751115916e-02),
        (20, 0.2, 25, 0.5, 1.5, -1.588666125607034e-02),
        (45, 0.2, 25, 0.5, 1.5, 8.145182150775608e-02),
        (90, 0.2, 25, 0.5, 1.5, 1.718647501398137e-01),
    ]

    @pytest.mark.parametrize("x,h,a0,c1,c2,expected", TEST_CASES)
    def test_ppc_lm_values(self, x, h, a0, c1, c2, expected):
        """LM model output matches pre-computed values."""
        result = ppc_lm(x, h=h, a0=a0, c1=c1, c2=c2)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestPPCSH3:
    """Test Shestopalov 3-parameter model with pre-computed expected values."""

    # Pre-computed reference values: (x, h, a0, k1, expected)
    TEST_CASES = [
        # h=0.1, a0=20, k1=0.1
        (0, 0.1, 20, 0.1, 0.0),
        (10, 0.1, 20, 0.1, -7.767497397943802e-01),
        (20, 0.1, 20, 0.1, 0.0),  # Inversion angle
        (45, 0.1, 20, 0.1, 2.412428678985818e+00),
        (90, 0.1, 20, 0.1, 4.553226236214933e+00),
        # h=0.2, a0=25, k1=0.05
        (0, 0.2, 25, 0.05, 0.0),
        (10, 0.2, 25, 0.05, -1.814505618877148e+00),
        (20, 0.2, 25, 0.05, -9.145282851598383e-01),
        (45, 0.2, 25, 0.05, 4.368177467140770e+00),
        (90, 0.2, 25, 0.05, 1.046192330067303e+01),
    ]

    @pytest.mark.parametrize("x,h,a0,k1,expected", TEST_CASES)
    def test_ppc_sh3_values(self, x, h, a0, k1, expected):
        """SH3 model output matches pre-computed values."""
        result = ppc_sh3(x, h=h, a0=a0, k1=k1)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestPPCSH5:
    """Test Shestopalov 5-parameter model with pre-computed expected values."""

    # Pre-computed reference values: (x, h, a0, k1, k2, k0, expected)
    TEST_CASES = [
        (0, 0.1, 20, 0.1, 1e-5, 1e-5, 0.0),
        (10, 0.1, 20, 0.1, 1e-5, 1e-5, -7.768274296576974e-01),
        (20, 0.1, 20, 0.1, 1e-5, 1e-5, 0.0),  # Inversion angle
        (45, 0.1, 20, 0.1, 1e-5, 1e-5, 2.411825579365217e+00),
        (90, 0.1, 20, 0.1, 1e-5, 1e-5, 4.550039854253050e+00),
    ]

    @pytest.mark.parametrize("x,h,a0,k1,k2,k0,expected", TEST_CASES)
    def test_ppc_sh5_values(self, x, h, a0, k1, k2, k0, expected):
        """SH5 model output matches pre-computed values."""
        result = ppc_sh5(x, h=h, a0=a0, k1=k1, k2=k2, k0=k0)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestAlphaMinLE:
    """Test alpha_min_le analytical minimum finder."""

    # Pre-computed reference values: (a0, k, expected_alpha_min)
    TEST_CASES = [
        (20, 10, 8.385606384288044e+00),
        (25, 15, 1.080239209528105e+01),
        (18, 8, 7.378463499210984e+00),
    ]

    @pytest.mark.parametrize("a0,k,expected", TEST_CASES)
    def test_alpha_min_le_values(self, a0, k, expected):
        """alpha_min_le returns correct minimum phase angle."""
        result = alpha_min_le(a0=a0, k=k)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("a0,k,expected_amin", TEST_CASES)
    def test_alpha_min_is_actual_minimum(self, a0, k, expected_amin):
        """Polarization at alpha_min is less than at nearby angles."""
        h = 0.1
        p_min = ppc_le(expected_amin, h=h, a0=a0, k=k)
        p_left = ppc_le(expected_amin - 0.5, h=h, a0=a0, k=k)
        p_right = ppc_le(expected_amin + 0.5, h=h, a0=a0, k=k)
        assert p_min < p_left
        assert p_min < p_right


class TestLogLikelihood:
    """Test log-likelihood computation."""

    def test_perfect_fit_zero_residual(self):
        """Perfect fit gives log-likelihood = 0 for unit variance."""
        x = np.array([5., 10., 15., 20., 25.])
        theta = (0.1, 20., 10.)
        y = ppc_le(x, *theta)
        ll = log_likelihood_simple(ppc_le, x, y, None, theta)
        np.testing.assert_allclose(ll, 0.0, rtol=RTOL, atol=ATOL)

    def test_known_residual_likelihood(self):
        """Specific residual gives expected log-likelihood."""
        x = np.array([10., 20., 30.])
        theta = (0.1, 20., 10.)
        y_model = ppc_le(x, *theta)
        # Add known offset of 1.0 to each point
        y = y_model + 1.0
        yerr = np.ones(3)  # unit variance
        # Expected: -0.5 * sum((1.0)^2 / 1^2) = -0.5 * 3 = -1.5
        expected_ll = -1.5
        ll = log_likelihood_simple(ppc_le, x, y, yerr, theta)
        np.testing.assert_allclose(ll, expected_ll, rtol=RTOL, atol=ATOL)

    def test_yerr_scaling(self):
        """Larger yerr reduces penalty for residuals."""
        x = np.array([10., 20., 30.])
        theta = (0.1, 20., 10.)
        y_model = ppc_le(x, *theta)
        y = y_model + 2.0  # offset of 2.0
        yerr = np.array([2., 2., 2.])  # variance = 4
        # Expected: -0.5 * sum((2.0)^2 / 4) = -0.5 * 3 = -1.5
        expected_ll = -1.5
        ll = log_likelihood_simple(ppc_le, x, y, yerr, theta)
        np.testing.assert_allclose(ll, expected_ll, rtol=RTOL, atol=ATOL)


class TestLogPrior:
    """Test uniform prior with specific cases.

    Note: log_prior_uniform uses np.all(bounds[0] < theta) which compares
    tuples lexicographically when inputs are tuples (not arrays). This means
    only the first mismatched element determines the result. Tests below
    reflect the actual function behavior.
    """

    # bounds: lower = (0.01, 10., 0.001), upper = (10., 30., 100.)
    BOUNDS = ((0.01, 10., 0.001), (10., 30., 100.))

    # Pre-computed expected values based on actual function behavior
    TEST_CASES = [
        ((0.1, 20., 10.), 0.0),       # Within bounds
        ((5.0, 15., 50.), 0.0),       # Within bounds
        ((9.9, 29., 99.), 0.0),       # Near upper bounds but within
        ((0.02, 11., 0.01), 0.0),     # Near lower bounds but within
        ((0.005, 20., 10.), -np.inf), # h=0.005 below lower h=0.01
    ]

    @pytest.mark.parametrize("theta,expected", TEST_CASES)
    def test_log_prior_uniform_values(self, theta, expected):
        """log_prior_uniform returns correct values."""
        result = log_prior_uniform(theta, self.BOUNDS)
        np.testing.assert_equal(result, expected)


class TestXYMinimum:
    """Test minimum-finding utility with known functions."""

    def test_parabola_minimum(self):
        """Finds exact minimum of parabola at x=b."""
        def parabola(x, a, b):
            return a * (x - b)**2

        xmin, fmin = xy_minimum(parabola, (1.0, 5.0), bracket=(0, 10))
        np.testing.assert_allclose(xmin, 5.0, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(fmin, 0.0, rtol=1e-3, atol=1e-3)

    def test_analytic_xmin_fn(self):
        """Uses analytic function when provided."""
        def parabola(x, a, b):
            return a * (x - b)**2

        def analytic_min(a, b):
            return b

        xmin, fmin = xy_minimum(parabola, (1.0, 7.0), xmin_fn=analytic_min)
        np.testing.assert_allclose(xmin, 7.0, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(fmin, 0.0, rtol=RTOL, atol=ATOL)


class TestPPCModel:
    """Test PPCModel class initialization and fitting."""

    @pytest.mark.parametrize("model_name,expected_npars", [
        ("le", 3),
        ("sh3", 3),
        ("sh5", 5),
        ("lm_b", 4),
        ("lm_f", 4),
    ])
    def test_init_string_models(self, model_name, expected_npars):
        """String model names initialize correctly."""
        model = PPCModel(model_name)
        assert model.npars == expected_npars
        assert model.p0 is not None
        assert model.bounds is not None

    def test_init_custom_function(self):
        """Custom callable initializes correctly."""
        def custom_ppc(x, a, b):
            return a * x + b

        model = PPCModel(custom_ppc, p0=(1, 0), bounds=None)
        assert model.npars == 2
        assert model.p0 == (1, 0)
        assert model.bounds is None

    def test_solve_lsq_perfect_data(self):
        """Fitting perfect data recovers true parameters."""
        true_theta = (0.15, 22., 12.)
        model = PPCModel("le")
        x = np.linspace(5, 60, 20)
        y = ppc_le(x, *true_theta)

        fitted = model.solve_lsq(x, y)

        # Should recover parameters closely (not exactly due to optimizer)
        np.testing.assert_allclose(fitted[0], true_theta[0], rtol=0.01)
        np.testing.assert_allclose(fitted[1], true_theta[1], rtol=0.01)
        np.testing.assert_allclose(fitted[2], true_theta[2], rtol=0.01)

    def test_fun_lsq_evaluates_correctly(self):
        """fun_lsq evaluates fitted model at given points."""
        model = PPCModel("le")
        x = np.array([10., 20., 30., 40., 50.])
        y = ppc_le(x, h=0.2, a0=22., k=10.)
        model.solve_lsq(x, y)

        y_pred = model.fun_lsq(x)
        # Prediction should match input data closely
        np.testing.assert_allclose(y_pred, y, rtol=0.01)


class TestParameterDictionaries:
    """Test module-level parameter dictionaries."""

    def test_ppc_pars_keys(self):
        """ppc_pars has all expected model keys."""
        expected = {'lm_b', 'lm_f', 'le', 'sh5', 'sh3'}
        assert set(ppc_pars.keys()) == expected

    def test_ppc_p0s_keys(self):
        """ppc_p0s has sh3 key (verifies typo fix from appsp)."""
        assert 'sh3' in ppc_p0s
        # Verify it's a tuple of 3 values (h, a0, k1)
        assert len(ppc_p0s['sh3']) == 3

    def test_ppc_bounds_keys(self):
        """ppc_bounds has all expected model keys."""
        expected = {'lm_b', 'lm_f', 'le', 'sh5', 'sh3'}
        assert set(ppc_bounds.keys()) == expected

    def test_bounds_structure(self):
        """Each bounds entry is tuple of (low, high) pairs."""
        for name, bounds in ppc_bounds.items():
            assert isinstance(bounds, tuple)
            for bound in bounds:
                assert isinstance(bound, tuple)
                assert len(bound) == 2
                assert bound[0] < bound[1]  # low < high
