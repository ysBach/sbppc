"""Comprehensive tests for PPC model fitting and MCMC analysis.

This module tests all PPC models (LE, SH3, SH5, LM_B, LM_F) using synthetic data
with hard-coded expected values to ensure reproducibility.
"""

import numpy as np
import pytest
from sbppc import PPCModel, ppc_le, run_emcee, summarize_samples


# ============================================================================
# Shared Test Fixtures
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Generate synthetic LE data with known parameters."""
    np.random.seed(42)
    h_true, a0_true, k_true = 0.2, 20.0, 10.0
    x_obs = np.linspace(1, 35, 20)
    y_true = ppc_le(x_obs, h_true, a0_true, k_true)
    y_err = 0.1 * np.ones_like(x_obs)
    y_obs = y_true + np.random.normal(0, 0.1, len(x_obs))
    return {
        "x": x_obs,
        "y": y_obs,
        "yerr": y_err,
        "theta_true": [h_true, a0_true, k_true],
    }


# ============================================================================
# Hard-Coded Expected LSQ Results
# ============================================================================

EXPECTED_LSQ = {
    "le": {
        "theta_lsq": [0.1937442980278323, 20.184252880193586, 10.96432845170253],
        "amin_lsq": 8.5854197018,
        "pmin_lsq": -1.3181091869,
        "amax_lsq": 180.0,
        "pmax_lsq": 46.3014734860,
    },
    "sh3": {
        "theta_lsq": [0.1975416513536957, 20.206525497468945, 0.04418123086203022],
        "amin_lsq": 8.7736597085,
        "pmin_lsq": -1.3170050762,
        "amax_lsq": 101.6976863960,
        "pmax_lsq": 13.2099285544,
    },
    "sh5": {
        "theta_lsq": [0.1911424553907202, 20.17439901202299, 0.09056562820631907,
                     -0.010489459899193898, -0.018533529909954292],
        "amin_lsq": 8.4665810483,
        "pmin_lsq": -1.3179297826,
        "amax_lsq": 140.9865888691,
        "pmax_lsq": 42.5710257794,
    },
    "lm_b": {
        "theta_lsq": [10.0, 20.085603456960644, 0.6345506209765426, 1e-06],
        "amin_lsq": 7.8409301183,
        "pmin_lsq": -1.1805811508,
        "amax_lsq": 102.2444886347,
        "pmax_lsq": 19.2361359255,
    },
    "lm_f": {
        "theta_lsq": [10.0, 20.139890582317957, 0.6245009482056758, -0.5724451615128253],
        "amin_lsq": 7.8075098843,
        "pmin_lsq": -1.1858836369,
        "amax_lsq": 118.2794896275,
        "pmax_lsq": 25.8471489946,
    },
}


# ============================================================================
# Test Classes for LSQ Fitting
# ============================================================================

class TestPPCModelLSQFitting:
    """Test least-squares fitting for all PPC models with exact value verification."""

    @pytest.mark.parametrize("model_name", ["le", "sh3", "sh5", "lm_b", "lm_f"])
    def test_theta_lsq_values(self, synthetic_data, model_name):
        """Test that theta_lsq matches expected values."""
        model = PPCModel(model_name)
        model.solve_lsq(synthetic_data["x"], synthetic_data["y"], yerr=synthetic_data["yerr"])

        expected = EXPECTED_LSQ[model_name]["theta_lsq"]
        np.testing.assert_allclose(model.theta_lsq, expected, rtol=1e-3)

    @pytest.mark.parametrize("model_name", ["le", "sh3", "sh5", "lm_b", "lm_f"])
    def test_amin_lsq_values(self, synthetic_data, model_name):
        """Test that amin_lsq matches expected values."""
        model = PPCModel(model_name)
        model.solve_lsq(synthetic_data["x"], synthetic_data["y"], yerr=synthetic_data["yerr"])

        expected = EXPECTED_LSQ[model_name]["amin_lsq"]
        np.testing.assert_allclose(model.amin_lsq, expected, rtol=1e-4)

    @pytest.mark.parametrize("model_name", ["le", "sh3", "sh5", "lm_b", "lm_f"])
    def test_pmin_lsq_values(self, synthetic_data, model_name):
        """Test that pmin_lsq matches expected values."""
        model = PPCModel(model_name)
        model.solve_lsq(synthetic_data["x"], synthetic_data["y"], yerr=synthetic_data["yerr"])

        expected = EXPECTED_LSQ[model_name]["pmin_lsq"]
        np.testing.assert_allclose(model.pmin_lsq, expected, rtol=1e-4)

    @pytest.mark.parametrize("model_name", ["le", "sh3", "sh5", "lm_b", "lm_f"])
    def test_amax_lsq_values(self, synthetic_data, model_name):
        """Test that amax_lsq matches expected values."""
        model = PPCModel(model_name)
        model.solve_lsq(synthetic_data["x"], synthetic_data["y"], yerr=synthetic_data["yerr"])

        expected = EXPECTED_LSQ[model_name]["amax_lsq"]
        np.testing.assert_allclose(model.amax_lsq, expected, rtol=1e-4)

    @pytest.mark.parametrize("model_name", ["le", "sh3", "sh5", "lm_b", "lm_f"])
    def test_pmax_lsq_values(self, synthetic_data, model_name):
        """Test that pmax_lsq matches expected values."""
        model = PPCModel(model_name)
        model.solve_lsq(synthetic_data["x"], synthetic_data["y"], yerr=synthetic_data["yerr"])

        expected = EXPECTED_LSQ[model_name]["pmax_lsq"]
        np.testing.assert_allclose(model.pmax_lsq, expected, rtol=1e-4)


class TestParNames:
    """Test that par_names attribute is correctly set."""

    @pytest.mark.parametrize("model_name,expected_names", [
        ("le", ["h", "a0", "k"]),
        ("sh3", ["h", "a0", "k1"]),
        ("sh5", ["h", "a0", "k1", "k2", "k0"]),
        ("lm_b", ["h", "a0", "c1", "c2"]),
        ("lm_f", ["h", "a0", "c1", "c2"]),
    ])
    def test_par_names(self, model_name, expected_names):
        """Test that par_names matches expected parameter names."""
        model = PPCModel(model_name)
        assert model.par_names == expected_names


# ============================================================================
# MCMC Tests
# ============================================================================

# Hard-coded MCMC reference values (50th percentile medians)
MCMC_REFERENCE_MEDIANS = {
    "le": {
        "h": 0.193765,
        "a0": 20.212329,
        "k": 11.165812,
        "amin": 8.623927,
        "pmin": -1.313904,
    }
}


class TestMCMCFitting:
    """Test MCMC fitting produces results within expected bounds.

    Note: MCMC is stochastic, so we only verify that the hard-coded median
    falls within the 16-84th percentile bounds of new runs.
    """

    def test_mcmc_le_percentile_bounds(self, synthetic_data):
        """Test that reference medians fall within MCMC credible intervals."""
        np.random.seed(456)  # Different seed for this test

        model = PPCModel("le")
        sampler = run_emcee(
            model, synthetic_data["x"], synthetic_data["y"], synthetic_data["yerr"],
            nwalkers=32, nsteps=500, burnin=100, parallel=True, progress=False
        )
        flat_samples = sampler.get_chain(discard=100, flat=True, thin=5)

        result = summarize_samples(model, flat_samples, percentiles=(16, 50, 84))

        # Check each parameter: reference median should be within 16-84 bounds
        for label, ref_median in MCMC_REFERENCE_MEDIANS["le"].items():
            pct = result["percentiles"][label]
            p16, p84 = pct[16], pct[84]

            # Allow some margin (3x the interval) for stochastic variation
            margin = (p84 - p16) * 1.5
            lower_bound = p16 - margin
            upper_bound = p84 + margin

            assert lower_bound < ref_median < upper_bound, (
                f"Reference median {label}={ref_median:.6f} not within extended bounds "
                f"[{lower_bound:.6f}, {upper_bound:.6f}]"
            )


class TestSummarizeSamples:
    """Test the summarize_samples helper function."""

    def test_summarize_samples_labels(self, synthetic_data):
        """Test that summarize_samples returns correct labels."""
        np.random.seed(789)
        model = PPCModel("le")
        sampler = run_emcee(
            model, synthetic_data["x"], synthetic_data["y"], synthetic_data["yerr"],
            nwalkers=16, nsteps=100, burnin=50, parallel=False, progress=False
        )
        flat_samples = sampler.get_chain(discard=50, flat=True, thin=5)

        result = summarize_samples(model, flat_samples)

        assert result["labels"] == ["h", "a0", "k", "amin", "pmin"]

    def test_summarize_samples_with_max(self, synthetic_data):
        """Test summarize_samples with include_max=True."""
        np.random.seed(789)
        model = PPCModel("le")
        sampler = run_emcee(
            model, synthetic_data["x"], synthetic_data["y"], synthetic_data["yerr"],
            nwalkers=16, nsteps=100, burnin=50, parallel=False, progress=False
        )
        flat_samples = sampler.get_chain(discard=50, flat=True, thin=5)

        result = summarize_samples(model, flat_samples, include_max=True)

        assert result["labels"] == ["h", "a0", "k", "amin", "pmin", "amax", "pmax"]

    def test_summarize_samples_percentiles(self, synthetic_data):
        """Test that summarize_samples returns correct percentile structure."""
        np.random.seed(789)
        model = PPCModel("le")
        sampler = run_emcee(
            model, synthetic_data["x"], synthetic_data["y"], synthetic_data["yerr"],
            nwalkers=16, nsteps=100, burnin=50, parallel=False, progress=False
        )
        flat_samples = sampler.get_chain(discard=50, flat=True, thin=5)

        result = summarize_samples(model, flat_samples, percentiles=(5, 50, 95))

        # Check structure
        for label in result["labels"]:
            assert 5 in result["percentiles"][label]
            assert 50 in result["percentiles"][label]
            assert 95 in result["percentiles"][label]
