"""Tests for MCMC module using emcee."""

import numpy as np
import pytest
from sbppc import PPCModel, ppc_le
from sbppc.tool_emcee import run_emcee, calc_derived_samples

# Skip tests if emcee is not installed (though it should be)
emcee = pytest.importorskip("emcee")


class TestMCMC:
    """Test MCMC workflow."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data from Linear-Exponential model."""
        true_theta = (0.2, 22.0, 10.0)  # h=0.2, a0=22, k=10
        x = np.linspace(5, 60, 20)
        np.random.seed(42)
        noise = 0.05
        y = ppc_le(x, *true_theta) + np.random.normal(0, noise, len(x))
        yerr = np.ones_like(x) * noise
        return x, y, yerr, true_theta

    def test_run_emcee_converges(self, synthetic_data):
        """MCMC should run and find parameters near truth."""
        x, y, yerr, true_theta = synthetic_data
        model = PPCModel("le")  # LE model has 3 parameters

        # Use few walkers/steps for speed in testing
        sampler = run_emcee(
            model, x, y, yerr=yerr,
            nwalkers=16, nsteps=200, burnin=0,  # No discard inside wrapper
            parallel=False, progress=False
        )

        assert sampler.chain.shape == (16, 200, 3)

        # Check if median of chains is reasonable
        flat_samples = sampler.get_chain(discard=100, flat=True)
        medians = np.median(flat_samples, axis=0)

        # Allow generous tolerance for short chain MCMC
        np.testing.assert_allclose(medians, true_theta, rtol=0.2)

    def test_calc_derived_samples(self, synthetic_data):
        """Derived parameters calculation should work."""
        x, y, yerr, _ = synthetic_data
        model = PPCModel("le")
        # Quick fit to get valid theta
        theta = model.solve_lsq(x, y, yerr)

        # Create dummy samples around best fit
        nsamples = 50
        samples = theta + 1e-3 * np.random.randn(nsamples, 3)

        derived = calc_derived_samples(model, samples)

        assert "alpha_min" in derived
        assert "p_min" in derived
        assert len(derived["alpha_min"]) == nsamples
        assert np.all(derived["alpha_min"] > 0)

    def test_parallel_execution(self, synthetic_data):
        """Test parallel execution mode (mocking multiprocessing)."""
        x, y, yerr, _ = synthetic_data
        model = PPCModel("le")

        # Very short run just to check no crash
        sampler = run_emcee(
            model, x, y, yerr=yerr,
            nwalkers=10, nsteps=10, burnin=0,
            parallel=True, progress=False
        )
        assert sampler.chain.shape == (10, 10, 3)
