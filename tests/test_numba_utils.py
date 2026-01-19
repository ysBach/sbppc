
import pytest
import numpy as np
try:
    from sbppc.numba_utils import calculate_derived_numba, ppc_le_jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from sbppc import ppc_le, alpha_min_le

@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_ppc_le_jit_values():
    """JIT version matches Python version."""
    x = 45.0
    theta = (0.1, 20.0, 10.0)
    res_jit = ppc_le_jit(x, *theta)
    res_py = ppc_le(x, *theta)
    np.testing.assert_allclose(res_jit, res_py)

@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_calculate_derived_numba_le():
    """Numba batched calculation is correct for LE model."""
    samples = np.array([
        [0.1, 20.0, 10.0],
        [0.2, 25.0, 15.0]
    ])
    # func_id 0 = LE
    amin, pmin, amax, pmax = calculate_derived_numba(samples, 0)

    # Verify against analytic alpha_min_le
    expected_amin_0 = alpha_min_le(a0=20.0, k=10.0)
    expected_pmin_0 = ppc_le(expected_amin_0, 0.1, 20.0, 10.0)

    expected_amin_1 = alpha_min_le(a0=25.0, k=15.0)
    expected_pmin_1 = ppc_le(expected_amin_1, 0.2, 25.0, 15.0)

    # Golden section search has tolerance 1e-5
    np.testing.assert_allclose(amin[0], expected_amin_0, rtol=1e-4)
    np.testing.assert_allclose(pmin[0], expected_pmin_0, rtol=1e-4)
    np.testing.assert_allclose(amin[1], expected_amin_1, rtol=1e-4)
    np.testing.assert_allclose(pmin[1], expected_pmin_1, rtol=1e-4)

@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_ppc_lm_jit_behaves_like_python():
    """Verify helpers (sin/cos) are correctly overridden in JIT."""
    from sbppc.numba_utils import ppc_lm_jit
    # From functions tests: h=0.1, a0=20, c1=1, c2=1, x=45 -> 0.081968...
    res = ppc_lm_jit(45.0, 0.1, 20.0, 1.0, 1.0)
    # Expected: 0.08196821239585392
    assert np.isclose(res, 0.081968212, rtol=1e-5)

