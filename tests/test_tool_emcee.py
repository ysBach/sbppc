"""Tests for emcee visualization utilities."""

import numpy as np
import pytest

from sbppc.tool_emcee import trace_plot


class TestTracePlot:
    """Test trace_plot function."""

    @pytest.fixture
    def mock_axes(self):
        """Create mock axes objects."""
        class MockAxes:
            def __init__(self):
                self.plots = []
                self.ylabel = None
                self.xlim = None

            def plot(self, *args, **kwargs):
                self.plots.append((args, kwargs))

            def set_ylabel(self, label):
                self.ylabel = label

            def set_xlim(self, *args):
                self.xlim = args

        return np.array([MockAxes(), MockAxes(), MockAxes()])

    def test_flat_samples_2d(self, mock_axes):
        """Should handle 2D (flattened) samples."""
        samples = np.random.randn(100, 3)
        trace_plot(mock_axes, samples)
        for ax in mock_axes:
            assert len(ax.plots) == 1
            assert ax.xlim == (0, 100)

    def test_full_chain_3d(self, mock_axes):
        """Should handle 3D (full chain) samples."""
        samples = np.random.randn(50, 4, 3)  # 50 steps, 4 walkers, 3 params
        trace_plot(mock_axes, samples)
        for ax in mock_axes:
            assert ax.xlim == (0, 50)

    def test_with_labels(self, mock_axes):
        """Should set labels when provided."""
        samples = np.random.randn(100, 3)
        labels = ['h', 'a0', 'k']
        trace_plot(mock_axes, samples, labels=labels)
        for i, ax in enumerate(mock_axes):
            assert ax.ylabel == labels[i]

    def test_wrong_ndim_raises(self, mock_axes):
        """Should raise ValueError for wrong sample dimensions."""
        samples = np.random.randn(100)  # 1D
        with pytest.raises(ValueError, match="is not 2 or 3"):
            trace_plot(mock_axes, samples)

    def test_axes_size_mismatch_raises(self):
        """Should raise ValueError when axes count != npars."""
        class MockAxes:
            def plot(self, *args, **kwargs): pass
            def set_ylabel(self, label): pass
            def set_xlim(self, *args): pass

        axes = np.array([MockAxes(), MockAxes()])  # 2 axes
        samples = np.random.randn(100, 3)  # 3 params
        with pytest.raises(ValueError, match="is different from"):
            trace_plot(axes, samples)

    def test_labels_size_mismatch_raises(self, mock_axes):
        """Should raise ValueError when labels count != npars."""
        samples = np.random.randn(100, 3)
        labels = ['h', 'a0']  # Only 2 labels for 3 params
        with pytest.raises(ValueError, match="is different from"):
            trace_plot(mock_axes, samples, labels=labels)
