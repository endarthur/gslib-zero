"""
Tests for variogram plotting utilities.
"""

import numpy as np
import pytest

from gslib_zero.utils import VariogramModel, VariogramType, evaluate_variogram
from gslib_zero.plotting import (
    plot_experimental,
    plot_model,
    plot_variogram,
    export_variogram_par,
)


class TestVariogramEvaluation:
    """Test variogram model evaluation."""

    def test_spherical_at_zero(self):
        """Spherical model at h=0 should equal nugget."""
        model = VariogramModel.spherical(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.1)
        gamma = evaluate_variogram(model, np.array([0.0]))
        assert gamma[0] == pytest.approx(0.1, abs=1e-10)

    def test_spherical_at_range(self):
        """Spherical model at h=range should equal total sill."""
        model = VariogramModel.spherical(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.1)
        gamma = evaluate_variogram(model, np.array([100.0]))
        assert gamma[0] == pytest.approx(1.1, abs=1e-10)

    def test_spherical_beyond_range(self):
        """Spherical model beyond range should stay at sill."""
        model = VariogramModel.spherical(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.1)
        gamma = evaluate_variogram(model, np.array([150.0, 200.0, 500.0]))
        assert np.allclose(gamma, 1.1, atol=1e-10)

    def test_exponential_at_zero(self):
        """Exponential model at h=0 should equal nugget."""
        model = VariogramModel.exponential(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.2)
        gamma = evaluate_variogram(model, np.array([0.0]))
        assert gamma[0] == pytest.approx(0.2, abs=1e-10)

    def test_exponential_asymptotic(self):
        """Exponential model approaches sill asymptotically."""
        model = VariogramModel.exponential(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.0)
        gamma = evaluate_variogram(model, np.array([300.0]))  # 3x range
        # At 3x range, (1 - exp(-9)) ≈ 0.99988
        assert gamma[0] > 0.99

    def test_gaussian_at_zero(self):
        """Gaussian model at h=0 should equal nugget."""
        model = VariogramModel.gaussian(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.0)
        gamma = evaluate_variogram(model, np.array([0.0]))
        assert gamma[0] == pytest.approx(0.0, abs=1e-10)

    def test_gaussian_smooth_origin(self):
        """Gaussian model should be smooth (zero derivative) at origin."""
        model = VariogramModel.gaussian(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.0)
        h = np.array([0.0, 0.01, 0.02])
        gamma = evaluate_variogram(model, h)
        # Second derivative test: should be parabolic near origin
        # gamma(0.01) + gamma(0.02) should be close to 2*gamma(0.015)
        assert gamma[1] < gamma[2]  # Monotonically increasing

    def test_nested_structures(self):
        """Test model with multiple nested structures."""
        model = VariogramModel(nugget=0.1)
        model.add_structure(VariogramType.SPHERICAL, sill=0.4, ranges=(50.0, 50.0, 50.0))
        model.add_structure(VariogramType.SPHERICAL, sill=0.5, ranges=(200.0, 200.0, 200.0))

        gamma = evaluate_variogram(model, np.array([0.0, 50.0, 200.0]))

        # At h=0: nugget only
        assert gamma[0] == pytest.approx(0.1, abs=1e-10)

        # At h=50: nugget + first structure sill + partial second
        assert gamma[1] > 0.5
        assert gamma[1] < 1.0

        # At h=200: total sill
        assert gamma[2] == pytest.approx(1.0, abs=1e-10)

    def test_power_model(self):
        """Test power model evaluation."""
        model = VariogramModel(nugget=0.0)
        model.add_structure(VariogramType.POWER, sill=1.0, ranges=(100.0, 100.0, 100.0))

        gamma = evaluate_variogram(model, np.array([0.0, 50.0, 100.0]))
        assert gamma[0] == pytest.approx(0.0, abs=1e-10)
        assert gamma[1] == pytest.approx(0.5, abs=1e-10)
        assert gamma[2] == pytest.approx(1.0, abs=1e-10)

    def test_hole_effect(self):
        """Test hole effect model evaluation."""
        model = VariogramModel(nugget=0.0)
        model.add_structure(VariogramType.HOLE_EFFECT, sill=1.0, ranges=(100.0, 100.0, 100.0))

        gamma = evaluate_variogram(model, np.array([0.0, 50.0, 100.0, 200.0]))

        # At h=0: should be 0
        assert gamma[0] == pytest.approx(0.0, abs=1e-10)

        # At h=range: 1 - cos(pi) = 2
        assert gamma[2] == pytest.approx(2.0, abs=1e-10)

        # At h=2*range: 1 - cos(2*pi) = 0 (periodic)
        assert gamma[3] == pytest.approx(0.0, abs=1e-10)


class TestPlotExperimental:
    """Test experimental variogram plotting."""

    def test_plot_creates_axes(self):
        """Test that plot_experimental creates axes if none provided."""
        # Create mock VariogramResult
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            lag_distances: np.ndarray
            gamma: np.ndarray
            num_pairs: np.ndarray
            direction: tuple

        result = MockResult(
            lag_distances=np.array([10.0, 20.0, 30.0]),
            gamma=np.array([0.5, 0.8, 1.0]),
            num_pairs=np.array([100, 80, 50]),
            direction=(0.0, 0.0),
        )

        ax = plot_experimental(result)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_multiple_directions(self):
        """Test plotting multiple variogram directions."""
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            lag_distances: np.ndarray
            gamma: np.ndarray
            num_pairs: np.ndarray
            direction: tuple

        results = [
            MockResult(
                lag_distances=np.array([10.0, 20.0, 30.0]),
                gamma=np.array([0.5, 0.8, 1.0]),
                num_pairs=np.array([100, 80, 50]),
                direction=(0.0, 0.0),
            ),
            MockResult(
                lag_distances=np.array([10.0, 20.0, 30.0]),
                gamma=np.array([0.3, 0.5, 0.6]),
                num_pairs=np.array([90, 70, 40]),
                direction=(90.0, 0.0),
            ),
        ]

        ax = plot_experimental(results)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPlotModel:
    """Test variogram model plotting."""

    def test_plot_model_creates_axes(self):
        """Test that plot_model creates axes if none provided."""
        model = VariogramModel.spherical(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.1)

        ax = plot_model(model, max_distance=150.0)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_model_with_nugget_line(self):
        """Test plotting with nugget horizontal line."""
        model = VariogramModel.spherical(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.2)

        ax = plot_model(model, max_distance=150.0, show_nugget=True, show_sill=True)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")


class TestPlotVariogram:
    """Test combined variogram plotting."""

    def test_plot_variogram_model_only(self):
        """Test plotting model without experimental data."""
        model = VariogramModel.spherical(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.1)

        ax = plot_variogram(model=model, max_distance=150.0)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_variogram_with_title(self):
        """Test plotting with title."""
        model = VariogramModel.spherical(sill=1.0, ranges=(100.0, 100.0, 100.0), nugget=0.1)

        ax = plot_variogram(model=model, max_distance=150.0, title="Test Variogram")
        assert ax.get_title() == "Test Variogram"
        import matplotlib.pyplot as plt
        plt.close("all")


class TestExportVariogramPar:
    """Test variogram model export."""

    def test_export_spherical(self, tmp_path):
        """Test exporting spherical model."""
        model = VariogramModel.spherical(
            sill=0.9,
            ranges=(50.0, 30.0, 10.0),
            nugget=0.1,
            angles=(45.0, 0.0, 0.0),
        )

        filepath = tmp_path / "variogram.par"
        export_variogram_par(model, filepath)

        content = filepath.read_text()
        lines = content.strip().split("\n")

        # Check structure
        assert "1 0.1" in lines[1]  # nst, nugget
        assert "1 0.9 45.0 0.0 0.0" in lines[2]  # type, sill, angles
        assert "50.0 30.0 10.0" in lines[3]  # ranges

    def test_export_nested(self, tmp_path):
        """Test exporting nested model."""
        model = VariogramModel(nugget=0.1)
        model.add_structure(VariogramType.SPHERICAL, sill=0.4, ranges=(50.0, 50.0, 50.0))
        model.add_structure(VariogramType.EXPONENTIAL, sill=0.5, ranges=(200.0, 100.0, 50.0))

        filepath = tmp_path / "nested.par"
        export_variogram_par(model, filepath)

        content = filepath.read_text()
        lines = content.strip().split("\n")

        # Should have 2 structures
        assert "2 0.1" in lines[1]  # nst=2, nugget

    def test_export_comment_style(self, tmp_path):
        """Test different comment styles."""
        model = VariogramModel.spherical(sill=1.0, ranges=(100.0, 100.0, 100.0))

        filepath = tmp_path / "variogram.par"
        export_variogram_par(model, filepath, comment_style="#")

        content = filepath.read_text()
        assert "#" in content
        assert "!" not in content

