"""Unit tests for spectral fitting models."""


import builtins

import numpy as np

from brimfile.fitting_models import FitModel, gaussian, get_fit_model, lorentzian, voigt


class TestGetFitModel:
    """Tests for fit model dispatch."""

    def test_get_fit_model_voigt(self):
        """Voigt should be available from fit model dispatch."""
        assert get_fit_model(FitModel.Voigt) is voigt


class TestVoigt:
    """Tests for the NumPy-only pseudo-Voigt implementation."""

    def test_voigt_does_not_import_scipy(self, monkeypatch):
        """Voigt evaluation should work even when SciPy imports are blocked."""
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("scipy"):
                raise ModuleNotFoundError("scipy blocked for test")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        x = np.linspace(-1.0, 1.0, 11)
        y = voigt(x, nu0=0.0, gamma_lorentz=0.5, gamma_gauss=0.5, a=2.0, b=0.1)

        assert np.all(np.isfinite(y))

    def test_voigt_reduces_to_gaussian(self):
        """Zero Lorentzian width should match the Gaussian model exactly."""
        x = np.linspace(-2.0, 2.0, 101)

        y_voigt = voigt(x, nu0=0.0, gamma_lorentz=0.0, gamma_gauss=0.8, a=1.7, b=0.2)
        y_gaussian = gaussian(x, nu0=0.0, gamma=0.8, a=1.7, b=0.2)

        np.testing.assert_allclose(y_voigt, y_gaussian)

    def test_voigt_reduces_to_lorentzian(self):
        """Zero Gaussian width should match the Lorentzian model exactly."""
        x = np.linspace(-2.0, 2.0, 101)

        y_voigt = voigt(x, nu0=0.0, gamma_lorentz=0.8, gamma_gauss=0.0, a=1.7, b=0.2)
        y_lorentzian = lorentzian(x, nu0=0.0, gamma=0.8, a=1.7, b=0.2)

        np.testing.assert_allclose(y_voigt, y_lorentzian)

    def test_voigt_center_height(self):
        """Peak height at center should be amplitude plus baseline."""
        x = np.array([0.0])
        y = voigt(x, nu0=0.0, gamma_lorentz=0.7, gamma_gauss=1.2, a=3.0, b=0.4)

        np.testing.assert_allclose(y, np.array([3.4]))
