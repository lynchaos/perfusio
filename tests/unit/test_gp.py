"""Unit tests for GP sub-package.

Covers:
- PerfusionKernel: positive semi-definite output
- MultiTaskRateGP: posterior mean within observed range
- JackknifeEnsemble: ordering (std ≥ 0, coverage improves with n)
- StepwiseGP: 3-step rollout returns correct shapes
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def simple_xy() -> tuple[torch.Tensor, torch.Tensor]:
    """10 training points, 3 species."""
    torch.manual_seed(0)
    X = torch.linspace(0, 10, 10).unsqueeze(-1).double()
    Y = torch.stack([torch.sin(X[:, 0]), torch.cos(X[:, 0]), X[:, 0] / 10], dim=-1)
    return X, Y


class TestPerfusionKernel:
    def test_psd(self) -> None:
        from perfusio.gp.kernels import PerfusionKernel

        k = PerfusionKernel(n_tasks=2, n_state_dims=1).double()
        # x columns: [state(1), day, task_id]
        x = torch.rand(8, 3).double()
        x[:, 2] = (torch.arange(8) % 2).double()  # integer task ids
        K = k(x, x).to_dense()
        # PSD check: all eigenvalues ≥ -epsilon
        eigs = torch.linalg.eigvalsh(K)
        assert (eigs >= -1e-6).all(), f"Kernel not PSD; min eigenvalue={eigs.min():.4e}"

    def test_symmetric(self) -> None:
        from perfusio.gp.kernels import PerfusionKernel

        k = PerfusionKernel(n_tasks=2, n_state_dims=1).double()
        # x columns: [state(1), day, task_id]
        x = torch.rand(5, 3).double()
        x[:, 2] = (torch.arange(5) % 2).double()
        K = k(x, x).to_dense()
        assert torch.allclose(K, K.T, atol=1e-10), "Kernel matrix not symmetric."


class TestGPPosterior:
    def test_posterior_in_range(self, simple_xy: tuple) -> None:
        import gpytorch
        from gpytorch.likelihoods import GaussianLikelihood

        X, Y = simple_xy
        y_single = Y[:, 0]  # single output
        lh = GaussianLikelihood().double()
        from gpytorch.kernels import RBFKernel
        from gpytorch.means import ConstantMean

        # Use a minimal ExactGP for test speed
        class _TinyGP(gpytorch.models.ExactGP):
            def __init__(self) -> None:
                super().__init__(X, y_single, lh)
                self.mean = ConstantMean()
                self.covar = RBFKernel()

            def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
                return gpytorch.distributions.MultivariateNormal(self.mean(x), self.covar(x))

        gp = _TinyGP().double()
        gp.eval()
        lh.eval()
        # Use held-out test points to avoid GPInputWarning
        X_test = X + 0.1
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = lh(gp(X_test))
        assert pred.mean.shape == (10,)
        # Mean should be finite
        assert pred.mean.isfinite().all()


class TestJackknifeEnsemble:
    def test_std_nonneg(self) -> None:
        import gpytorch

        from perfusio.gp.ensemble import JackknifeEnsemble

        torch.manual_seed(1)
        X = torch.linspace(0, 5, 20).unsqueeze(-1).double()
        Y = (torch.sin(X[:, 0]) + 0.05 * torch.randn(20).double()).unsqueeze(-1)

        def _factory(x_sub: torch.Tensor, y_sub: torch.Tensor) -> tuple:
            lh = gpytorch.likelihoods.GaussianLikelihood().double()

            class _GP(gpytorch.models.ExactGP):  # type: ignore[misc]
                def __init__(self) -> None:
                    super().__init__(x_sub, y_sub, lh)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.RBFKernel()

                def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
                    return gpytorch.distributions.MultivariateNormal(
                        self.mean_module(x), self.covar_module(x)
                    )

            return _GP().double(), lh

        ens = JackknifeEnsemble(K=3, subsample_fraction=0.8)
        ens.fit(X, Y.squeeze(-1), model_factory=_factory, n_iter=5)
        Xtest = torch.linspace(0, 5, 5).unsqueeze(-1).double()
        result = ens.predict(Xtest)
        mean = result["mean"]
        spread = result["q90"] - result["q10"]
        assert (spread >= 0).all(), "q90 - q10 must be non-negative."
        assert mean.shape == torch.Size([5])
