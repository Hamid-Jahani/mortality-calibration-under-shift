"""Validation of the conformal wrappers: given a base model whose native
uncertainty is deliberately MISCALIBRATED (sigma scaled by 0.4, so native
95% intervals cover far below nominal), each wrapper must restore coverage
from calibration data alone — the property the audit's conformal arms exist
to test. Pattern of tests/test_synthetic_calibration.py; finite-sample slack
per the module docstring (uniform-sampling quantile shrinkage)."""
import numpy as np
import pytest

from test_synthetic_calibration import simulate_plc, H
from mortcal.eval import interval_coverage, joint_path_coverage
from mortcal.models import PoissonLeeCarter
from mortcal.uq import SplitConformalMx, EnbPIMx, CopulaPathConformal

N_WORLDS = 12
ALPHA = 0.10          # construct 90% intervals
N_SAMP = 600


class NarrowPLC(PoissonLeeCarter):
    """Poisson-LC with its k_t innovation and drift uncertainty crushed 0.4x —
    a stand-in for the literature's overconfident native mechanisms."""

    def fit(self, D, E):
        super().fit(D, E)
        self.kt.sigma *= 0.4
        self.kt.se_mu *= 0.4
        return self


def _worlds():
    rng = np.random.default_rng(20260825)
    for _ in range(N_WORLDS):
        (D, E), true_mx = simulate_plc(rng)
        yield D, E, np.log(true_mx)[:H], rng


def _wrapper_coverage(make):
    covs, joints, native_covs, native_joints = [], [], [], []
    for D, E, truth, rng in _worlds():
        base = NarrowPLC().fit(D, E)
        s = np.log(base.sample_mx(H, N_SAMP, rng))
        cov, _ = interval_coverage(s, truth, 1 - ALPHA)
        native_covs.append(cov.mean())
        native_joints.append(joint_path_coverage(s, truth, 1 - ALPHA))

        w = make().fit(D, E)
        sw = np.log(w.sample_mx(H, N_SAMP, rng))
        covw, _ = interval_coverage(sw, truth, 1 - ALPHA)
        covs.append(covw.mean())
        joints.append(joint_path_coverage(sw, truth, 1 - ALPHA))
    return (float(np.mean(covs)), float(np.mean(joints)),
            float(np.mean(native_covs)), float(np.mean(native_joints)))


@pytest.fixture(scope="module")
def native_baseline():
    """The miscalibration we claim to repair must actually exist."""
    covs = []
    for D, E, truth, rng in _worlds():
        base = NarrowPLC().fit(D, E)
        s = np.log(base.sample_mx(H, N_SAMP, rng))
        cov, _ = interval_coverage(s, truth, 1 - ALPHA)
        covs.append(cov.mean())
    return float(np.mean(covs))


def test_narrow_base_actually_undercovers(native_baseline):
    assert native_baseline < 0.72, f"native coverage {native_baseline:.3f} not broken enough to test repair"


def test_split_conformal_restores_marginal_coverage():
    cov, _, native, _ = _wrapper_coverage(
        lambda: SplitConformalMx(NarrowPLC, alpha=ALPHA, n_median_samples=400))
    assert cov >= 0.85, f"split conformal coverage {cov:.3f} (native {native:.3f})"


def test_enbpi_restores_marginal_coverage():
    cov, _, native, _ = _wrapper_coverage(
        lambda: EnbPIMx(NarrowPLC, alpha=ALPHA, n_median_samples=400))
    assert cov >= 0.85, f"EnbPI coverage {cov:.3f} (native {native:.3f})"


def test_copula_path_conformal_restores_joint_coverage():
    cov, joint, native, native_joint = _wrapper_coverage(
        lambda: CopulaPathConformal(NarrowPLC, alpha=ALPHA, n_median_samples=400))
    assert native_joint < 0.40, f"native joint {native_joint:.3f} not broken — test uninformative"
    assert joint >= 0.75, f"copula joint path coverage {joint:.3f} (native joint {native_joint:.3f})"
    assert cov >= 0.85, f"copula marginal coverage {cov:.3f}"
