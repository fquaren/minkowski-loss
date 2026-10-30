"""The proximal anchor is evaluated on the model's own coordinates (DECISIONS §39 open item)."""

import copy

import torch

from src.trainers.reward_finetune import proximal_anchor


class _Recorder(torch.nn.Module):
    """Velocity field a * x that records the states it is evaluated at."""

    def __init__(self, a=1.0):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(float(a)))
        self.seen = []

    def velocity(self, x, t, c):
        self.seen.append(x.detach().clone())
        return self.a * x


def _residual():
    g = torch.Generator().manual_seed(0)
    # scaled log-residuals: mostly O(1), a few large outliers, as in r / sigma_r on hourly data
    r = torch.randn(2, 1, 16, 16, generator=g)
    r[0, 0, 3, 3] = 30.0
    return r


def test_zero_at_initialisation():
    fm = _Recorder(1.3)
    ref = copy.deepcopy(fm)
    assert float(proximal_anchor(fm, ref, _residual(), cond=None)) == 0.0


def test_endpoint_is_the_generated_residual():
    """At t = 1 the anchor state must be r_gen itself, unclamped, not a mm field / 6.42."""
    r = _residual()
    fm, ref = _Recorder(1.0), _Recorder(1.0)
    proximal_anchor(fm, ref, r, cond=None, t=torch.ones(r.shape[0]))
    xt = ref.seen[-1]
    assert torch.allclose(xt, r)
    assert float(xt.max()) == 30.0            # the old version saturated at 1
    assert float(xt.min()) < 0.0              # and could not be negative


def test_gradient_reaches_model_not_reference():
    r = _residual()
    fm, ref = _Recorder(1.5), _Recorder(1.0)
    proximal_anchor(fm, ref, r, cond=None).backward()
    assert fm.a.grad is not None and float(fm.a.grad) > 0
    assert ref.a.grad is None
