"""The synthetic market has a known effect. Every estimator must find it, and must not find one where there is none."""
import numpy as np
import pytest

from fuellab import noonrule as A
from fuellab.daily import summarise
from fuellab.simulate import make_market


def _panel(delta, seed):
    m = make_market(n_de=40, n_fr=30, start="2026-01-12", end="2026-05-31", delta=delta, seed=seed)
    ev = m.events
    de = summarise(ev[ev["station"].str.startswith("DE")], "2026-01-12", "2026-05-31", warmup_days=1)
    fr = summarise(ev[ev["station"].str.startswith("FR")], "2026-01-12", "2026-05-31", warmup_days=1)
    return A.build_panel(de, fr), m


@pytest.fixture(scope="module")
def effect():
    return _panel(0.02, 3)


@pytest.fixture(scope="module")
def null():
    return _panel(0.0, 4)


def test_did_recovers_known_effect(effect):
    p, _ = effect
    r = A.did(p)
    assert abs(r["coef"] - 2.0) < 0.15          # cents
    assert r["ci_low"] < 2.0 < r["ci_high"] or abs(r["coef"] - 2.0) < 0.05


def test_no_effect_when_there_is_none(null):
    p, _ = null
    r = A.did(p)
    assert abs(r["coef"]) < 0.15


def test_event_study_has_flat_pre_period(effect):
    es = A.event_study(effect[0])
    pre = es[es["week"] < -1]["coef"]
    post = es[es["week"] >= 0]["coef"]
    assert pre.abs().max() < 0.3 and abs(post.mean() - 2.0) < 0.2


def test_mechanics_detect_the_rule(effect):
    m = A.mechanics(effect[0])
    assert m.loc[("DE", "calm"), "share_of_increases_at_noon"] < 0.05
    assert m.loc[("DE", "rule"), "share_of_increases_at_noon"] > 0.5
    assert m.loc[("DE", "rule"), "increases_per_day"] < m.loc[("DE", "calm"), "increases_per_day"]


def test_profile_peaks_at_noon_after_the_rule(effect):
    prof = A.profile(effect[0])
    assert prof["rule"].idxmax() == 12
    assert np.isclose(prof["rule"].mean(), 0.0, atol=0.3)
