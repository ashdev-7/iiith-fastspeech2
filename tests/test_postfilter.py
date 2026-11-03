"""GV post-filter: restores per-bin variance, keeps the per-bin mean, is a no-op at strength 0."""
import pytest

np = pytest.importorskip("numpy")
import postfilter  # noqa: E402


def test_gv_postfilter_restores_variance():
    rng = np.random.default_rng(0)
    nat = rng.normal(-5, 2.0, size=(200, 80))
    pred = -5 + (nat + 5) * 0.8                           # over-smoothed: 64 % of the variance
    ratio = postfilter.estimate_ratio([nat], [pred])
    assert np.allclose(ratio, 1 / 0.64, rtol=1e-6)
    out = postfilter.apply(pred, ratio, 1.0)
    assert np.allclose(out.var(0), nat.var(0), rtol=1e-6)
    assert np.allclose(out.mean(0), pred.mean(0))
    assert postfilter.apply(pred, ratio, 0.0) is pred
    half = postfilter.apply(pred, ratio, 0.5)
    assert np.all(pred.var(0) < half.var(0)) and np.all(half.var(0) < out.var(0))
    # never attenuates, never amplifies beyond MAX_SCALE
    assert np.allclose(postfilter.apply(pred, [0.5] * 80, 1.0), pred)
    assert np.all(postfilter.apply(pred, [100.0] * 80, 1.0).std(0) <= pred.std(0) * postfilter.MAX_SCALE + 1e-9)
