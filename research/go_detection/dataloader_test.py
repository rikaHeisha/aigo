import numpy as np
import pytest
from go_detection.dataloader import DistSampler, NonReplacementSampler


def test_infinite_sampler_no_shuffle():
    length = 5

    sampler = NonReplacementSampler(length, shuffle=False, repeat=False)
    sampler_iter = iter(sampler)
    indices = [next(sampler_iter) for _ in range(length)]
    assert indices == [0, 1, 2, 3, 4]
    with pytest.raises(StopIteration):
        next(sampler_iter)

    sampler = NonReplacementSampler(length, shuffle=False, repeat=True)
    sampler_iter = iter(sampler)
    indices = [next(sampler_iter) for _ in range(length)]
    assert indices == [0, 1, 2, 3, 4]
    assert next(sampler_iter) == 0


def test_infinite_sampler_shuffle():
    length = 5

    sampler = NonReplacementSampler(length, shuffle=True, repeat=False)
    sampler_iter = iter(sampler)
    indices = [next(sampler_iter) for _ in range(length)]
    assert set(indices) == {0, 1, 2, 3, 4}
    with pytest.raises(StopIteration):
        next(sampler_iter)

    sampler = NonReplacementSampler(length, shuffle=True, repeat=True)
    sampler_iter = iter(sampler)
    indices = [next(sampler_iter) for _ in range(length + 1)]
    assert set(indices) == {0, 1, 2, 3, 4}


def test_dist_sampler():
    weights = np.array([1, 2, 2, 4, 1])
    pmf = weights / weights.sum()

    sampler = DistSampler(pmf)
    # Verify if the sampled distribution is close to pmf
    indices = sampler.sample(10**6)
    _, counts = np.unique(indices, return_counts=True)
    counts = counts / counts.sum()
    assert np.isclose(pmf, counts, atol=1e-3).all()

    # Verify if it is an infinite sampler
    sampler = DistSampler(pmf, 5)
    sampler_iter = iter(sampler)
    indices = []
    for idx in range(20):
        indices.append(next(sampler_iter))
    indices = np.array(indices)
    #  No need to add any asserts. This was just to check that next(...) works
