import pytest
from go_detection.dataloader import InfiniteSampler


def test_infinite_sampler_no_shuffle():
    length = 5

    sampler = InfiniteSampler(length, shuffle=False, repeat=False)
    sampler_iter = iter(sampler)
    indices = [next(sampler_iter) for _ in range(length)]
    assert indices == [0, 1, 2, 3, 4]
    with pytest.raises(StopIteration):
        next(sampler_iter)

    sampler = InfiniteSampler(length, shuffle=False, repeat=True)
    sampler_iter = iter(sampler)
    indices = [next(sampler_iter) for _ in range(length)]
    assert indices == [0, 1, 2, 3, 4]
    assert next(sampler_iter) == 0


def test_infinite_sampler_shuffle():
    length = 5

    sampler = InfiniteSampler(length, shuffle=True, repeat=False)
    sampler_iter = iter(sampler)
    indices = [next(sampler_iter) for _ in range(length)]
    assert set(indices) == {0, 1, 2, 3, 4}
    with pytest.raises(StopIteration):
        next(sampler_iter)

    sampler = InfiniteSampler(length, shuffle=True, repeat=True)
    sampler_iter = iter(sampler)
    indices = [next(sampler_iter) for _ in range(length + 1)]
    assert set(indices) == {0, 1, 2, 3, 4}
