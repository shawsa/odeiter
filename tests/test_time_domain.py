from collections.abc import Iterable
import numpy as np
from odeiter import (
    TimeDomain,
    TimeDomain_Start_Stop_MaxSpacing,
)
from odeiter.time_domain import (
    TimeDomain_Start_Stop_Steps,
    Ray,
    TimeRay,
)
import pytest


def test_init_array():
    time = TimeDomain(0, 1, 4)
    assert np.all(time.array == np.array([0, 1, 2, 3, 4], dtype=float))


def test_iter():
    time = TimeDomain(0, 1, 4)
    true_times = np.arange(5, dtype=float)
    for test_time, true_time in zip(time, true_times):
        assert type(test_time) is type(true_time)
        assert test_time == true_time


def test_start_stop_init():
    time = TimeDomain_Start_Stop_MaxSpacing(0, 4, 1)
    assert time.start == 0
    assert time.steps == 4
    assert time.spacing == 1

    time = TimeDomain_Start_Stop_MaxSpacing(0, 4, 1.1)
    assert time.start == 0
    assert time.steps == 4
    assert time.spacing == 1

    time = TimeDomain_Start_Stop_MaxSpacing(0, 4, 0.9)
    assert time.start == 0
    assert time.steps == 5
    assert time.spacing == 0.8


def test_start_stop_Steps():
    time = TimeDomain_Start_Stop_Steps(0, 5, 10)
    assert time.start == 0
    assert time.steps == 10
    assert time.spacing == 0.5


def test_ray_init():
    time = Ray(0, 0.5)
    assert time.start == 0
    assert time.step == 0.5


def test_ray_getitem_error():
    time = Ray(0, 0.5)
    with pytest.raises(ValueError):
        assert time["0"]


def test_ray_getitem_int():
    time = Ray(0, 0.5)
    assert time[3] == 1.5


def test_ray_getitem_slice():
    time = Ray(0, 0.5)
    my_slice = time[::2]
    assert my_slice.start == 0
    assert my_slice.step == 1.0

    my_slice = time[3:]
    assert my_slice.start == 1.5
    assert my_slice.step == 0.5

    my_slice = time[3:5:3]
    assert my_slice.start == 1.5
    assert my_slice.step == 1.5


def test_ray_iter():
    time = Ray(0, 0.2)
    true_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for my_time, true_time in zip(time, true_times):
        assert true_time == pytest.approx(my_time)


def test_ray_repr():
    time = Ray(0.0, 1.0)
    assert str(time) == "Ray(start=0.0, step=1.0)"


def test_time_ray_init():
    time = TimeRay(0, 1)
    assert time.start == 0
    assert time.spacing == 1


def test_time_ray_array():
    time = TimeRay(0, 1)
    array = time.array
    assert isinstance(array, Ray)
    assert array.start == 0
    assert array.step == 1


def test_time_ray_iter():
    time = TimeRay(0, 1)
    assert isinstance(iter(time), Iterable)
    my_iter = iter(time)
    assert 0.0 == next(my_iter)
    assert 1.0 == next(my_iter)
    assert 2.0 == next(my_iter)
