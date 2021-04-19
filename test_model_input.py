"""
Unit tests for model input.
"""

import unittest

import numpy as np
from chspy import CubicHermiteSpline

from model_input import (
    PeriodicRandomSquareInput,
    PoissonNoiseWithExpKernel,
    ZeroMeanConcatenatedInput,
)


class TestPoissonNoiseWithExpKernel(unittest.TestCase):

    DURATION = 2000
    DT = 0.1
    SEED = 42

    def test_numba(self):
        pois = PoissonNoiseWithExpKernel(
            freq=100.0, amp=10.0, tau_syn=5.0
        ).as_array(self.DURATION, self.DT)
        self.assertTrue(isinstance(pois, np.ndarray))
        self.assertTupleEqual(pois.shape, (int(self.DURATION / self.DT), 1))

    def test_jitcdde(self):
        pois = PoissonNoiseWithExpKernel(
            freq=100.0, amp=10.0, tau_syn=5.0
        ).as_cubic_splines(self.DURATION, self.DT)
        self.assertTrue(isinstance(pois, CubicHermiteSpline))

    def test_equiv(self):
        array = PoissonNoiseWithExpKernel(
            freq=100.0, amp=10.0, tau_syn=5.0, seed=self.SEED
        ).as_array(self.DURATION, self.DT)
        splines = PoissonNoiseWithExpKernel(
            freq=100.0, amp=10.0, tau_syn=5.0, seed=self.SEED
        ).as_cubic_splines(self.DURATION, self.DT)
        splines_array = splines.get_state(
            np.arange(self.DT, self.DURATION + self.DT, self.DT)
        )
        np.testing.assert_allclose(array, splines_array)


class TestZeroMeanConcatenatedInput(unittest.TestCase):

    DURATION = 2000
    DT = 0.1
    SEED = 42

    def setUp(self):
        self.pois1 = PoissonNoiseWithExpKernel(
            freq=100.0, amp=10.0, tau_syn=5.0, seed=self.SEED
        )
        self.pois2 = PoissonNoiseWithExpKernel(
            freq=200.0, amp=-8.0, tau_syn=2.5, seed=self.SEED
        )

    def test_numba(self):
        concat = ZeroMeanConcatenatedInput([self.pois1, self.pois2]).as_array(
            self.DURATION, self.DT
        )
        self.assertTrue(isinstance(concat, np.ndarray))
        self.assertTupleEqual(concat.shape, (int(self.DURATION / self.DT), 1))
        np.testing.assert_almost_equal(np.mean(concat), 0.0)

    def test_jitcdde(self):
        concat = ZeroMeanConcatenatedInput(
            [self.pois1, self.pois2]
        ).as_cubic_splines(self.DURATION, self.DT)
        self.assertTrue(isinstance(concat, CubicHermiteSpline))
        concat_array = concat.get_state(
            np.arange(self.DT, self.DURATION + self.DT, self.DT)
        )
        np.testing.assert_almost_equal(np.mean(concat_array), 0.0)


class TestPeriodicRandomSquareInput(unittest.TestCase):

    DURATION = 2000
    DT = 0.1
    SEED = 42

    def test_numba(self):
        pois = PeriodicRandomSquareInput(
            step_size=150,
            step_duration=10,
            step_period=500,
            max_jitter=80,
            seed=self.SEED,
        ).as_array(self.DURATION, self.DT)
        self.assertTrue(isinstance(pois, np.ndarray))
        self.assertTupleEqual(pois.shape, (int(self.DURATION / self.DT), 1))

    def test_jitcdde(self):
        pois = PeriodicRandomSquareInput(
            step_size=150,
            step_duration=10,
            step_period=500,
            max_jitter=80,
            seed=self.SEED,
        ).as_cubic_splines(self.DURATION, self.DT)
        self.assertTrue(isinstance(pois, CubicHermiteSpline))

    def test_equiv(self):
        array = PeriodicRandomSquareInput(
            step_size=150,
            step_duration=10,
            step_period=500,
            max_jitter=80,
            seed=self.SEED,
        ).as_array(self.DURATION, self.DT)
        splines = PeriodicRandomSquareInput(
            step_size=150,
            step_duration=10,
            step_period=500,
            max_jitter=80,
            seed=self.SEED,
        ).as_cubic_splines(self.DURATION, self.DT)
        splines_array = splines.get_state(
            np.arange(self.DT, self.DURATION + self.DT, self.DT)
        )
        np.testing.assert_allclose(array, splines_array)
