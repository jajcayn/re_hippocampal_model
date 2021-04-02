"""
Inputs for the hippocampal CA3 model.
"""

import numpy as np
from chspy import CubicHermiteSpline
from neurolib.utils.stimulus import ConcatenatedInput, ModelInput, StimulusInput


class PoissonNoiseWithExpKernel(ModelInput):
    """
    Poissoin noise with exponential kernel.
    """

    def __init__(self, freq, amp, tau_syn, seed=None):
        self.freq = freq
        self.amp = amp
        self.tau_syn = tau_syn
        super().__init__(num_iid=1, seed=seed)

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        x = np.zeros((self.times.shape[0]))
        return self.poisson_exp_kernel(
            x, self.times, self.freq, self.amp, self.tau_syn
        )[:, np.newaxis]

    @staticmethod
    def poisson_exp_kernel(x, times, freq, amp, tau_syn):
        """
        Generation of Poisson spiking noise, convoluted with exponential kernel.
        """
        total_spikes = int(freq * (times[-1] - times[0]) / 1000.0)  # in seconds
        spike_indices = np.random.choice(len(x), total_spikes, replace=True)
        x[spike_indices] = 1.0
        time_spike_end = -tau_syn * np.log(0.001)
        arg_spike_end = np.argmin(np.abs(times - time_spike_end))
        spike_kernel = np.exp(-times[:arg_spike_end] / tau_syn)
        x = np.convolve(x, spike_kernel, mode="same")
        return x * amp


class ZeroMeanConcatenatedInput(ConcatenatedInput):
    """
    Concatenated input, i.e. sum with subtraction of the mean.
    """

    def as_array(self, duration, dt):
        """
        Return as array for numba backend in neurolib.
        """
        sum_ = super().as_array(duration, dt)
        return sum_ - np.mean(sum_, axis=0)

    def as_cubic_splines(self, duration, dt):
        """
        Return as CubicHermiteSpline for jitcdde backend in neurolib.
        """
        self.stim_start = None
        self.stim_end = None
        self._get_times(duration, dt)
        return CubicHermiteSpline.from_data(
            self.times, self.as_array(duration, dt)
        )


class PeriodicRandomSquareInput(StimulusInput):
    """
    Square pulse with random timing.
    """

    def __init__(self, step_size, step_duration, step_period, seed=None):
        """
        :param step_size: size of the stimulus
        :type step_size: float
        :param step_duration: duration of the single pulse, in miliseconds
        :type step_duration: float
        :param step_period: period of the square stimilus, in miliseconds
        :type step_period: float
        """
        self.step_size = step_size
        self.step_duration = step_duration
        self.step_period = step_period
        super().__init__(
            stim_start=None,
            stim_end=None,
            num_iid=1,
            seed=seed,
        )

    def generate_input(self, duration, dt):
        self._get_times(duration=duration, dt=dt)
        # get periodic stimuli
        stim_times = np.arange(
            self.step_period, self.times[-1], self.step_period
        )
        # get random delay in [0, 90] ms
        jitters = np.random.randint(91, size=len(stim_times))
        stim_times += jitters
