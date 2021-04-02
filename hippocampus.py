"""
Hippocampus CA3 rate model able to generate sharp wave ripples.
"""

from itertools import product

import numpy as np
import symengine as se
from jitcdde import input as system_input
from neurolib.models.multimodel.builder.base.constants import EXC, INH
from neurolib.models.multimodel.builder.base.network import (
    SingleCouplingExcitatoryInhibitoryNode,
)
from neurolib.models.multimodel.builder.base.neural_mass import NeuralMass

from model_input import PoissonNoiseWithExpKernel, ZeroMeanConcatenatedInput

PYR_DEFAULT_PARAMS = {
    "tau": 3.0,  # ms
    "F": 0.001,  # kHz
    "k": 0.47,  # 1/pA
    "threshold": 131.66,  # pA
}

BPV_DEFAULT_PARAMS = {
    "tau": 2.0,  # ms
    "F": 0.001,  # kHz
    "k": 0.41,  # 1/pA
    "threshold": 131.96,  # pA
}

ASWR_NODP_DEFAULT_PARAMS = {
    "tau": 6.0,  # ms
    "F": 0.001,  # kHz
    "k": 0.48,  # 1/pA
    "threshold": 131.09,  # pA
    "e": 0.404,
}

ASWR_DEFAULT_PARAMS = {
    "tau": 6.0,  # ms
    "F": 0.001,  # kHz
    "k": 0.48,  # 1/pA
    "threshold": 131.09,  # pA
    "tau_d": 250.0,  # ms
    "eta_d": 0.18,
}

ASWR_SF_DEFAULT_PARAMS = {
    **ASWR_DEFAULT_PARAMS,
    "tau_f": 230.0,  # ms
    "eta_f": 0.32,
    "z_max": 1.0,
}

ASWR = "aSWR"

# matrix as [to, from], masses as (EXC, INH, aSWR)
# scale by 1e3 for kHz -> Hz
HIPPOCAMPUS_NODE_DEFAULT_CONNECTIVITY = 1e3 * np.array(
    [[1.72, 1.24, 12.6], [8.86, 3.24, 13.44], [1.72, 5.67, 8.40]]
)


class HippocampalCA3Mass(NeuralMass):
    """
    Base for hippocampal CA3 rate populations.

    Reference:
        Evangelista, R., Cano, G., Cooper, C., Schmitz, D., Maier, N., & Kempter
        , R. (2020). Generation of sharp wave-ripple events by disinhibition.
        Journal of Neuroscience, 40(41), 7811-7836.
    """

    name = "Hippocampal mass"
    label = "HC"

    def _soft_plus(self, activity):
        return self.params["F"] * se.log(
            1.0
            + se.exp(self.params["k"] * (activity + self.params["threshold"]))
        )

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        np.random.seed(self.seed)
        self.initial_state = [0.01 * np.random.uniform(0, 1)]


class PyramidalHippocampalMass(HippocampalCA3Mass):
    """
    Excitatory mass representing pyramidal cells in CA3 region of hippocampus.
    """

    name = "Pyramidal population CA3"
    label = "PYR-HC"
    mass_type = EXC

    num_state_variables = 1
    num_noise_variables = 1
    coupling_variables = {0: f"r_mean_{EXC}"}
    required_couplings = [
        "node_exc_exc",
        "node_exc_inh",
        "node_exc_aswr",
        "network_exc_exc",
    ]
    state_variable_names = ["r_mean"]
    required_params = ["tau", "F", "k", "threshold"]

    _noise_input = [
        ZeroMeanConcatenatedInput(
            [
                PoissonNoiseWithExpKernel(freq=159.08, amp=1.3125, tau_syn=2.0),
                PoissonNoiseWithExpKernel(freq=89.1, amp=-1.53125, tau_syn=1.5),
                PoissonNoiseWithExpKernel(freq=376.8, amp=-13.125, tau_syn=4.0),
            ]
        )
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or PYR_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [P] = self._unwrap_state_vector()
        d_P = (
            -P
            + self._soft_plus(
                coupling_variables["node_exc_exc"]
                + coupling_variables["network_exc_exc"]
                - coupling_variables["node_exc_inh"]
                - coupling_variables["node_exc_aswr"]
                + system_input(self.noise_input_idx[0])
            )
        ) / self.params["tau"]

        return [d_P]


class BasketPVHippocampalMass(HippocampalCA3Mass):
    """
    Inhibitory mass representing parvalbumin-positive (PV1) basket cells in CA3
    region of the hippocampus.
    """

    name = "Basket ppopulation CA3"
    label = "BPV-HC"
    mass_type = INH

    num_state_variables = 1
    num_noise_variables = 1
    coupling_variables = {0: f"r_mean_{INH}"}
    required_couplings = ["node_inh_exc", "node_inh_inh", "node_inh_aswr"]
    state_variable_names = ["r_mean"]
    required_params = ["tau", "F", "k", "threshold"]

    _noise_input = [
        ZeroMeanConcatenatedInput(
            [
                PoissonNoiseWithExpKernel(freq=3181.6, amp=0.3375, tau_syn=2.0),
                PoissonNoiseWithExpKernel(freq=35.64, amp=-10.0, tau_syn=1.5),
                PoissonNoiseWithExpKernel(freq=376.8, amp=-14.0, tau_syn=4.0),
            ]
        )
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or BPV_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [B] = self._unwrap_state_vector()
        d_B = (
            -B
            + self._soft_plus(
                coupling_variables["node_inh_exc"]
                - coupling_variables["node_inh_inh"]
                - coupling_variables["node_inh_aswr"]
                + system_input(self.noise_input_idx[0])
            )
        ) / self.params["tau"]

        return [d_B]


class BaseAntiSWRHippocampalMass(HippocampalCA3Mass):
    """
    Base for all versions of AntiSWR mass - inhibitory mass representing
    unidentified class of anti-SWR interneurons in CA3 region of the
    hippocampus.
    """

    name = "Base Anti-SWR ppopulation CA3"
    label = "BaSWR-HC"
    mass_type = ASWR

    num_state_variables = 1
    num_noise_variables = 1

    coupling_variables = {0: f"r_mean_{ASWR}"}
    required_couplings = ["node_aswr_exc", "node_aswr_inh", "node_aswr_aswr"]

    state_variable_names = ["r_mean"]

    required_params = ["tau", "F", "k", "threshold"]

    _noise_input = [
        ZeroMeanConcatenatedInput(
            [
                PoissonNoiseWithExpKernel(freq=159.08, amp=1.3125, tau_syn=2.0),
                PoissonNoiseWithExpKernel(freq=35.64, amp=-17.5, tau_syn=1.5),
                PoissonNoiseWithExpKernel(freq=376.8, amp=-8.75, tau_syn=4.0),
            ]
        )
    ]


class AntiSWRHippocampalMassNoDepression(BaseAntiSWRHippocampalMass):
    """
    Basic anti-SWR mass without synaptic depression, which is treated as a
    constant.
    """

    name = "Anti-SWR without depression ppopulation CA3"
    label = "aSWR-HC-NoDP"

    required_params = ["tau", "F", "k", "threshold", "e"]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or ASWR_NODP_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [A] = self._unwrap_state_vector()
        d_A = (
            -A
            + self._soft_plus(
                coupling_variables["node_aswr_exc"]
                - self.params["e"] * coupling_variables["node_aswr_inh"]
                - coupling_variables["node_aswr_aswr"]
                + system_input(self.noise_input_idx[0])
            )
        ) / self.params["tau"]

        return [d_A]


class AntiSWRHippocampalMass(BaseAntiSWRHippocampalMass):
    """
    Anti-SWR mass with synaptic depression slow variable `e`.
    """

    name = "Anti-SWR ppopulation CA3"
    label = "aSWR-HC"

    num_state_variables = 2
    state_variable_names = ["r_mean", "e"]
    required_params = ["tau", "F", "k", "threshold", "tau_d", "eta_d"]
    required_couplings = [
        "node_aswr_exc",
        "node_aswr_inh",
        "node_aswr_aswr",
        "node_aswr_inh_syn_dep",
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or ASWR_DEFAULT_PARAMS, seed=seed)

    def _initialize_state_vector(self):
        super()._initialize_state_vector()
        self.initial_state += [0.649]

    def _derivatives(self, coupling_variables):
        [A, e] = self._unwrap_state_vector()
        d_A = (
            -A
            + self._soft_plus(
                coupling_variables["node_aswr_exc"]
                - e * coupling_variables["node_aswr_inh"]
                - coupling_variables["node_aswr_aswr"]
                + system_input(self.noise_input_idx[0])
            )
        ) / self.params["tau"]

        d_e = (1.0 - e) / self.params["tau_d"] - (
            self.params["eta_d"]
            * coupling_variables["node_aswr_inh_syn_dep"]
            * e
        )

        return [d_A, d_e]


class AntiSWRHippocampalMassWithFacilitation(BaseAntiSWRHippocampalMass):
    """
    Anti-SWR mass with additional short-term plasticity mechanism in the form of
    synaptic facilitation on the P -> A connection.
    """

    name = "Anti-SWR ppopulation CA3 with facilitation"
    label = "aSWR-HC-SF"

    num_state_variables = 3
    state_variable_names = ["r_mean", "e", "z"]
    required_params = [
        "tau",
        "F",
        "k",
        "threshold",
        "tau_d",
        "eta_d",
        "eta_f",
        "tau_f",
        "z_max",
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or ASWR_SF_DEFAULT_PARAMS, seed=seed)

    def _initialize_state_vector(self):
        super()._initialize_state_vector()
        self.initial_state += [0.649, 0.0]

    def _derivatives(self, coupling_variables):
        [A, e, z] = self._unwrap_state_vector()
        d_A = (
            -A
            + self._soft_plus(
                (1.0 + z) * coupling_variables["node_aswr_exc"]
                - e * coupling_variables["node_aswr_inh"]
                - coupling_variables["node_aswr_aswr"]
                + system_input(self.noise_input_idx[0])
            )
        ) / self.params["tau"]

        d_e = (1.0 - e) / self.params["tau_d"] - (
            self.params["eta_d"] * coupling_variables["node_aswr_inh"] * e
        )

        d_z = (-z / self.params["tau_f"]) + (
            self.params["eta_f"]
            * coupling_variables["node_aswr_exc"]
            * (self.params["z_max"] - z)
        )

        return [d_A, d_e, d_z]


class HippocampalCA3Node(SingleCouplingExcitatoryInhibitoryNode):
    """
    Hippocampal CA3 region mass model with 1 pyramidal, 1 basket parvalbumin-
    positive, and 1 anti-SWR interneurons mass due to Evangelista et al.
    """

    name = "Hippocampal mass model node"
    label = "HCnode"

    default_network_coupling = {"network_exc_exc": 0.0}
    default_output = f"r_mean_{EXC}"
    output_vars = [f"r_mean_{EXC}", f"r_mean_{INH}", f"r_mean_{ASWR}"]

    sync_variables = [
        "node_exc_exc",
        "node_exc_aswr",
        "node_exc_inh",
        "node_inh_exc",
        "node_inh_inh",
        "node_inh_aswr",
        "node_aswr_exc",
        "node_aswr_inh",
        "node_aswr_aswr",
    ]

    def __init__(
        self,
        exc_params=None,
        inh_params=None,
        aswr_params=None,
        connectivity=HIPPOCAMPUS_NODE_DEFAULT_CONNECTIVITY,
        aswr_mass_type="constant_depression",
    ):
        """
        :param exc_params: parameters for the excitatory (PYR) mass
        :type exc_params: dict|None
        :param inh_params: parameters for the inhibitory (INH) mass
        :type inh_params: dict|None
        :param aswr_params: parameters for the anti-SWR (aSWR) mass
        :type aswr_params: dict|None
        :param connectivity: local connectivity matrix
        :type connectivity: np.ndarray
        :param aswr_mass_type: type of anti-SWR interneurons mass, can be:
            - constant_depression: synaptic depression variable treated as a
                constant
            - variable_depression: synaptic depression BPV -> aSWR treated as a
                slow dynamical variable
            - synaptic_facilitation: depression as variable plus additional
                synaptic faciliation of PYR -> aSWR treated as a dynamical
                variable
        :type aswr_mass_type: str
        """
        pyr_mass = PyramidalHippocampalMass(params=exc_params)
        pyr_mass.index = 0
        basket_mass = BasketPVHippocampalMass(params=inh_params)
        basket_mass.index = 1
        if aswr_mass_type == "constant_depression":
            aswr_mass = AntiSWRHippocampalMassNoDepression(params=aswr_params)
        elif aswr_mass_type == "variable_depression":
            aswr_mass = AntiSWRHippocampalMass(params=aswr_params)
            self.sync_variables.append("node_aswr_inh_syn_dep")
            self.output_vars += [f"e_{ASWR}"]
        elif aswr_mass_type == "synaptic_facilitation":
            aswr_mass = AntiSWRHippocampalMassWithFacilitation(
                params=aswr_params
            )
            self.output_vars += [f"e_{ASWR}", f"z_{ASWR}"]
        else:
            raise ValueError(f"Unknown type of aSWR mass: {aswr_mass_type}")
        self.aswr_mass_type = aswr_mass_type
        aswr_mass.index = 2
        super().__init__(
            neural_masses=[pyr_mass, basket_mass, aswr_mass],
            local_connectivity=connectivity,
            # within hippocampal node there are no local delays
            local_delays=None,
        )
        # manualy set mass indices
        self.excitatory_masses = [0]
        self.inhibitory_masses = [1]
        self.aswr_masses = [2]

    def _sync(self):
        # connectivity as [to, from]
        connectivity = self.connectivity * self.inputs
        syncs = []
        for (mass_type1, indices1), (mass_type2, indices2) in product(
            zip(
                ["exc", "inh", "aswr"],
                [
                    self.excitatory_masses,
                    self.inhibitory_masses,
                    self.aswr_masses,
                ],
            ),
            repeat=2,
        ):
            syncs.append(
                (
                    self.sync_symbols[
                        f"node_{mass_type1}_{mass_type2}_{self.index}"
                    ],
                    sum(
                        [
                            connectivity[row, col]
                            for row in indices1
                            for col in indices2
                        ]
                    ),
                )
            )
        if self.aswr_mass_type == "variable_depression":
            # manually add synaptic depression on B -> A connection
            syncs.append(
                (
                    self.sync_symbols[f"node_aswr_inh_syn_dep_{self.index}"],
                    self.inputs[2, 1],
                )
            )

        return syncs
