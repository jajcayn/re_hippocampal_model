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
}

SYN_DEP_DEFAULT_PARAMS = {
    "tau_d": 250.0,  # ms
    "eta_d": 0.18,
}

SYN_FAC_DEFAULT_PARAMS = {
    "tau_f": 230.0,  # ms
    "eta_f": 0.32,
    "z_max": 1.0,
}

ASWR = "aSWR"
eHC = "SynDep"
zHC = "SynFac"

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
        "node_exc_e",
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
                - coupling_variables["node_exc_e"]
                * coupling_variables["node_exc_inh"]
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

    name = "Basket population CA3"
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

    name = "Base Anti-SWR population CA3"
    label = "BaSWR-HC"
    mass_type = ASWR

    num_state_variables = 1
    num_noise_variables = 1

    coupling_variables = {0: f"r_mean_{ASWR}"}
    required_couplings = [
        "node_aswr_exc",
        "node_aswr_inh",
        "node_aswr_aswr",
        "node_aswr_z",
    ]

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

    name = "Anti-SWR without depression population CA3"
    label = "aSWR-HC-NoDP"

    required_params = ["tau", "F", "k", "threshold", "e"]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or ASWR_NODP_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [A] = self._unwrap_state_vector()
        d_A = (
            -A
            + self._soft_plus(
                (1.0 + coupling_variables["node_aswr_z"])
                * coupling_variables["node_aswr_exc"]
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

    name = "Anti-SWR population CA3"
    label = "aSWR-HC"

    required_couplings = [
        "node_aswr_exc",
        "node_aswr_inh",
        "node_aswr_aswr",
        "node_aswr_z",
        "node_aswr_e",
    ]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or ASWR_DEFAULT_PARAMS, seed=seed)

    def _derivatives(self, coupling_variables):
        [A] = self._unwrap_state_vector()
        d_A = (
            -A
            + self._soft_plus(
                (1.0 + coupling_variables["node_aswr_z"])
                * coupling_variables["node_aswr_exc"]
                - coupling_variables["node_aswr_e"]
                * coupling_variables["node_aswr_inh"]
                - coupling_variables["node_aswr_aswr"]
                + system_input(self.noise_input_idx[0])
            )
        ) / self.params["tau"]

        return [d_A]


class SynapticDepressionHippocampus(NeuralMass):
    """
    Dummy mass that computes synaptic depression variable `e`. This depression
    variable is shared over multiple connections, hence the mass.
    """

    name = "Synaptic depression in CA3"
    label = "e-HC"
    mass_type = eHC

    num_state_variables = 1
    num_noise_variables = 0

    coupling_variables = {0: f"e_{eHC}"}
    required_couplings = ["node_e_inh"]

    state_variable_names = ["e"]

    required_params = ["tau_d", "eta_d"]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or SYN_DEP_DEFAULT_PARAMS, seed=seed)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        self.initial_state = [0.649]

    def _derivatives(self, coupling_variables):
        [e] = self._unwrap_state_vector()

        d_e = (1.0 - e) / self.params["tau_d"] - (
            self.params["eta_d"] * coupling_variables["node_e_inh"] * e
        )

        return [d_e]


class SynapticFacilitationHippocampus(NeuralMass):
    """
    Dummy mass that computes synaptic facilitation variable `z`. Facilitation is
    optionally included on the P->A connection.
    """

    name = "Synaptic facilitation in CA3"
    label = "z-HC"
    mass_type = zHC

    num_state_variables = 1
    num_noise_variables = 0

    coupling_variables = {0: f"z_{zHC}"}
    required_couplings = ["node_z_exc"]

    state_variable_names = ["z"]

    required_params = ["eta_f", "tau_f", "z_max"]

    def __init__(self, params=None, seed=None):
        super().__init__(params=params or SYN_FAC_DEFAULT_PARAMS, seed=seed)

    def _initialize_state_vector(self):
        """
        Initialize state vector.
        """
        self.initial_state = [0.0]

    def _derivatives(self, coupling_variables):
        [z] = self._unwrap_state_vector()

        d_z = (-z / self.params["tau_f"]) + (
            self.params["eta_f"]
            * coupling_variables["node_z_exc"]
            * (self.params["z_max"] - z)
        )

        return [d_z]


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
        "node_exc_e",
        "node_inh_exc",
        "node_inh_inh",
        "node_inh_aswr",
        "node_aswr_exc",
        "node_aswr_inh",
        "node_aswr_aswr",
        "node_aswr_z",
    ]

    def __init__(
        self,
        exc_params=None,
        inh_params=None,
        aswr_params=None,
        connectivity=HIPPOCAMPUS_NODE_DEFAULT_CONNECTIVITY,
        constant_depression=True,
        b_p_depression=False,
        syn_facilitation=False,
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
        :param constant_depression: whether the synaptic depression on the B->A
            connection is treated as a constant parameter [True], or dynamical
            variable [False]
        :type constant_depression: bool
        :param b_p_depression: whether to include additional B->P synaptic
            depression, governed by the same variable as B->A depression
        :type b_p_depression: bool
        :param syn_facilitation: whether to include synaptic facilitation on
            P->A connection
        :type syn_facilitation: bool
        """
        self.output_vars = [f"r_mean_{EXC}", f"r_mean_{INH}", f"r_mean_{ASWR}"]

        masses = []
        # init PYR and B-PV mass
        pyr_mass = PyramidalHippocampalMass(params=exc_params)
        pyr_mass.index = 0
        masses.append(pyr_mass)
        basket_mass = BasketPVHippocampalMass(params=inh_params)
        basket_mass.index = 1
        masses.append(basket_mass)

        if constant_depression:
            aswr_mass = AntiSWRHippocampalMassNoDepression(params=aswr_params)
            aswr_mass.index = 2
            masses.append(aswr_mass)

        else:
            aswr_mass = AntiSWRHippocampalMass(params=aswr_params)
            aswr_mass.index = 2
            masses.append(aswr_mass)
            e_mass = SynapticDepressionHippocampus()
            e_mass.index = 3
            masses.append(e_mass)
            self.sync_variables += ["node_aswr_e", "node_e_inh"]
            self.output_vars += [f"e_{eHC}"]

        if syn_facilitation:
            z_mass = SynapticFacilitationHippocampus()
            z_mass.index = len(masses)
            masses.append(z_mass)
            self.sync_variables += ["node_z_exc"]
            self.output_vars += [f"z_{zHC}"]
            self.z_masses = [z_mass.index]

        self.b_p_dep = b_p_depression
        self.b_a_dep = not constant_depression
        self.syn_fac = syn_facilitation

        ones = np.ones((len(masses), len(masses)))
        ones[:3, :3] = connectivity
        connectivity = ones.copy()

        super().__init__(
            neural_masses=masses,
            local_connectivity=connectivity,
            # within hippocampal node there are no local delays
            local_delays=None,
        )
        # manually set mass indices
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
        # resolve B -> P depression
        if self.b_p_dep:
            b_p_e = self.inputs[0, 3]
        else:
            b_p_e = 1.0
        syncs.append((self.sync_symbols[f"node_exc_e_{self.index}"], b_p_e))

        # resolve P->A facilitation
        if self.syn_fac:
            p_a_z = self.inputs[2, self.z_masses[0]]
            syncs.append(
                (
                    self.sync_symbols[f"node_z_exc_{self.index}"],
                    self.inputs[self.z_masses[0], 0],
                )
            )
        else:
            p_a_z = 0.0
        syncs.append((self.sync_symbols[f"node_aswr_z_{self.index}"], p_a_z))

        if self.b_a_dep:
            syncs.append(
                (
                    self.sync_symbols[f"node_aswr_e_{self.index}"],
                    self.inputs[2, 3],
                )
            )
            syncs.append(
                (
                    self.sync_symbols[f"node_e_inh_{self.index}"],
                    self.inputs[3, 1],
                )
            )

        return syncs
