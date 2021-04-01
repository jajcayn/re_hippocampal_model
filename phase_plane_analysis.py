"""
Simple phase plane tools - mainly for nullclines.
"""

import logging

import symengine as se
import sympy as sp


class PhasePlaneAnalysis:
    """
    Phase plane analysis tools.
    """

    NEEDED_ATTRIBUTES = [
        "_derivatives",
        "_sync",
        "state_variable_names",
        "num_noise_variables",
    ]
    CURRENT_Y = "current_y({idx})"
    SYSTEM_INPUT = (
        "past_y(-external_input + t, {prefix}input_base_n, "
        "anchors(-external_input + t))"
    )

    def __init__(self, system):
        """
        :param system: system to analyse
        :type system: `models.base.network.Node`|`models.base.network.Network`|
            any initialised class with all necessary attributes and symbolic
            derivatives
        """
        assert all(hasattr(system, attr) for attr in self.NEEDED_ATTRIBUTES)
        logging.info("Substituting helpers...")
        derivatives = self._substitute_helpers(
            derivatives=system._derivatives(), helpers=system._sync()
        )
        self.state_vars = self._unwrap_state_variable_names(
            system.state_variable_names
        )
        derivatives = self._substitute_variable_names(
            derivatives, self.state_vars
        )
        derivatives = self._nullify_system_input(
            derivatives, system.num_noise_variables
        )
        # ode system as symbolic matrix
        self._ode_system = sp.Matrix(derivatives)

    def _unwrap_state_variable_names(self, state_var_names):
        if len(state_var_names) == 1:
            return state_var_names[0]
        else:
            unwrapped = []
            for idx, node_vars in enumerate(state_var_names):
                unwrapped += [f"{var}_{idx}" for var in node_vars]
            return unwrapped

    def _substitute_variable_names(self, derivatives, state_variable_names):
        assert len(derivatives) == len(state_variable_names)
        substitutions = {
            self.CURRENT_Y.format(idx=idx): var_name
            for idx, var_name in enumerate(state_variable_names)
        }
        return [derivative.subs(substitutions) for derivative in derivatives]

    def _nullify_system_input(self, derivatives, num_noise_variables):
        substitutions = {}
        for idx in range(num_noise_variables):
            prefix = "" if idx == 0 else f"{idx} + "
            substitutions[self.SYSTEM_INPUT.format(prefix=prefix)] = 0.0
        return [derivative.subs(substitutions) for derivative in derivatives]

    def _substitute_helpers(self, derivatives, helpers):
        """
        Substitute helpers (usually used for coupling) to derivatives.

        :param derivatives: list of symbolic expressions for derivatives
        :type derivatives: list
        :param helpers: list of tuples as (helper name, symbolic expression) for
            helpers
        :type helpers: list[tuple]
        """
        sympified_helpers = [
            (se.sympify(helper[0]), se.sympify(helper[1])) for helper in helpers
        ]
        sympified_derivatives = [
            se.sympify(derivative) for derivative in derivatives
        ]
        substitutions = {helper[0]: helper[1] for helper in sympified_helpers}
        return [
            derivative.subs(substitutions)
            for derivative in sympified_derivatives
        ]

    @property
    def state_var_symbols(self):
        """
        Return state variables as sympy symbols.

        :return: sympy symbols for state variables
        :rtype: tuple[`sp.Symbol`]
        """
        return sp.symbols(",".join(self.state_vars))

    @property
    def ode_system(self):
        """
        Return read-only ODE system.

        :return: symbolic derivatives, i.e. the ODE system
        :rtype: `sp.Matrix`
        """
        return self._ode_system

    @property
    def jacobian(self):
        """
        Return Jacobian of the system.

        :return: symbolic Jacobian of the system
        :rtype: `sp.Matrix`
        """
        return self.ode_system.jacobian(self.state_vars)

    def lambdify_odes(self):
        """
        Return lambdified ODEs. This means, you can call them as a function of
        state variables.

        :return: list of lambdified functions
        :rtype: list[callable]
        """
        return [
            sp.lambdify(self.state_var_symbols, derivative)
            for derivative in self.ode_system
        ]
