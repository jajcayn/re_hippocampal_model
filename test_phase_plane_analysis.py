"""
Unit tests for phase plane analysis helper.
"""

import unittest
import sympy as sp

from phase_plane_analysis import PhasePlaneAnalysis
from hippocampus import HippocampalCA3Node


class TestPhasePlaneAnalysis(unittest.TestCase):
    def _init_node(self, node):
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def test_phase_plane(self):
        node = self._init_node(
            HippocampalCA3Node(aswr_mass_type="constant_depression")
        )
        phase_plane = PhasePlaneAnalysis(node)
        self.assertTrue(isinstance(phase_plane.ode_system, sp.Matrix))
        self.assertTrue(isinstance(phase_plane.jacobian, sp.Matrix))
        self.assertTrue(isinstance(phase_plane.lambdify_odes(), list))
        for ode in phase_plane.lambdify_odes():
            self.assertTrue(callable(ode))


if __name__ == "__main__":
    unittest.main()
