"""
Unit tests for hippocampal model.
"""

import unittest

import xarray as xr
from neurolib.models.multimodel import MultiModel

from hippocampus import HippocampalCA3Node

DURATION = 2000
DT = 0.05
BACKEND = "numba"


class TestHippocampalCA3Node(unittest.TestCase):
    def _init_node(self, node):
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        return node

    def _multimodel_run(self, node):
        model = MultiModel(node)
        model.params["duration"] = DURATION
        model.params["dt"] = DT
        model.params["backend"] = BACKEND
        model.run()
        return model

    def test_constant_depression(self):
        node = self._init_node(
            HippocampalCA3Node(aswr_mass_type="constant_depression")
        )
        self.assertEqual(len(node), 3)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 3)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))

    def test_variable_depression(self):
        node = self._init_node(
            HippocampalCA3Node(aswr_mass_type="variable_depression")
        )
        self.assertEqual(len(node), 4)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 4)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))

    def test_synaptic_facilitation(self):
        node = self._init_node(
            HippocampalCA3Node(aswr_mass_type="synaptic_facilitation")
        )
        self.assertEqual(len(node), 4)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 5)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))

    def test_b_p_depression(self):
        node = self._init_node(
            HippocampalCA3Node(
                aswr_mass_type="variable_depression", b_p_depression=True
            )
        )
        self.assertEqual(len(node), 4)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 4)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))


if __name__ == "__main__":
    unittest.main()
