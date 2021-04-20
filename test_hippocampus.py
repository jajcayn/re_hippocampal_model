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
        node = self._init_node(HippocampalCA3Node(constant_depression=True))
        self.assertEqual(len(node), 3)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 3)
        self.assertEqual(len(node._sync()), 11)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))

    def test_variable_depression(self):
        node = self._init_node(HippocampalCA3Node(constant_depression=False))
        self.assertEqual(len(node), 4)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 4)
        self.assertEqual(len(node._sync()), 13)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))

    def test_synaptic_facilitation(self):
        node = self._init_node(
            HippocampalCA3Node(constant_depression=True, syn_facilitation=True)
        )
        self.assertEqual(len(node), 4)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 4)
        self.assertEqual(len(node._sync()), 12)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))

    def test_b_p_depression(self):
        node = self._init_node(
            HippocampalCA3Node(constant_depression=False, b_p_depression=True)
        )
        self.assertEqual(len(node), 4)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 4)
        self.assertEqual(len(node._sync()), 13)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))

    def test_depression_and_facilitation(self):
        node = self._init_node(
            HippocampalCA3Node(constant_depression=False, syn_facilitation=True)
        )
        self.assertEqual(len(node), 5)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 5)
        self.assertEqual(len(node._sync()), 14)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))

    def test_all_features(self):
        node = self._init_node(
            HippocampalCA3Node(
                constant_depression=False,
                syn_facilitation=True,
                b_p_depression=True,
            )
        )
        self.assertEqual(len(node), 5)
        self.assertTrue(isinstance(node._derivatives(), list))
        self.assertTrue(isinstance(node._sync(), list))
        self.assertEqual(node.num_state_variables, 5)
        self.assertEqual(len(node._sync()), 14)
        # run
        model = self._multimodel_run(node)
        self.assertTrue(isinstance(model.xr(), xr.DataArray))


if __name__ == "__main__":
    unittest.main()
