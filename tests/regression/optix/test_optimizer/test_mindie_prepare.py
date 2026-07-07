"""Regression tests for PSOOptimizer.mindie_prepare."""

from __future__ import annotations

import unittest
from math import inf
from unittest.mock import MagicMock, patch

from optix.config.config import OptimizerConfigField
from tests.regression.optix.test_optimizer.test_pso_optimizer import _make_pso_optimizer


def _optimizer_with_max_batch_field(**field_overrides):
    field = OptimizerConfigField(name="max_batch_size", min=1, max=1000, dtype="int", value=100)
    for key, value in field_overrides.items():
        setattr(field, key, value)
    return _make_pso_optimizer(
        target_field=(field,),
        scheduler=MagicMock(error_info=None),
    )


class TestMindiePrepare(unittest.TestCase):
    @patch("optix.optimizer.optimizer.is_mindie", return_value=True)
    @patch("optix.config.config.get_settings")
    def test_skips_when_mc_is_none(self, mock_settings, mock_is_mindie):
        mock_settings.return_value.theory_guided_enable = True
        opt = _optimizer_with_max_batch_field()
        opt.mindie_prepare(None)
        self.assertEqual(opt.target_field[0].max, 1000)

    @patch("optix.config.config.get_settings")
    def test_skips_when_theory_guided_disabled(self, mock_settings):
        mock_settings.return_value.theory_guided_enable = False
        opt = _optimizer_with_max_batch_field()
        mc = MagicMock()
        opt.mindie_prepare(mc)
        mc.get_max_batch_size_bound.assert_not_called()

    @patch("optix.config.config.get_settings")
    def test_inf_upper_bound_does_not_overflow(self, mock_settings):
        mock_settings.return_value.theory_guided_enable = True
        mock_settings.return_value.scaling_coefficient = 0.5
        opt = _optimizer_with_max_batch_field(max=1000)
        mc = MagicMock()
        mc.get_max_batch_size_bound.return_value = (10, inf)
        opt.mindie_prepare(mc)
        self.assertEqual(opt.target_field[0].max, 1000)

    @patch("optix.config.config.get_settings")
    def test_scaling_coefficient_tightens_max_batch_size(self, mock_settings):
        mock_settings.return_value.theory_guided_enable = True
        mock_settings.return_value.scaling_coefficient = 0.5
        opt = _optimizer_with_max_batch_field(min=1, max=1000)
        mc = MagicMock()
        mc.get_max_batch_size_bound.return_value = (10, 200)
        opt.mindie_prepare(mc)
        self.assertEqual(opt.target_field[0].max, 100)

    @patch("optix.config.config.get_settings")
    def test_invalid_bounds_leave_target_field_unchanged(self, mock_settings):
        mock_settings.return_value.theory_guided_enable = True
        mock_settings.return_value.scaling_coefficient = 1.0
        opt = _optimizer_with_max_batch_field(min=5, max=500)
        mc = MagicMock()
        mc.get_max_batch_size_bound.return_value = (100, 50)
        opt.mindie_prepare(mc)
        self.assertEqual(opt.target_field[0].max, 500)

    @patch("optix.config.config.get_settings")
    def test_non_positive_bounds_are_ignored(self, mock_settings):
        mock_settings.return_value.theory_guided_enable = True
        mock_settings.return_value.scaling_coefficient = 1.0
        opt = _optimizer_with_max_batch_field(min=5, max=500)
        mc = MagicMock()
        mc.get_max_batch_size_bound.return_value = (0, 100)
        opt.mindie_prepare(mc)
        self.assertEqual(opt.target_field[0].max, 500)
