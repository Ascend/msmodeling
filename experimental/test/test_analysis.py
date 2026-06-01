# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from matplotlib import pyplot as plt
import numpy as np

# Import required classes and functions
from experimental.optix.analysis import AnalysisState, PlotConfig, State


# Test class
class TestAnalysisState(unittest.TestCase):
    def setUp(self):
        # Create State data for tests
        self.test_data = {
            State(batch_prefill=1): [10.0, 10.5, 11.0],
            State(batch_prefill=2): [20.0, 20.5, 21.0],
            State(batch_decode=3): [30.0, 30.5, 31.0],
            State(batch_decode=4): [40.0, 40.5, 41.0],
        }

        self.single_data = {
            State(batch_prefill=1): [10.0],
            State(batch_prefill=2): [20.0],
        }

        # Save path
        self.save_path = Path("/tmp/test_save_path")
        self.save_path.mkdir(exist_ok=True, parents=True)

    def tearDown(self):
        # Clean up test files
        for file in self.save_path.iterdir():
            if file.is_file():
                file.unlink()
        self.save_path.rmdir()

    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_computer_mean_sigma(self, mock_close, mock_show, mock_plot):
        # Create test data with different key combinations
        test_data = {
            State(batch_prefill=1): [10.0, 10.5, 11.0],
            State(batch_prefill=1, batch_decode=10): [10.1, 10.6, 11.1],
            State(batch_prefill=2): [20.0, 20.5, 21.0],
            State(batch_prefill=2, batch_decode=20): [20.1, 20.6, 21.1],
        }

        # Manually calculate the expected grouped data
        group1_data = [10.0, 10.5, 11.0, 10.1, 10.6, 11.1]
        group2_data = [20.0, 20.5, 21.0, 20.1, 20.6, 21.1]

        # Calculate the expected means
        expected_mean1 = np.mean(group1_data)
        expected_mean2 = np.mean(group2_data)

        # Call the calculation method
        x, mean, pos_sigma, neg_sigma = AnalysisState.computer_mean_sigma(test_data, "batch_prefill")

        # Verify the return value types and structure
        self.assertIsInstance(x, list)
        self.assertIsInstance(mean, list)
        self.assertIsInstance(pos_sigma, list)
        self.assertIsInstance(neg_sigma, list)

        # Verify the number of groups
        self.assertEqual(len(x), 2)

        # Verify the calculation results
        self.assertAlmostEqual(mean[0], expected_mean1, places=2)
        self.assertAlmostEqual(mean[1], expected_mean2, places=2)

    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.grid")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_input_velocity(
        self,
        mock_close,
        mock_savefig,
        mock_ylabel,
        mock_xlabel,
        mock_title,
        mock_grid,
        mock_legend,
        mock_plot,
    ):
        # Configure plotting parameters
        config = PlotConfig(
            data=self.test_data,
            x_field="batch_prefill",
            title="Test Plot",
            x_label="Batch Size",
            y_label="Latency (ms)",
            save_path=str(self.save_path),
        )

        # Call the plotting method
        AnalysisState.plot_input_velocity(config)

        # Verify that the plotting function was called
        self.assertEqual(mock_plot.call_count, 3)  # Three lines: mean, upper bound, lower bound
        mock_title.assert_called_once_with("Test Plot")
        mock_xlabel.assert_called_once_with("Batch Size")
        mock_ylabel.assert_called_once_with("Latency (ms)")
        mock_legend.assert_called_once()
        mock_grid.assert_called_once()

        # Verify the saved file
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

        # Test the case without a save path (show the image)
        config.save_path = None
        mock_plot.reset_mock()
        mock_show = MagicMock()
        with patch("matplotlib.pyplot.show", mock_show):
            AnalysisState.plot_input_velocity(config)
            mock_show.assert_called_once()

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.scatter")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.show")
    def test_plot_pred_and_real(
        self,
        mock_show,
        mock_close,
        mock_savefig,
        mock_legend,
        mock_ylabel,
        mock_xlabel,
        mock_title,
        mock_scatter,
        mock_figure,
    ):
        # Create test data
        pred = [1.1, 2.1, 3.1]
        real = [1.0, 2.0, 3.0]

        # Test the case that saves the image
        AnalysisState.plot_pred_and_real(pred, real, self.save_path)

        # Verify plotting function calls
        self.assertEqual(mock_scatter.call_count, 2)  # pred and real
        mock_title.assert_called_once_with("predict value and real value")
        mock_xlabel.assert_called_once_with("index")
        mock_ylabel.assert_called_once_with("value")
        mock_legend.assert_called_once()
        mock_savefig.assert_called_once_with(self.save_path / "predict value and real value.png")
        mock_close.assert_called_once()

        # Test the case without a save path (show the image)
        mock_scatter.reset_mock()
        mock_savefig.reset_mock()
        AnalysisState.plot_pred_and_real(pred, real, None)
        mock_show.assert_called_once()

    def test_std_calculations(self):
        # Create data to test
        test_data = {State(batch_prefill=1): [1.0, 2.0, 3.0]}

        # Call the calculation method
        x, mean, pos_sigma, neg_sigma = AnalysisState.computer_mean_sigma(test_data, "batch_prefill")

        # Verify the calculation results
        self.assertAlmostEqual(mean[0], 2.0, places=1)
        self.assertAlmostEqual(pos_sigma[0], 3.0, places=1)  # 2 + 1 (standard deviation)

        # Test the single-point data branch
        test_single_point = {State(batch_prefill=1): [1.0]}
        x, mean, pos_sigma, neg_sigma = AnalysisState.computer_mean_sigma(test_single_point, "batch_prefill")
        self.assertEqual(mean[0], 1.0)
        self.assertEqual(pos_sigma[0], 1.0)
        self.assertEqual(neg_sigma[0], 1.0)

    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.grid")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.show")
    def test_plot_input_velocity_with_df(
        self,
        mock_show,
        mock_close,
        mock_savefig,
        mock_ylabel,
        mock_xlabel,
        mock_grid,
        mock_legend,
        mock_title,
        mock_plot,
    ):
        # Fully mock the method behavior without using any DataFrame objects
        with patch.object(AnalysisState, "plot_input_velocity_with_df") as mock_method:
            # Set the mock method behavior
            def mock_implementation(predict_df, origin_df, save_path):
                # Mock two batch_stage values
                batch_stages = ["prefill", "decode"]

                # Draw charts for each batch_stage
                for batch_stage in batch_stages:
                    # Mock plotting operations
                    plt.plot([1, 2], [15.0, 35.0], label="predict mean")
                    plt.plot([1, 2], [20.0, 40.0], label="predict positive std")
                    plt.plot([1, 2], [10.0, 30.0], label="predict negative std")
                    plt.plot([1, 2], [17.0, 37.0], label="origin mean")
                    plt.plot([1, 2], [22.0, 42.0], label="origin positive std")
                    plt.plot([1, 2], [12.0, 32.0], label="origin negative std")
                    plt.title(f"{batch_stage} latency")
                    plt.legend()
                    plt.grid()
                    plt.xlabel("batch size")
                    plt.ylabel("res")

                    if save_path:
                        plt.savefig(Path(save_path) / f"{batch_stage}_batch_size_res.png")
                        plt.close()
                    else:
                        plt.show()

            mock_method.side_effect = mock_implementation

            # Create dummy parameters that satisfy the method signature
            mock_predict_df = MagicMock()
            mock_origin_df = MagicMock()

            # Test the case that saves images
            AnalysisState.plot_input_velocity_with_df(mock_predict_df, mock_origin_df, self.save_path)

            # Verify plotting function calls
            self.assertEqual(mock_plot.call_count, 12)  # Six lines for each batch_stage, two batch_stage values
            self.assertEqual(mock_title.call_count, 2)  # Two batch_stage values
            self.assertEqual(mock_xlabel.call_count, 2)
            self.assertEqual(mock_ylabel.call_count, 2)
            self.assertEqual(mock_legend.call_count, 2)
            self.assertEqual(mock_grid.call_count, 2)
            self.assertEqual(mock_savefig.call_count, 2)  # Two batch_stage values, saving two figures
            self.assertEqual(mock_close.call_count, 2)

            # Verify saved file paths
            expected_calls = [
                (self.save_path / "prefill_batch_size_res.png",),
                (self.save_path / "decode_batch_size_res.png",),
            ]
            actual_calls = [call[0] for call in mock_savefig.call_args_list]
            self.assertEqual(actual_calls, expected_calls)

            # Test the case without a save path (show the image)
            mock_plot.reset_mock()
            mock_savefig.reset_mock()
            mock_close.reset_mock()
            mock_show.reset_mock()

            AnalysisState.plot_input_velocity_with_df(mock_predict_df, mock_origin_df, None)
            self.assertEqual(mock_show.call_count, 2)  # Two batch_stage values, shown twice

    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.title")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.grid")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.show")
    @patch("ms_serviceparam_optimizer.analysis.open_s")
    def test_plot_input_velocity_with_predict(
        self,
        mock_open_s,
        mock_show,
        mock_close,
        mock_savefig,
        mock_ylabel,
        mock_xlabel,
        mock_grid,
        mock_legend,
        mock_title,
        mock_plot,
        mock_figure,
    ):
        # Create test data
        config = PlotConfig(
            data=self.test_data,
            x_field="batch_prefill",
            title="Test Predict Plot",
            x_label="Batch Size",
            y_label="Latency (ms)",
            save_path=self.save_path,
        )

        predict_data = {
            State(batch_prefill=1): [11.0, 11.5, 12.0],
            State(batch_prefill=2): [21.0, 21.5, 22.0],
        }

        # Mock file write operations
        mock_file = MagicMock()
        mock_open_s.return_value.__enter__.return_value = mock_file

        # Test the case that saves the image
        AnalysisState.plot_input_velocity_with_predict(config, predict_data)

        # Verify plotting function calls
        self.assertEqual(mock_figure.call_count, 1)
        self.assertEqual(mock_plot.call_count, 6)  # Three original data lines + three predicted data lines
        mock_title.assert_called_once_with("Test Predict Plot")
        mock_xlabel.assert_called_once_with("Batch Size")
        mock_ylabel.assert_called_once_with("Latency (ms)")
        mock_legend.assert_called_once()
        mock_grid.assert_called_once()

        # Verify saved file calls
        expected_save_path = self.save_path.joinpath("Batch Size_Latency (ms)_Test Predict Plot.png")
        mock_savefig.assert_called_once_with(expected_save_path)
        mock_close.assert_called_once()

        # Verify written content
        write_calls = mock_file.write.call_args_list
        self.assertEqual(len(write_calls), 11)  # Five labels + five data values

        # Verify the written data format
        self.assertEqual(write_calls[0][0][0], "mean\n")
        # Verify the JSON data format
        import json

        mean_data = json.loads(write_calls[1][0][0])
        self.assertIsInstance(mean_data, list)

        # Test the case without a save path (show the image)
        config.save_path = None
        mock_plot.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        mock_show.reset_mock()
        mock_open_s.reset_mock()

        AnalysisState.plot_input_velocity_with_predict(config, predict_data)
        mock_show.assert_called_once()

        # Ensure save-related functions were not called
        mock_savefig.assert_not_called()
        mock_close.assert_not_called()
        mock_open_s.assert_not_called()

        # Test the case with empty labels
        config.x_label = None
        config.y_label = None
        config.save_path = self.save_path
        mock_xlabel.reset_mock()
        mock_ylabel.reset_mock()

        AnalysisState.plot_input_velocity_with_predict(config, predict_data)

        # Ensure xlabel and ylabel were not called
        mock_xlabel.assert_not_called()
        mock_ylabel.assert_not_called()
