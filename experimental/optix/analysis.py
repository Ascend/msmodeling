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

import copy
import json
import statistics
from pathlib import Path
from statistics import mean, stdev
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from loguru import logger
from msguard.security import open_s
from .common import State


@dataclass
class PlotConfig:
    data: Dict[State, List]
    x_field: str
    title: str
    x_label: str
    y_label: str
    save_path: str = None


class AnalysisState:
    @staticmethod
    def computer_mean_sigma(
        data: Dict[State, List],
        x_field: str,
    ):
        # Merge decode-only and prefill groups
        res = {}
        tmp_data = copy.deepcopy(data)
        for k, v in tmp_data.items():
            if k.batch_prefill:
                s = State(batch_prefill=k.batch_prefill)
            else:
                s = State(batch_decode=k.batch_decode)
            if s in res:
                res[s].extend(v)
            else:
                res[s] = v
        # Compute mean and sigma
        _x = []
        _mean = []
        _positive_sigma = []
        _negative_sigma = []
        for k in sorted(res.keys(), key=lambda x: getattr(x, x_field)):
            v = res.get(k, [])
            if len(v) == 1:
                _x.append(getattr(k, x_field))
                _mean.append(v[0])
                _positive_sigma.append(v[0])
                _negative_sigma.append(v[0])
                continue
            _x.append(getattr(k, x_field))
            _mean.append(mean(v))
            try:
                _sigma = stdev(v)
            except AssertionError:
                try:
                    _sigma = np.std(v)
                except Exception:
                    logger.warning('Failed stdev', v)
                    _sigma = 0 if not v else statistics.pstdev(v)
            _positive_sigma.append(_mean[-1] + _sigma)
            _negative_sigma.append(_mean[-1] - _sigma)
        return _x, _mean, _positive_sigma, _negative_sigma

    @staticmethod
    def plot_input_velocity(config: PlotConfig):
        """
        Plot the mean, upper deviation, and lower deviation curves of the input data.
        :return:
        """
        # Merge decode-only and prefill groups
        _x, _mean, _positive_sigma, _negative_sigma = AnalysisState.computer_mean_sigma(config.data, config.x_field)
        plt.plot(_x, _mean, label="mean")
        plt.plot(_x, _positive_sigma, label="positive std")
        plt.plot(_x, _negative_sigma, label="negative std")
        plt.title(config.title)
        plt.legend()
        plt.grid()
        if config.x_label:
            plt.xlabel(config.x_label)
        if config.y_label:
            plt.ylabel(config.y_label)
        if config.save_path:
            plt.savefig(Path(config.save_path).joinpath(f"{config.x_label}_{config.y_label}_{config.title}.png"))
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_input_velocity_with_df(predict_df, origin_df, save_path=None):
        """
        Plot the mean, upper deviation, and lower deviation curves of the input data.
        :return:
        """
        # Merge decode-only and prefill groups
        _batch_stage = predict_df["batch_stage"].unique()
        for _bs in _batch_stage:
            _cur_predict_df = predict_df[predict_df["batch_stage"] == _bs]
            _cur_origin_df = origin_df[origin_df["batch_stage"] == _bs]
            _p_group = _cur_predict_df.groupby("batch_size")
            _p_mean = _p_group.mean()
            _p_sigma = _p_group.std().fillna(0)
            _o_mean = _cur_origin_df.groupby("batch_size").mean().values.flatten()
            _o_sigma = _cur_origin_df.groupby("batch_size").std().fillna(0).values.flatten()

            plt.plot(_p_mean.index.values, _p_mean.values.flatten(), label="predict mean")
            plt.plot(_p_mean.index.values, (_p_mean.values + _p_sigma.values).flatten(), label="predict positive std")
            plt.plot(_p_mean.index.values, (_p_mean.values - _p_sigma.values).flatten(), label="predict negative std")

            plt.plot(_o_mean.index.values, _o_mean.values.flatten(), label="origin mean")
            plt.plot(_o_mean.index.values, (_o_mean.values + _o_sigma.values).flatten(), label="origin positive std")
            plt.plot(_o_mean.index.values, (_o_mean.values - _o_sigma.values).flatten(), label="origin negative std")
            plt.title(f"{_bs} latency")
            plt.legend()
            plt.grid()
            plt.xlabel("batch size")
            plt.ylabel("res")
            if save_path:
                plt.savefig(Path(save_path).joinpath(f"{_bs}_batch_size_res.png"))
                plt.close()
            else:
                plt.show()

    @staticmethod
    def plot_input_velocity_with_predict(config: PlotConfig, predict_data: Dict[State, List]):
        """
        Plot the mean, upper deviation, and lower deviation curves of the input data and predicted data.
        :return:
        """
        # Merge decode-only and prefill groups
        # Merge decode-only and prefill groups
        (
            _x,
            _mean,
            _positive_sigma,
            _negative_sigma,
        ) = AnalysisState.computer_mean_sigma(config.data, config.x_field)
        plt.figure()
        plt.plot(_x, _mean, label="mean")
        plt.plot(_x, _positive_sigma, label="positive std")
        plt.plot(_x, _negative_sigma, label="negative std")
        _x, _predict, _predict_positive_sigma, _predict_negative_sigma = AnalysisState.computer_mean_sigma(
            predict_data, config.x_field
        )

        plt.plot(_x, _predict, label="predict")
        plt.plot(_x, _predict_positive_sigma, label="predict positive std")
        plt.plot(_x, _predict_negative_sigma, label="predict negative std")
        plt.title(config.title)
        plt.legend()
        plt.grid()
        if config.x_label:
            plt.xlabel(config.x_label)
        if config.y_label:
            plt.ylabel(config.y_label)
        if config.save_path:
            plt.savefig(Path(config.save_path).joinpath(f"{config.x_label}_{config.y_label}_{config.title}.png"))
            plt.close()
            with open_s(config.save_path.joinpath(f"{config.title}.txt"), 'w') as f:
                f.write('mean\n')
                f.write(json.dumps(_mean))
                f.write('\n')
                f.write('positive std\n')
                f.write(json.dumps(_positive_sigma))
                f.write('\n')
                f.write('negative std\n')
                f.write(json.dumps(_negative_sigma))
                f.write('\n')
                f.write('predict \n')
                f.write(json.dumps([float(i) for i in _predict]))
        else:
            plt.show()

    @staticmethod
    def plot_pred_and_real(pred, real, save_path: Optional[Path] = None):
        plt.figure()
        plt.scatter(range(len(pred)), pred, label='pred', alpha=0.5)
        plt.scatter(range(len(real)), real, label="real", alpha=0.5)
        plt.title("predict value and real value")
        plt.xlabel("index")
        plt.ylabel("value")
        plt.legend()
        if save_path:
            plt.savefig(save_path.joinpath("predict value and real value.png"))
            plt.close()
        else:
            plt.show()
