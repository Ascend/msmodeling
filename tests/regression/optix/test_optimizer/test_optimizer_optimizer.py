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
import json
import sys
import tempfile
import unittest
from math import inf
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


from optix.config.config import (
    PdDisaggConfig,
    field_to_param,
    get_settings,
    default_support_field,
    PerformanceIndex,
    OptimizerConfigField,
)
from optix.config.model_config import MindieModelConfig
from optix.optimizer.optimizer import PSOOptimizer, main
from optix.optimizer.experience_fine_tunning import StopFineTune


def _settings_mock() -> MagicMock:
    """Build a settings stub for the CLI-driven main() flow.

    Provides the fields required by standard/pd_disagg mode validation and
    PSOOptimizer/FineTune construction (pd_disagg uses the real default
    config to avoid MagicMock attributes being always truthy).
    """
    settings = MagicMock()
    settings.skip_pso = False
    settings.manage_simulator_lifecycle = True
    settings.n_particles = 3
    settings.iters = 2
    settings.ttft_penalty = 0
    settings.tpot_penalty = 0
    settings.success_rate_penalty = 0
    settings.ttft_slo = 1.0
    settings.tpot_slo = 0.1
    settings.success_rate_slo = 1.0
    settings.generate_speed_target = 100
    settings.max_fine_tune = 1
    settings.output = MagicMock()
    settings.step_size = 0.1
    settings.slo_coefficient = 1.0
    settings.ftol = 1e-3
    settings.ftol_iter = 2
    settings.data_storage = MagicMock()
    settings.fine_tune_mode = "concurrency_and_request_rate"
    settings.optimizer_strategy = "pso"
    settings.use_request_rate_calibration = False
    settings.pd_disagg = PdDisaggConfig()
    return settings


def _registry_mocks():
    """Create registry plugin stubs with minimal data_field so the target_field non-empty validation passes."""
    mock_simu = MagicMock()
    mock_simu.data_field = [OptimizerConfigField(name="max_batch_size", min=10, max=100, dtype="int")]
    mock_bench = MagicMock()
    mock_bench.data_field = [OptimizerConfigField(name="CONCURRENCY", min=1, max=64, dtype="int")]
    return mock_simu, mock_bench


@pytest.fixture
def generate_store(tmpdir):
    _config = Path(tmpdir).joinpath("data_storage_20250815175148.csv")
    data = [
        """generate_speed,time_to_first_token,time_per_output_token,success_rate,throughput,ttft_max,ttft_min,
        ttft_p75,ttft_p90,ttft_p99,tpot_max,tpot_min,tpot_p75,tpot_p90,tpot_p99,prefill_batch_size,
        prefill_batch_size_min,prefill_batch_size_max,prefill_batch_size_p75,prefill_batch_size_p90,
        prefill_batch_size_p99,decoder_batch_size,decoder_batch_size_min,decoder_batch_size_max,
        decoder_batch_size_p75,decoder_batch_size_p90,decoder_batch_size_p99,max_batch_size,
        max_prefill_batch_size,prefill_time_ms_per_req,max_queue_deloy_mircroseconds,support_select_batch,
        prefill_policy_type,decode_policy_type,CONCURRENCY,REQUESTRATE,
        error,bakcup,fitness,llm_model,dataset_path,simulator""",
        """1651.9238,25.858309600000002,0.459911,1.0,5.6491,122.56516400000001,0.6183994,23.629147099999997,
        97.3381163,110.01047340000001,96.42427740000001,0.0196638,0.3387629,0.493587,0.9246494000000001,
        19.2875,2.0,20.0,20.0,20.0,20.0,116.1429,104.0,119.0,118.0,118.0,118.66,118,20,673,579294,
        False,0,1,1000,0,,,2.3926191461362146e+65,qwen3-8b,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """1983.8449,0.3056046,0.0532928,1.0,7.7568,1.0516901,0.0407746,0.4211261,0.46272840000000004,
        0.5370466,0.7059757,1.26e-05,0.049732399999999996,0.0842894,0.17034649999999998,3.632,2.0,8.0,
        4.0,5.0,7.0,89.0854,28.0,126.0,110.0,117.9,125.19,493,296,580,350200,False,1,0,1000,8,,,
        3.858581664468014,qwen3-8b,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """1719.4973,0.30205509999999997,0.045293799999999995,1.0,6.7237,0.6956194,0.0435473,0.4166629,
        0.45262670000000005,0.5302,0.454728,1.2800000000000001e-05,0.042326,0.051689,0.1506695,
        3.1847,2.0,7.0,4.0,5.0,6.0,69.519,28.0,96.0,82.0,90.0,95.22,493,296,580,350200,False,1,0,1000,
        7,,,4.22999054887793,qwen3-8b,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """2628.1846,0.3350128,0.14603380000000002,1.0,10.276,0.7852948,0.0529368,0.4539578,0.5215298,
        0.6312122,0.445232,1.4999999999999999e-05,0.1741628,0.23369720000000002,0.3331318,5.2632,1.0,
        10.0,6.0,7.0,9.0,336.7906,71.0,487.0,418.0,459.4,484.24,493,296,580,350200,False,1,0,1000,12,,,
        98.08625743412847,qwen3-8b,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """1973.7498,0.2979828,0.0526314,1.0,7.7173,0.6064501,0.038523499999999995,0.41426260000000004,
        0.4516434,0.5381849,0.6366436,1.2800000000000001e-05,0.0486303,0.0840513,0.17018960000000002,
        3.5461,1.0,8.0,4.0,5.0,7.0,88.7778,27.0,123.0,107.0,115.0,122.2,613,228,430,344700,False,1,3,1000,
        8,,,3.858610905842756,qwen3-8b,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """1733.9595,0.2985424,0.0468999,1.0,6.7796,0.7885291,0.0356113,0.4122982,0.4491465,0.5274601999999999,
        0.36779480000000003,1.31e-05,0.0435739,0.0623085,0.1526414,3.1881,1.0,7.0,4.0,5.0,6.0,73.7917,25.0,100.0,
        86.25,92.9,99.29,613,228,430,344700,False,1,3,1000,7,,,4.220650109401511,qwen3-8b,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """2593.1693,0.3375048,0.1625388,1.0,10.1381,0.7262981,0.057505099999999996,0.4546796,
        0.5310524,0.6384890000000001,0.5680927,1.35e-05,0.198526,0.2595833,0.36268700000000004,
        5.3476,3.0,11.0,6.0,7.0,9.0,397.1239,61.0,591.0,499.5,555.4,587.62,613,228,430,344700,
        False,1,3,1000,12,,,259.53835449706077,qwen3-8b,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """2060.1813,0.49612459999999997,0.0503404,1.0,7.6264,1.1575395000000002,0.0538583,
        0.7074349999999999,0.7941196,0.903362,0.5112064,1.3e-05,0.0469313,0.0518141,0.196128,
        6.0729,3.0,11.0,7.0,8.0,10.0,90.7671,30.0,121.0,106.0,113.8,120.28,969,139,650,670600,
        False,0,3,1000,8,,,3.98983999087121,qwen3-8b,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """1745.8056,0.49327150000000003,0.0438012,1.0,6.8259,0.9928296,0.0596509,0.7072557,
        0.787812,0.8864957000000001,0.3891669,1.32e-05,0.041196199999999995,0.045159,0.1718553,
        5.4845,3.0,10.0,6.0,7.0,9.0,73.88,26.0,102.0,84.5,94.6,101.26,969,139,650,670600,False,0,
        3,1000,7,,,4.283805244651319,qwen3-8b,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """2407.8866,0.5219253,0.07873909999999999,1.0,9.4147,1.0784282,0.0694427,0.7191335,
        0.8201502,0.9298464,0.6178497,1.21e-05,0.0737019,0.097402,0.26629450000000005,7.5567,
        2.0,13.0,8.0,10.0,12.0,168.3263,44.0,223.0,200.5,213.6,222.06,969,139,650,670600,
        False,0,3,1000,10,,,4.754963590692261,
        qwen3-8b,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """1246.1593,0.4876233,0.0344552,1.0,4.8728,1.2842791,0.0470783,0.7252981,
        0.7706660000000001,0.86063,0.5466711000000001,1.23e-05,0.0328953,0.0346846,
        0.1309505,4.0541,1.0,8.0,5.0,6.0,7.0,42.0455,15.0,54.0,46.25,49.0,53.13,969,139,
        650,670600,False,0,3,1000,5,,,5.706182639610976,qwen3-8b,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """1966.4359,0.4968034,0.0497617,1.0,7.6895,1.0411084000000002,0.0623178,
        0.7018625,0.7927137,0.9036163,0.4520387,1.31e-05,0.0463816,0.051256100000000006,
        0.1904178,6.0852,4.0,11.0,7.0,8.0,10.0,89.8289,33.0,122.0,104.0,114.5,121.25,
        743,154,640,670600,False,0,1,1000,8,,,3.952181054954989,qwen3-8b,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """2157.4913,0.5037943,0.0586365,1.0,8.435,1.2262198,0.062042299999999995,
        0.7029421,0.8011816,0.9243581000000001,0.5565085000000001,1.3e-05,0.0547994,
        0.061973,0.2205356,6.7114,4.0,12.0,8.0,9.0,11.0,114.7286,31.0,155.0,136.75,
        147.1,153.62,743,154,640,670600,False,0,1,1000,9,,,3.870976410628638,
        qwen3-8b,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
        """1503.396,0.4886381,0.0381119,1.0,5.8787,0.9642645,0.050871400000000004,
        0.7220547,0.7747441,0.8654031,0.4425547,1.2800000000000001e-05,0.0362437,
        0.0388145,0.1474528,4.7771,3.0,10.0,5.0,6.0,8.0,55.6761,24.0,71.0,61.0,
        65.0,70.3,743,154,640,670600,False,0,1,1000,6,,,4.829008909894092,qwen3-8b,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,Simulator""",
    ]
    columns = data[0].replace('\n', '').replace(' ', '').split(',')
    rows = [row.split(',') for row in data[1:]]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(_config, index=False)
    yield _config
    _config.unlink()


@pytest.fixture
def generate_store2(tmpdir):
    _config = Path(tmpdir).joinpath("data_storage_20250904114749.csv")
    data = [
        """generate_speed,time_to_first_token,time_per_output_token,success_rate,throughput,
        ttft_max,ttft_min,ttft_p75,ttft_p90,ttft_p99,tpot_max,tpot_min,tpot_p75,tpot_p90,
        tpot_p99,prefill_batch_size,prefill_batch_size_min,prefill_batch_size_max,prefill_batch_size_p75,
        prefill_batch_size_p90,prefill_batch_size_p99,decoder_batch_size,decoder_batch_size_min,
        decoder_batch_size_max,decoder_batch_size_p75,decoder_batch_size_p90,decoder_batch_size_p99,
        max_batch_size,max_prefill_batch_size,prefill_time_ms_per_req,decode_time_ms_per_req,
        support_select_batch,max_queue_deloy_mircroseconds,prefill_policy_type,decode_policy_type,
        max_preempt_count,CONCURRENCY,REQUESTRATE,error,backup,duration,real_evaluation,fitness,
        llm_model,dataset_path,num_prompts,max_output_len,simulator""",
        """2243.6598,1.3100832,0.051145,0.9990000000000001,9.1665,7.1490709,0.056097799999999996,
        1.7559865,2.2709803,4.264462900000001,0.9048023,1.78e-05,0.048797,0.054466099999999996,
        0.24111259999999998,8.4661,1.0,10.0,10.0,10.0,10.0,111.3077,66.0,128.0,119.25,122.0,126.98,200,
        10,600,50,True,5000,1,1,0,nan,nan,,,434.9610929489136,True,30.239561616685595,qwen2.5,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,3000,256,Simulator""",
        """1810.5347,0.1802269,0.1273566,0.9990000000000001,7.3775,0.6179893,0.0340387,0.222336,
        0.2489846,0.350022,41.6759139,1.8100000000000003e-05,0.0439948,0.0745859,0.1728956,
        2.0885,1.0,6.0,3.0,3.0,5.0,70.0864,14.0,86.0,86.0,86.0,86.0,86,39,330,390,False,
        153900,3,0,35,1000,8.604645000000001,,,512.5828998088837,True,36.193461882497374,
        qwen2.5,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,3000,256,Simulator""",
        """,,,,,,,,,,,,,,,,,,,,,,,,,,,86,39,330,390,False,153900,3,0,35,1000,1.9483841859300024,
        The current runtime is more than twice the duration of the first run.,,872.4831893444061,
        True,inf,qwen2.5,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,3000,256,Simulator""",
        """2319.1496,0.43147620000000003,0.0591992,0.9990000000000001,9.474,1.3781652999999998,
        0.0527782,0.6037235,0.6838597,0.7894299,0.8658081,1.8200000000000002e-05,0.0577695,
        0.07392789999999999,0.2155833,6.6308,2.0,146.0,7.0,8.0,10.0,114.0842,30.0,162.0,140.5,
        154.6,161.06,181,94,890,980,False,555700,3,2,166,1000,10.34502,,,472.5490171909332,True,
        4.623869543910539,qwen2.5,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,3000,256,Simulator""",
        """2181.9876,0.426743,0.050761099999999997,0.9990000000000001,8.9273,0.9393971,0.0614861,
        0.6018292000000001,0.6744909,0.7756989999999999,0.6022356,1.7999999999999997e-05,0.048820300000000004,
        0.056043,0.1932687,5.9701,1.0,11.0,7.0,8.0,10.0,97.4,17.0,136.0,119.25,127.1,134.42,181,94,890,980,
        False,555700,3,2,166,1000,9.393360920160001,,,442.7061245441437,True,4.656442391641238,qwen2.5,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,3000,256,Simulator""",
        """2705.9621,0.514597,0.0992732,0.9990000000000001,11.0497,1.0416181,0.0828408,0.7071916,0.8129012,
        0.9275639999999999,0.4548811,1.84e-05,0.1052222,0.1670604,0.3017118,8.7632,3.0,15.0,10.0,11.0,
        13.0,234.6018,22.0,366.0,302.0,335.0,357.8,445,210,70,170,False,646300,0,2,55,1000,12.79383,,,
        380.01217889785767,True,9.403965887786503,qwen2.5,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,
        3000,256,Simulator""",
        """1569.3499,0.47306909999999996,0.03462,0.9990000000000001,6.4296,0.9368759999999999,
        0.051654200000000004,0.6978179999999999,0.7497718,0.8353029000000001,0.4186953,
        1.8100000000000003e-05,0.033464100000000004,0.0365732,0.1395875,4.9867,2.0,10.0,6.0,7.0,
        9.0,50.8523,8.0,72.0,59.25,65.0,71.13,445,210,70,170,False,646300,0,2,55,1000,6.48990055644,,,
        576.2215044498444,True,6.108580632749877,qwen2.5,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,
        3000,256,Simulator""",
        """2566.7401,0.5480697999999999,0.0736538,0.9990000000000001,10.5049,1.1309966999999999,0.0664723,
        0.7539478000000001,0.8749999,0.9987252,2.0093673,1.7899999999999998e-05,0.07268770000000001,
        0.09044840000000001,0.2738134,8.9731,6.0,15.0,10.0,11.0,13.0,160.7815,29.0,224.0,197.5,215.2,
        224.0,224,70,850,650,False,724300,3,0,202,1000,11.36709,,,394.799791097641,True,5.104055938300174,
        qwen2.5,/data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,3000,256,Simulator""",
        """2033.3873,0.5258246000000001,0.044589199999999996,0.9990000000000001,8.319,1.1236912,0.0649515,
        0.7391481,0.8455796,0.9509563000000001,0.5170958000000001,1.7999999999999997e-05,0.0429984,0.0479505,
        0.18713370000000001,6.9698,4.0,13.0,8.0,9.0,11.71,81.7692,27.0,117.0,98.0,107.3,114.69,224,70,850,650,
        False,724300,3,0,202,1000,8.678341265579999,,,468.3717620372772,True,4.964495470257898,qwen2.5,
        /data/ms_serviceparam_optimizer/datasets/short_input_64_2k.jsonl,3000,256,Simulator""",
    ]
    columns = data[0].replace('\n', '').replace(' ', '').split(',')
    rows = [row.split(',') for row in data[1:]]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(_config, index=False)
    yield _config
    _config.unlink()


def test_computer_fitness():
    # Create a PSOOptimizer instance
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field[:1])
    optimizer.minimum_algorithm = MagicMock(return_value=1.0)
    # Mock the load_history_data behavior
    optimizer.load_history_data = [
        {
            'generate_speed': 1,
            'time_to_first_token': 2,
            'time_per_output_token': 3,
            'success_rate': 1.0,
            "max_batch_size": 150,
        },
        {
            'generate_speed': 4,
            'time_to_first_token': 5,
            'time_per_output_token': 6,
            'success_rate': 1.0,
            "max_batch_size": 160,
        },
        {
            'generate_speed': 7,
            'time_to_first_token': 8,
            'time_per_output_token': 9,
            'success_rate': 1.0,
            "max_batch_size": 170,
        },
    ]
    # Call the computer_fitness method
    positions, costs = optimizer.computer_fitness()

    # Check the results
    assert np.array(positions).size == 3
    assert costs == [1.0, 1.0, 1.0]


def test_computer_fitness_with_key_error():
    # Create a PSOOptimizer instance
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)
    optimizer.minimum_algorithm = MagicMock(return_value=1.0)
    # Mock the load_history_data behavior
    optimizer.load_history_data = [
        {
            'generate_speed': 1,
            'time_to_first_token': 2,
            'time_per_output_token': 3,
            'success_rate': 1.0,
            "max_batch_size": 3,
            "fitness": 3,
        },
        {
            'generate_speed': 4,
            'time_to_first_token': 5,
            'time_per_output_token': 6,
            'success_rate': 1.0,
            "max_batch_size": 4,
            "fitness": 32,
        },
        {
            'generate_speed': 7,
            'time_to_first_token': 8,
            'time_per_output_token': 9,
            'success_rate': 1.0,
            "max_batch_size": 5,
            "fitness": inf,
        },
    ]

    # Call the computer_fitness method
    positions, costs = optimizer.computer_fitness()

    # Check the results; records with missing fields should be ignored
    assert len(positions) == 0
    assert len(costs) == 0


# Test data
TEST_PARAMS = np.array([[1.0, 2.0], [3.0, 4.0]])
TEST_PARAMS_FIELD = ('field1', 'field2')


# Test cases
def test_op_func_success():
    # Create a PSOOptimizer instance
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)
    # Mock the scheduler.run method
    optimizer.scheduler.run_with_request_rate = MagicMock(
        return_value=PerformanceIndex(
            generate_speed=100, time_to_first_token=0.1, time_per_output_token=0.1, success_rate=1
        )
    )

    # Call the op_func method
    result = optimizer.op_func(TEST_PARAMS)

    # Verify scheduler.run and minimum_algorithm are called correctly
    optimizer.scheduler.run_with_request_rate.assert_called()
    # Verify the return value
    assert result.size == 2


def test_op_func_exception():
    # Create a PSOOptimizer instance
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field)

    # Mock the scheduler.run method to raise an exception
    optimizer.scheduler = MagicMock()
    optimizer.scheduler.run_with_request_rate = MagicMock(side_effect=Exception("Test Exception"))

    # Call the op_func method
    result = optimizer.op_func(TEST_PARAMS)

    # Verify scheduler.run and minimum_algorithm are called correctly
    optimizer.scheduler.run_with_request_rate.assert_called()

    # Verify the return value
    assert np.array_equal(result, np.array([float('inf'), float('inf')]))


class MockField:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value


class TestPSOOptimizer:
    @classmethod
    def test_constructing_bounds_empty_target_field(cls, optimizer):
        optimizer.target_field = []
        min_bounds, max_bounds = optimizer.constructing_bounds()
        assert min_bounds == ()
        assert max_bounds == ()

    @classmethod
    def test_constructing_bounds_single_target_field(cls, optimizer):
        optimizer.target_field = [OptimizerConfigField(min=0, max=10)]
        min_bounds, max_bounds = optimizer.constructing_bounds()
        assert min_bounds == (0,)
        assert max_bounds == (10,)

    @classmethod
    def test_constructing_bounds_multiple_target_fields(cls, optimizer):
        optimizer.target_field = [OptimizerConfigField(min=0, max=10), OptimizerConfigField(min=20, max=30)]
        min_bounds, max_bounds = optimizer.constructing_bounds()
        assert min_bounds == (0, 20)
        assert max_bounds == (10, 30)

    @classmethod
    def test_dimensions_all_fields_equal(cls, optimizer):
        optimizer.target_field = [
            OptimizerConfigField(min=1, max=1),
            OptimizerConfigField(min=2, max=2),
            OptimizerConfigField(min=3, max=3),
        ]
        assert optimizer.dimensions() == 0

    @classmethod
    def test_dimensions_some_fields_different(cls, optimizer):
        optimizer.target_field = [
            OptimizerConfigField(min=1, max=1),
            OptimizerConfigField(min=2, max=3),
            OptimizerConfigField(min=4, max=4),
        ]
        assert optimizer.dimensions() == 1

    @classmethod
    def test_dimensions_empty_fields(cls, optimizer):
        optimizer.target_field = []
        assert optimizer.dimensions() == 0

    @classmethod
    def test_is_within_boundary(cls):
        target_pos = [1, 2, 3, 4]
        min_bound = [0, 2, 2, 3]
        max_bound = [2, 3, 3, 5]
        assert PSOOptimizer.is_within_boundary(target_pos, min_bound, max_bound)
        min_bound = [0, 2, 2, 5]
        max_bound = [2, 3, 3, 10]
        assert not PSOOptimizer.is_within_boundary(target_pos, min_bound, max_bound)

    @classmethod
    def test_params_in_records(cls):
        _params = np.array([1, 2, 3, 4])
        _param_records = [np.array([5, 6, 7, 8]), np.array([1, 2, 3, 4])]
        assert PSOOptimizer.params_in_records(_params, _param_records)
        _params = np.array([4, 3, 2, 1])
        assert not PSOOptimizer.params_in_records(_params, _param_records)

    @pytest.fixture
    def optimizer(self):
        return PSOOptimizer(MagicMock(), target_field=default_support_field)


@patch("optix.optimizer.optimizer.field_to_param")
def test_refine_optimization_candidates(field_to_param_patch):
    field_to_param_patch.side_effect = [[3, 4], [7, 1]]
    pso = PSOOptimizer(MagicMock(), target_field=default_support_field)
    pso.default_res = PerformanceIndex()
    pso.default_fitness = 3.333
    pso.default_run_param = [5.5, 6.3]
    best_results = pd.DataFrame({"field1": [1, 2], "field2": [3, 4]})
    pso.scheduler = MagicMock()
    pso.scheduler.data_storage.config.pso_top_k = 1
    pso.scheduler.run = MagicMock(return_value=PerformanceIndex())
    pso.scheduler.save_result = MagicMock()
    pso.fine_tune = MagicMock()
    pso.fine_tune.fine_tune_with_concurrency_and_request_rate = MagicMock(side_effect=StopFineTune)
    pso.minimum_algorithm = MagicMock(return_value=2)
    pso.get_target_field_from_case_data = MagicMock(return_value=default_support_field)
    pso.params_in_records = MagicMock(return_value=False)
    fitness, params, res = pso.refine_optimization_candidates(best_results)
    assert fitness == [3.333, 2, 2]
    assert params == [[5.5, 6.3], [3, 4], [7, 1]]
    assert len(res) == 3


@patch("optix.optimizer.optimizer.field_to_param")
def test_refine_optimization_candidates_last_concurrency(field_to_param_patch):
    field_to_param_patch.side_effect = [[3, 4], [7, 1], [2, 5], [3, 1], [2, 6], [9, 1]]
    my_support_field = [
        # The min value of max_batch_size must be greater than the max value of max_prefill_batch_size.
        OptimizerConfigField(
            name="max_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxBatchSize",
            min=10,
            max=1000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="max_prefill_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize",
            min=0.1,
            max=0.7,
            dtype="ratio",
            dtype_param="max_batch_size",
        ),
        OptimizerConfigField(
            name="prefill_time_ms_per_req",
            config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq",
            max=1000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="support_select_batch",
            config_position="BackendConfig.ScheduleConfig.supportSelectBatch",
            max=1,
            dtype="bool",
        ),
        OptimizerConfigField(
            name="max_queue_deloy_mircroseconds",
            config_position="BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds",
            min=500,
            max=1000000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="prefill_policy_type",
            config_position="BackendConfig.ScheduleConfig.prefillPolicyType",
            min=0,
            max=1,
            dtype="enum",
            dtype_param=[0, 1, 3],
        ),
        OptimizerConfigField(
            name="decode_policy_type",
            config_position="BackendConfig.ScheduleConfig.decodePolicyType",
            min=0,
            max=1,
            dtype="enum",
            dtype_param=[0, 1, 3],
        ),
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int"),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int"),
    ]
    pso = PSOOptimizer(MagicMock(), target_field=my_support_field[-2:])
    pso.default_res = (
        PerformanceIndex(
            generate_speed=2000, time_to_first_token=5.5, time_per_output_token=1.3, success_rate=1, throughput=5
        ),
    )
    pso.default_fitness = 3.333
    pso.default_run_param = [500, 6.3]
    best_results = pd.DataFrame({"CONCURRENCY": [700], "REQUESTRATE": [3.8]})
    pso.scheduler = MagicMock()
    pso.scheduler.run = MagicMock(
        side_effect=[
            PerformanceIndex(
                generate_speed=2000, time_to_first_token=5, time_per_output_token=1, success_rate=1, throughput=5
            ),
            PerformanceIndex(
                generate_speed=2435, time_to_first_token=3.23, time_per_output_token=0.14, success_rate=1, throughput=5
            ),
            PerformanceIndex(
                generate_speed=1700, time_to_first_token=0.4, time_per_output_token=0.03, success_rate=1, throughput=5
            ),
            PerformanceIndex(
                generate_speed=1750, time_to_first_token=0.43, time_per_output_token=0.038, success_rate=1, throughput=5
            ),
            PerformanceIndex(
                generate_speed=1820, time_to_first_token=0.57, time_per_output_token=0.038, success_rate=1, throughput=5
            ),
            PerformanceIndex(
                generate_speed=1800, time_to_first_token=0.44, time_per_output_token=0.04, success_rate=1, throughput=5
            ),
        ]
    )
    pso.scheduler.save_result = MagicMock()
    pso.scheduler.data_storage.config.pso_top_k = 1
    pso.fine_tune = MagicMock()
    pso.fine_tune.fine_tune_with_concurrency_and_request_rate = MagicMock(side_effect=StopFineTune)
    pso.minimum_algorithm = MagicMock(side_effect=[4, 3.8, 2.9, 3.3, 1.2, 3.9, 3.6])
    pso.params_in_records = MagicMock(return_value=False)
    fitness, params, res = pso.refine_optimization_candidates(best_results)
    assert fitness == [3.333, 4]
    assert params == [[500, 6.3], [3, 4]]
    assert len(res) == 2


@patch("optix.optimizer.optimizer.is_mindie", return_value=True)
@patch("optix.config.model_config.MindieModelConfig")
def test_prepare(mock_mindie_model_config, mock_is_mindie, mindie_config_file):
    optimizer = PSOOptimizer(MagicMock(), target_field=default_support_field[:5])
    with open(mindie_config_file, 'r', encoding="utf-8") as f:
        optimizer.scheduler.simulator.default_config = json.load(f)
    optimizer.scheduler.run = MagicMock(
        return_value=PerformanceIndex(
            generate_speed=369, time_to_first_token=0.3, time_per_output_token=0.05, success_rate=1
        )
    )
    optimizer.scheduler.save_result = MagicMock(return_value=True)
    optimizer.scheduler.error_info = None
    optimizer.mindie_prepare = MagicMock(return_value=None)
    assert optimizer.default_run_param is None
    assert optimizer.default_res is None
    assert optimizer.default_fitness is None
    optimizer.prepare_plugin()
    assert optimizer.default_res
    assert optimizer.default_fitness
    assert optimizer.default_run_param is not None


def test_run_plugin():
    # Create a PSOOptimizer instance
    optimizer = PSOOptimizer(
        MagicMock(),
        pso_options=get_settings().pso_options,
        target_field=default_support_field[:3],
        pso_init_kwargs={"ftol": get_settings().ftol, "ftol_iter": get_settings().ftol_iter},
        load_breakpoint=True,
    )
    performance_index = PerformanceIndex(
        generate_speed=888, time_to_first_token=0.5, time_per_output_token=0.05, success_rate=1
    )
    # Mock the prepare method

    with patch('optix.optimizer.global_best_custom.CustomGlobalBestPSO', autospec=True) as mock_custom_global_best_pso:
        # Mock the enable_simulate context manager
        with patch('optix.optimizer.optimizer.enable_simulate', autospec=True):
            custom_global_instance = mock_custom_global_best_pso.return_value
            custom_global_instance.optimize.return_value = (100, [200, 10, 100])
            # Mock the op_func method
            optimizer.prepare_plugin = MagicMock()
            optimizer.op_func = MagicMock()
            # Mock the refine_optimization_candidates method
            optimizer.refine_optimization_candidates = MagicMock(
                return_value=([100], [[200, 10, 100]], [performance_index])
            )
            # Mock the best_params method
            optimizer.best_params = MagicMock(return_value=(100, [200, 10, 100], performance_index))
            # Mock scheduler.data_storage
            optimizer.scheduler.data_storage = MagicMock()
            # Mock scheduler.simulator
            optimizer.scheduler.simulator = MagicMock()
            # Call the run method
            optimizer.run_plugin()
            # Verify the prepare method is called
            optimizer.prepare_plugin.assert_called_once()
            # Verify the context manager is used correctly
            # Verify the CustomGlobalBestPSO class is instantiated correctly
            mock_custom_global_best_pso.assert_called_once()
            # Verify refine_optimization_candidates and best_params are called
            optimizer.refine_optimization_candidates.assert_called_once()
            optimizer.best_params.assert_called_once()
            # Verify scheduler.data_storage and scheduler.simulator are used correctly
            optimizer.scheduler.data_storage.get_best_result.assert_called_once()


@patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
@patch("optix.optimizer.optimizer.PSOOptimizer")
@patch("optix.optimizer.scheduler.Scheduler")
@patch("optix.optimizer.store.DataStorage")
@patch("optix.optimizer.experience_fine_tunning.FineTune")
@patch("optix.deploy_env.emit_runtime_hints")
@patch("optix.deploy_env.validate_deploy_stack")
@patch("optix.deploy_env.resolve_deploy_context", return_value=(MagicMock(), {}))
@patch("optix.config.config.get_settings")
@patch("optix.optimizer.register.register_ori_functions")
@patch("optix.plugins.load_general_plugins")
@patch("optix.optimizer.optimizer.is_root", return_value=False)
def test_plugin_main(
    mock_is_root,
    mock_load_plugins,
    mock_register,
    mock_get_settings,
    mock_resolve_deploy,
    mock_emit_hints,
    mock_validate_deploy,
    mock_fine_tune,
    mock_ds,
    mock_scheduler,
    mock_pso,
    mock_which,
):
    mock_get_settings.return_value = _settings_mock()
    mock_simu, mock_bench = _registry_mocks()
    argv = ["optix", "-b", "vllm_benchmark", "-e", "vllm"]
    with (
        patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
        patch.dict("optix.optimizer.register.benchmarks", {"vllm_benchmark": lambda **kw: mock_bench}),
        patch.object(sys, "argv", argv),
    ):
        main()

    mock_pso.assert_called_once()
    mock_scheduler.assert_called_once()


def test_best_params(generate_store):  # pylint: disable=redefined-outer-name
    from optix.optimizer.experience_fine_tunning import FineTune

    optimizer_result = pd.read_csv(generate_store)
    my_support_field = [
        # The min value of max_batch_size must be greater than the max value of max_prefill_batch_size.
        OptimizerConfigField(
            name="max_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxBatchSize",
            min=10,
            max=1000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="max_prefill_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize",
            min=0.1,
            max=0.7,
            dtype="ratio",
            dtype_param="max_batch_size",
        ),
        OptimizerConfigField(
            name="prefill_time_ms_per_req",
            config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq",
            max=1000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="support_select_batch",
            config_position="BackendConfig.ScheduleConfig.supportSelectBatch",
            max=1,
            dtype="bool",
        ),
        OptimizerConfigField(
            name="max_queue_deloy_mircroseconds",
            config_position="BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds",
            min=500,
            max=1000000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="prefill_policy_type",
            config_position="BackendConfig.ScheduleConfig.prefillPolicyType",
            min=0,
            max=1,
            dtype="enum",
            dtype_param=[0, 1, 3],
        ),
        OptimizerConfigField(
            name="decode_policy_type",
            config_position="BackendConfig.ScheduleConfig.decodePolicyType",
            min=0,
            max=1,
            dtype="enum",
            dtype_param=[0, 1, 3],
        ),
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int"),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int"),
    ]
    pso = PSOOptimizer(MagicMock(), target_field=tuple(my_support_field))
    _fitness_list = []
    _params_list = []
    _performance_index_list = []
    for _, row in optimizer_result.iterrows():
        try:
            _target_field = pso.get_target_field_from_case_data(row)
        except Exception:
            _target_field = {}
            continue
        params = field_to_param(_target_field)
        _params_list.append(params)
        _fitness = row.get("fitness")
        _fitness_list.append(_fitness)
        _params = {}
        for k in PerformanceIndex.model_fields.keys():
            if k in row:
                _params[k] = row[k]
        performance_index = PerformanceIndex(**_params)
        _performance_index_list.append(performance_index)
    pso.fine_tune = FineTune(
        ttft_penalty=get_settings().ttft_penalty,
        tpot_penalty=get_settings().tpot_penalty,
        target_field=_target_field,
        ttft_slo=get_settings().ttft_slo,
        tpot_slo=get_settings().tpot_slo,
        slo_coefficient=get_settings().slo_coefficient,
        step_size=get_settings().step_size,
    )

    best_fitness, best_param, best_performance_index = pso.best_params(
        _fitness_list, _params_list, _performance_index_list
    )
    assert best_performance_index.generate_speed == 1966.4359
    assert best_fitness == 3.952181054954989
    pso.tpot_penalty = 0
    pso.ttft_penalty = 0
    best_fitness, best_param, best_performance_index = pso.best_params(
        _fitness_list, _params_list, _performance_index_list
    )
    assert best_performance_index.generate_speed == 2628.1846
    pso.tpot_penalty = 3.0
    best_fitness, best_param, best_performance_index = pso.best_params(
        _fitness_list, _params_list, _performance_index_list
    )
    assert best_performance_index.generate_speed == 1966.4359
    # Case where ttft and tpot are not both within SLO
    _fitness_list = []
    _params_list = []
    _performance_index_list = []
    for i in [0, 3, 6, 9, 12]:
        row = optimizer_result.iloc[i]
        _target_field = pso.get_target_field_from_case_data(row)
        params = field_to_param(_target_field)
        _params_list.append(params)
        _fitness = row.get("fitness")
        _fitness_list.append(_fitness)
        _params = {}
        for k in PerformanceIndex.model_fields.keys():
            if k in row:
                _params[k] = row[k]
        performance_index = PerformanceIndex(**_params)
        _performance_index_list.append(performance_index)
    pso.tpot_penalty = 3.0
    pso.ttft_penalty = 3.0
    best_fitness, best_param, best_performance_index = pso.best_params(
        _fitness_list, _params_list, _performance_index_list
    )
    assert best_performance_index.generate_speed == 2157.4913
    _fitness_list = []
    _params_list = []
    _performance_index_list = []
    for i in [3, 6, 9, 12]:
        row = optimizer_result.iloc[i]
        _target_field = pso.get_target_field_from_case_data(row)
        params = field_to_param(_target_field)
        _params_list.append(params)
        _fitness = row.get("fitness")
        _fitness_list.append(_fitness)
        _params = {}
        for k in PerformanceIndex.model_fields.keys():
            if k in row:
                _params[k] = row[k]
        performance_index = PerformanceIndex(**_params)
        _performance_index_list.append(performance_index)
    pso.ttft_penalty = 0
    best_fitness, best_param, best_performance_index = pso.best_params(
        _fitness_list, _params_list, _performance_index_list
    )
    assert best_performance_index.generate_speed == 2157.4913


def test_best_params2(generate_store2):  # pylint: disable=redefined-outer-name
    from optix.optimizer.experience_fine_tunning import FineTune

    optimizer_result = pd.read_csv(generate_store2)
    my_support_field = [
        # The min value of max_batch_size must be greater than the max value of max_prefill_batch_size.
        OptimizerConfigField(
            name="max_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxBatchSize",
            min=10,
            max=1000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="max_prefill_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxPrefillBatchSize",
            min=0.1,
            max=0.7,
            dtype="ratio",
            dtype_param="max_batch_size",
        ),
        OptimizerConfigField(
            name="prefill_time_ms_per_req",
            config_position="BackendConfig.ScheduleConfig.prefillTimeMsPerReq",
            max=1000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="support_select_batch",
            config_position="BackendConfig.ScheduleConfig.supportSelectBatch",
            max=1,
            dtype="bool",
        ),
        OptimizerConfigField(
            name="max_queue_deloy_mircroseconds",
            config_position="BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds",
            min=500,
            max=1000000,
            dtype="int",
        ),
        OptimizerConfigField(
            name="prefill_policy_type",
            config_position="BackendConfig.ScheduleConfig.prefillPolicyType",
            min=0,
            max=1,
            dtype="enum",
            dtype_param=[0, 1, 3],
        ),
        OptimizerConfigField(
            name="decode_policy_type",
            config_position="BackendConfig.ScheduleConfig.decodePolicyType",
            min=0,
            max=1,
            dtype="enum",
            dtype_param=[0, 1, 3],
        ),
        OptimizerConfigField(name="CONCURRENCY", config_position="env", min=10, max=1001, dtype="int"),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", min=0, max=1001, dtype="int"),
    ]
    pso = PSOOptimizer(MagicMock(), target_field=tuple(my_support_field))
    _fitness_list = []
    _params_list = []
    _performance_index_list = []
    for _, row in optimizer_result.iterrows():
        try:
            _target_field = pso.get_target_field_from_case_data(row)
        except Exception:
            _target_field = {}
            continue
        params = field_to_param(_target_field)
        _params_list.append(params)
        _fitness = row.get("fitness")
        _fitness_list.append(_fitness)
        _params = {}
        for k in PerformanceIndex.model_fields.keys():
            if k in row:
                _params[k] = row[k]
        performance_index = PerformanceIndex(**_params)
        _performance_index_list.append(performance_index)

    pso.fine_tune = FineTune(
        ttft_penalty=get_settings().ttft_penalty,
        tpot_penalty=get_settings().tpot_penalty,
        target_field=_target_field,
        ttft_slo=get_settings().ttft_slo,
        tpot_slo=get_settings().tpot_slo,
        slo_coefficient=get_settings().slo_coefficient,
        step_size=get_settings().step_size,
    )

    best_fitness, best_param, best_performance_index = pso.best_params(
        _fitness_list, _params_list, _performance_index_list
    )
    assert best_performance_index.generate_speed == 2033.3873
    assert best_fitness == 4.964495470257898
    pso.tpot_penalty = 0
    pso.ttft_penalty = 0
    best_fitness, best_param, best_performance_index = pso.best_params(
        _fitness_list, _params_list, _performance_index_list
    )
    assert best_performance_index.generate_speed == 2705.9621
    pso.tpot_penalty = 3.0
    best_fitness, best_param, best_performance_index = pso.best_params(
        _fitness_list, _params_list, _performance_index_list
    )
    assert best_performance_index.generate_speed == 2033.3873


def test_mindie_prepare_theory_guided_disable():
    optimizer = PSOOptimizer(MagicMock())
    optimizer.mindie_prepare(None)
    assert True  # No exception should be raised


def test_mindie_prepare_mc_none(monkeypatch):
    optimizer = PSOOptimizer(MagicMock())
    monkeypatch.setattr(get_settings(), "theory_guided_enable", False)
    optimizer.mindie_prepare(MagicMock())
    assert True  # No exception should be raised


def test_mindie_prepare_valid_input(monkeypatch):
    target_field = (
        OptimizerConfigField(
            name="max_batch_size",
            config_position="BackendConfig.ScheduleConfig.maxBatchSize",
            min=10,
            max=1000,
            dtype="int",
        ),
    )
    monkeypatch.setattr(get_settings(), "theory_guided_enable", True)
    optimizer = PSOOptimizer(MagicMock(), target_field=target_field)
    mc = MagicMock(spec=MindieModelConfig)
    mc.get_max_batch_size_bound = MagicMock(return_value=(100, 300))
    optimizer.scheduler = MagicMock()
    optimizer.scheduler.benchmark.get_performance_metric = MagicMock(return_value=15)
    optimizer.scheduler.benchmark.benchmark_config.command.max_output_len = 30
    optimizer.mindie_prepare(mc)
    assert optimizer.target_field[0].min == 100
    assert optimizer.target_field[0].max == 390


class TestArgParse(unittest.TestCase):
    """Tests for argument parsing behavior of the optimizer CLI entry."""

    def _settings_mock(self):
        return _settings_mock()

    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.deploy_env.emit_runtime_hints")
    @patch("optix.deploy_env.validate_deploy_stack")
    @patch("optix.deploy_env.resolve_deploy_context", return_value=(MagicMock(), {}))
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_arg_parse_with_plugin(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_resolve_deploy,
        mock_emit_hints,
        mock_validate_deploy,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_which,
    ):
        mock_get_settings.return_value = self._settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        argv = ["optix"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            main()
        mock_pso.assert_called_once()

    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.deploy_env.emit_runtime_hints")
    @patch("optix.deploy_env.validate_deploy_stack")
    @patch("optix.deploy_env.resolve_deploy_context", return_value=(MagicMock(), {}))
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_arg_parse_with_load_breakpoint(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_resolve_deploy,
        mock_emit_hints,
        mock_validate_deploy,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_which,
    ):
        mock_get_settings.return_value = self._settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        argv = ["optix", "--load_breakpoint"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            main()
        mock_pso.assert_called_once()
        self.assertTrue(mock_pso.call_args.kwargs["load_breakpoint"])

    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.deploy_env.emit_runtime_hints")
    @patch("optix.deploy_env.validate_deploy_stack")
    @patch("optix.deploy_env.resolve_deploy_context", return_value=(MagicMock(), {}))
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_arg_parse_with_backup(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_resolve_deploy,
        mock_emit_hints,
        mock_validate_deploy,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_which,
    ):
        tmp_path = Path(tempfile.mkdtemp())
        settings = self._settings_mock()
        settings.output = tmp_path
        mock_get_settings.return_value = settings
        mock_simu, mock_bench = _registry_mocks()
        argv = ["optix", "--backup"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            main()
        self.assertTrue((tmp_path / "back_up").is_dir())
        mock_pso.assert_called_once()

    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.deploy_env.emit_runtime_hints")
    @patch("optix.deploy_env.validate_deploy_stack")
    @patch("optix.deploy_env.resolve_deploy_context", return_value=(MagicMock(), {}))
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_arg_parse_with_deploy_policy(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_resolve_deploy,
        mock_emit_hints,
        mock_validate_deploy,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_which,
    ):
        mock_get_settings.return_value = self._settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        argv = ["optix", "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            main()
        self.assertEqual(mock_scheduler.call_args.kwargs["engine"], "vllm")

    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.deploy_env.emit_runtime_hints")
    @patch("optix.deploy_env.validate_deploy_stack")
    @patch("optix.deploy_env.resolve_deploy_context", return_value=(MagicMock(), {}))
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_arg_parse_with_pd(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_resolve_deploy,
        mock_emit_hints,
        mock_validate_deploy,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_which,
    ):
        mock_phase_runner = patch("optix.optimizer.pd_disagg.OptixPhaseRunner").start()
        mock_orchestrator = patch("optix.optimizer.pd_disagg.PdDisaggOrchestrator").start()
        self.addCleanup(patch.stopall)
        mock_get_settings.return_value = self._settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        # make_token_type builds CLI choices from the registry during parsing;
        # register_ori_functions is stubbed, so inject default policy tokens to pass validation
        argv = ["optix", "--mode", "pd_disagg"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            main()
        mock_phase_runner.assert_called_once()
        mock_orchestrator.return_value.run.assert_called_once_with()
