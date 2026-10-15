# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Minimum sampling gates work with scalar and structured candidate values."""

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from optix.config.config import AgentOptimizerConfig, OptimizerConfigField, PerformanceIndex
from optix.optimizer.agentic.coverage import search_space_coverage
from optix.optimizer.agentic.orchestrator import AgentOptimizer


@pytest.mark.parametrize('min_rounds,min_trials,expected', [(2, 3, True), (3, 3, False), (2, 4, False)])
def test_round_and_trial_minima_with_structured_params(tmp_path, min_rounds, min_trials, expected):
    for round_id, params in [
        (1, [{'cudagraph_capture_sizes': [1, 4]}, {'structured_param': {'enabled': True}}]),
        (2, [{'tp': 8}]),
    ]:
        trials = [
            {'params': value, 'metadata': {'round_id': round_id}, 'error': None, 'fitness': 1.0} for value in params
        ]
        (tmp_path / f'results.round-{round_id}.json').write_text(json.dumps({'trials': trials}), encoding='utf-8')

    coverage = search_space_coverage(tmp_path, min_rounds=min_rounds, min_trials=min_trials)

    assert coverage['rounds_done'] == 2
    assert coverage['trials_done'] == 3
    assert coverage['ok'] is expected
    assert bool(coverage['reasons']) is not expected


def test_empty_run_does_not_satisfy_minimum_coverage(tmp_path):
    coverage = search_space_coverage(tmp_path, min_rounds=1, min_trials=1)
    assert coverage['ok'] is False
    assert coverage['rounds_done'] == coverage['trials_done'] == 0
    assert len(coverage['reasons']) == 2


@pytest.mark.parametrize(
    'failed_trial',
    [
        {'params': {'tp': 8}, 'error': 'oom', 'fitness': 1.0},
        {'params': {'tp': 8}, 'error': None, 'fitness': float('inf')},
        {'params': {'tp': 8}, 'error': None},
    ],
)
def test_failed_trials_do_not_count_towards_coverage(tmp_path, failed_trial):
    """失败 trial（error 非空 / fitness 非有限）与 orchestrator 完成轮口径一致，不计入覆盖。"""
    good = {'params': {'tp': 4}, 'error': None, 'fitness': 1.0}
    failed = {**failed_trial, 'metadata': {'round_id': 2}}
    (tmp_path / 'results.round-1.json').write_text(json.dumps({'trials': [good]}), encoding='utf-8')
    (tmp_path / 'results.round-2.json').write_text(json.dumps({'trials': [failed]}), encoding='utf-8')

    coverage = search_space_coverage(tmp_path, min_rounds=2, min_trials=2)

    assert coverage['rounds_done'] == 1
    assert coverage['trials_done'] == 1
    assert coverage['ok'] is False


@pytest.mark.parametrize('context', [{}, {'search_space': None}, {'search_space': ['not', 'a', 'dict']}])
def test_search_space_context_missing_or_malformed_raises(tmp_path, context):
    (tmp_path / 'context.json').write_text(json.dumps(context), encoding='utf-8')
    optimizer = AgentOptimizer(pso=Mock(), agent_config=Mock(), run_dir=tmp_path)
    with pytest.raises(ValueError, match='skill workflow'):
        optimizer._load_search_space_context()


def test_missing_search_space_context_file_raises(tmp_path):
    optimizer = AgentOptimizer(pso=Mock(), agent_config=Mock(), run_dir=tmp_path)
    with pytest.raises(ValueError, match='skill workflow'):
        optimizer._load_search_space_context()


def test_search_space_context_loads_when_well_formed(tmp_path):
    (tmp_path / 'context.json').write_text(
        json.dumps({'search_space': {'parameters': [], 'constants': []}}), encoding='utf-8'
    )
    context = AgentOptimizer(pso=Mock(), agent_config=Mock(), run_dir=tmp_path)._load_search_space_context()
    assert context['search_space'] == {'parameters': [], 'constants': []}


def test_orchestrator_applies_coverage_gate_to_array_candidates(tmp_path):
    config = tmp_path / 'config.toml'
    config.write_text('[vllm.command]\nothers=""\n', encoding='utf-8')
    context = {
        'engine': 'vllm',
        'source': {'config_toml': str(config)},
        'constraints': {'world_size': 1, 'expressions': ['$TP==$WORLD_SIZE']},
        'search_space': {
            'parameters': [
                {'name': 'tp', 'dtype': 'int', 'min': 1, 'max': 1, 'default': 1, 'config_position': 'run'},
                {
                    'name': 'cudagraph_capture_sizes',
                    'dtype': 'enum',
                    'min': 0,
                    'max': 1,
                    'choices': [[1, 4]],
                    'default': [1, 4],
                    'config_position': 'run',
                },
            ],
            'constants': [],
        },
    }
    fields = [OptimizerConfigField(name='tp', dtype='int', value=1, min=1, max=1, config_position='run')]
    for round_id in range(1, 5):
        candidate = {
            'candidate_id': f'r{round_id}-a',
            'params': {
                'cudagraph_capture_sizes': [1, 4],
            },
        }
        (tmp_path / f'candidates.round-{round_id}.json').write_text(
            json.dumps({'round': round_id, 'candidates': [candidate]}), encoding='utf-8'
        )
    scheduler = Mock(error_info='', last_outcome=None)
    scheduler.disable_early_exit.return_value = contextlib.nullcontext()
    scheduler.run.return_value = PerformanceIndex(
        generate_speed=100, time_to_first_token=0.5, time_per_output_token=0.02, success_rate=1
    )
    evaluator = SimpleNamespace(scheduler=scheduler, minimum_algorithm=lambda perf: 1.0, ttft_slo=1, tpot_slo=0.1)
    agent_config = AgentOptimizerConfig(max_rounds=4, convergence_rounds=2, min_rounds=3, min_trials=3)

    AgentOptimizer(pso=evaluator, agent_config=agent_config, run_dir=tmp_path)._run_rounds(
        set(), float('inf'), fields, context
    )

    assert scheduler.run.call_count == 3
    assert not (tmp_path / 'results.round-4.json').exists()
    result = json.loads((tmp_path / 'results.round-3.json').read_text(encoding='utf-8'))
    assert result['trials'][0]['params']['cudagraph_capture_sizes'] == [1, 4]
