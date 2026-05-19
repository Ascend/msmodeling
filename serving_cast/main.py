# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import os

from serving_cast.config import Config
from serving_cast.instance import Instance
from serving_cast.load_gen import FixedLengthLoadGen
from serving_cast.profiler import profiler_interface
from serving_cast.serving import PdAggregationServing, PdDisaggregationServing
from serving_cast.utils import (
    gen_profiling_config_set_env_variable,
    get_basic_timestamp,
    main_processing,
    summarize,
)

import serving_cast.stime as stime


def parse_command_line_args():
    """
    Parse command line arguments for the simulation.

    Expected usage:
    python main.py --instance_config=xx.json,yy.json --common_config=zz.json
    """
    parser = argparse.ArgumentParser(description="Simulation service")

    def validate_file_path(path):
        """Validate that a file path exists"""
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")
        return path

    parser.add_argument(
        "--instance_config_path",
        type=validate_file_path,
        required=True,
        help="Path to a YAML file that declares one or more instance groups. "
        "Each group defines a homogeneous pool of nodes (role, count, TP/DP parallelism) "
        "and can be mixed-and-matched in a single benchmark run.",
    )

    parser.add_argument(
        "--common_config_path",
        type=validate_file_path,
        required=True,
        help="Path to a YAML file with global settings: model architecture, "
        "request-generation workload, and serving limits.",
    )

    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="Enable profiling during simulation (default: False)",
    )

    parser.add_argument(
        "--profiling_output_path",
        default="./profiling_results",
        help="Path to directory where profiling results will be saved (default: ./profiling_results)",
    )

    args = parser.parse_args()

    return args


def instance_group2pd_type(instance_group):
    is_pd_aggregation = (
        len(instance_group["both"]) > 0 and len(instance_group["prefill"]) == 0 and len(instance_group["decode"]) == 0
    )
    is_pd_disaggregation = (
        len(instance_group["both"]) == 0 and len(instance_group["prefill"]) > 0 and len(instance_group["decode"]) > 0
    )

    if is_pd_aggregation and not is_pd_disaggregation:
        return "pd_aggregation"
    elif not is_pd_aggregation and is_pd_disaggregation:
        return "pd_disaggregation"
    else:
        return None


def get_instance_group(instance_config_list, common_config):
    instance_group = {"prefill": [], "decode": [], "both": []}

    for instance_config in instance_config_list:
        for _ in range(instance_config.num_instances):
            instance = Instance(instance_config)
            if instance_config.pd_role not in instance_group:
                raise ValueError(f"{instance_config.pd_role} is not supported")
            else:
                instance_group[instance_config.pd_role].append(instance)

    pd_type = instance_group2pd_type(instance_group)
    if pd_type in ["pd_aggregation", "pd_disaggregation"]:
        return instance_group
    else:
        raise ValueError("check instance's pd_role")


def get_serving(instance_group):
    pd_type = instance_group2pd_type(instance_group)
    if pd_type == "pd_aggregation":
        serving = PdAggregationServing(instance_group["both"])
    elif pd_type == "pd_disaggregation":
        serving = PdDisaggregationServing(instance_group["prefill"], instance_group["decode"])
    else:
        raise ValueError(f"Unknown pd type: {pd_type}")

    return serving


def get_load_gen(load_gen_config):
    if load_gen_config.load_gen_type == "fixed_length":
        load_gen = FixedLengthLoadGen(
            model_name=None,
            num_requests=load_gen_config.num_requests,
            num_input_tokens=load_gen_config.num_input_tokens,
            num_output_tokens=load_gen_config.num_output_tokens,
            request_rate=load_gen_config.request_rate,
        )
        return load_gen
    else:
        raise ValueError(
            f"Unknown load generator type: {load_gen_config.load_gen_type!r}"
        )


def init_profiling(args):
    profiling_path_with_timestamp = os.path.join(args.profiling_output_path, get_basic_timestamp())
    os.makedirs(profiling_path_with_timestamp, exist_ok=True)
    gen_profiling_config_set_env_variable(prof_dir=profiling_path_with_timestamp)
    profiler_interface.init_profiling()
    return profiling_path_with_timestamp


def parse_profiling_results(profiling_path_with_timestamp):
    profiler_interface.parse_profiling_results(profiling_path_with_timestamp)


def main():
    args = parse_command_line_args()
    if args.enable_profiling:
        profiling_path_with_timestamp = init_profiling(args)

    config = Config(parsed_args=args)

    stime.init_simulation()

    instance_group = get_instance_group(config.instance_config_list, config.common_config)
    load_gen = get_load_gen(config.common_config.load_gen)
    serving = get_serving(instance_group)

    _ = stime.CallableTask(main_processing, serving, load_gen)
    stime.start_simulation()

    summarize(load_gen.get_finished_requests().values())

    if args.enable_profiling:
        parse_profiling_results(profiling_path_with_timestamp)

    for pd_type in instance_group:
        for instance in instance_group[pd_type]:
            instance.shutdown()


if __name__ == "__main__":
    main()
