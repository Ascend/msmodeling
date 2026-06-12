from typing import Optional

import torch


# TODO(jgong5): add meaning for each configuration item
class performance_model:
    class empirical:
        runtime_device_override: Optional[torch.device] = None
        warmup_runs = 1
        benchmark_runs = 10


class compilation:
    enable_freezing = True

    class multistream:
        enable = True
        # Backward-compatible aliases; scheduling logic should read role_to_stream_ids.
        compute_stream_id = 0
        comm_stream_id = 1
        # Role-based stream assignment. Keep defaults to 2 lanes while allowing future expansion.
        role_to_stream_ids = {
            "compute": (compute_stream_id,),
            "comm": (comm_stream_id,),
        }
        cross_stream_sync_overhead_s = 0.0
        # Enable analytic estimator for schedulable FX nodes when metadata is available.
        enable_analytic_cost_model = True

    class passes:
        enable_life_combine_quant = True
        enable_merge_linear = True
        enable_sink_split = True
        enable_sequence_parallel = False

    class fusion_patterns:
        enable_rms_norm = True
        enable_rms_norm_quant = enable_rms_norm
        enable_add_rms_norm = enable_rms_norm
        enable_rope = True
        enable_swiglu = True
        enable_matmul_allreduce = True
        enable_grouped_matmul_swiglu = True
        enable_dispatch_ffn_combine = False

    class debug:
        graph_log_url = None
