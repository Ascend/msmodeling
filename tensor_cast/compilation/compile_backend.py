import functools
import logging
from typing import Any, Callable, Optional, Sequence

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.freezing import freeze
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized

from .. import config, ops  # noqa: F401
from . import patterns
from .constant_folding import fold_meta_constants
from .freezing_passes import patterns as freezing_patterns
from .freezing_passes.dispatch_ffn_combine_pass import DispatchFFNCombinePass
from .freezing_passes.grouped_matmul_swiglu_pass import GroupedMatmulSwigluPass
from .freezing_passes.sink_split_pass import SinkSplitPass
from .passes.lift_quant_pass import LiftCombineQuantPass
from .passes.merge_linear_pass import MergeLinearPass
from .passes.multistream_pass import MultiStreamSchedulePass
from .passes.peep_hole_pass import PeepHolePass
from .passes.redundant_node_elimination_pass import ReduandantNodeEliminationPass
from .passes.sequence_parallel_pass import SequenceParallelPass


logger = logging.getLogger(__name__)


class CompilerBackend:
    """
    The compilation backend for 'torch.compile'.
    It is used to process the FX graph and perform custom operation fusing etc.
    """

    def __init__(self, device_name: Optional[str] = None):
        self._multistream_device_name = device_name

    def __call__(self, gm: fx.GraphModule, example_inputs) -> Callable:
        """
        Process the FX graph and perform custom operation fusing.

        Args:
            graph (fx.Graph): The FX graph to be processed.
            example_inputs (optional): Example inputs for the graph.

        Returns:
            fx.Graph: The processed FX graph with custom operation fusing applied.
        """
        gm = self.compile(gm, example_inputs)
        return gm

    def compile(
        self,
        gm: fx.GraphModule,
        example_inputs,
        **kwargs,
    ) -> tuple[Callable, Optional[Any]]:
        def freezing_compile(compile_inner, aot_autograd_gm, example_inputs):
            # Freeze the graph first before passing to AOT Autograd.
            frozen_gm, preserved_arg_indices = freeze(
                gm, aot_autograd_gm, example_inputs
            )
            example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
            optimized_function = compile_inner(frozen_gm, example_inputs)

            def wrapper(args: list[object]) -> Sequence[torch.Tensor]:
                args_new = [args[i] for i in preserved_arg_indices]
                args.clear()
                return optimized_function(*args_new)

            wrapper._boxed_call = True  # type: ignore[attr-defined]

            return wrapper

        def graph_rewrite_before_freezing(fx_graph, inputs):
            logger.debug("Graph before compiling:")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(fx_graph.print_readable(print_output=False))
            self.apply_peep_hole_pass(fx_graph, inputs)
            self.apply_redundant_node_elimination_pass(fx_graph, inputs)
            self.apply_quantization_passes(fx_graph, inputs)
            self.apply_pattern_match_passes(fx_graph, inputs)
            self.apply_sequence_parallel_pass(fx_graph, inputs)
            return fx_graph

        def graph_rewrite_after_freezing(fx_graph, inputs):
            self.apply_merge_linear_pass(fx_graph, inputs)
            fold_meta_constants(fx_graph)
            self.apply_redundant_node_elimination_pass(fx_graph, inputs)
            # make sure we add freezing passes after constant folding
            self.apply_freezing_passes(fx_graph, inputs)
            # Run multistream scheduling on the pure-functional graph before
            # decompose_auto_functionalized introduces mutation-style forms.
            # MultiStreamSchedulePass internally invokes DCE, which assumes
            # pure-functional graph semantics.
            self.apply_multistream_pass(fx_graph, inputs)
            self.apply_decompose_auto_functionalized_pass(fx_graph)
            logger.debug("Graph after compiling:")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(fx_graph.print_readable(print_output=False))
            return fx_graph

        def compile_inner(fx_graph, inputs):
            # we split the rewrite into two phases: before and after freezing
            # since freezing would do CSE which might break some assumptions in
            # the rewrite rules.
            graph_rewrite_before_freezing(fx_graph, inputs)
            if config.compilation.enable_freezing:
                return freezing_compile(graph_rewrite_after_freezing, fx_graph, inputs)
            else:
                return graph_rewrite_after_freezing(fx_graph, inputs)

        # Use the default decomposition table to decompose operators.
        decompositions = select_decomp_table()
        # Use AOT Autograd to handle the forward compilation.
        return aot_autograd(
            fw_compiler=compile_inner,
            decompositions=decompositions,
        )(gm, example_inputs)

    def apply_redundant_node_elimination_pass(self, gm: fx.GraphModule, inputs):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="redundant_node_elimination_pass",
            log_url=config.compilation.debug.graph_log_url,
        )
        GraphTransformObserver(gm, "redundant_node_elimination_pass").apply_gm_pass(
            ReduandantNodeEliminationPass()
        )
        logger.debug("Graph after redundant node elimination pass:")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(gm.print_readable(print_output=False))

    def apply_quantization_passes(self, gm: fx.GraphModule, inputs):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="quantization_passes",
            log_url=config.compilation.debug.graph_log_url,
        )
        if config.compilation.passes.enable_life_combine_quant:
            GraphTransformObserver(gm, "life_combine_quant_pass").apply_gm_pass(
                LiftCombineQuantPass()
            )
        logger.debug("Graph after quantization passes:")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(gm.print_readable(print_output=False))

    def apply_pattern_match_passes(self, gm: fx.GraphModule, inputs):
        patterns.lazy_init()
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="pattern_match_passes",
            log_url=config.compilation.debug.graph_log_url,
        )
        for i, pattern_match_pass in enumerate(patterns.all_passes):
            GraphTransformObserver(gm, f"pattern_match_pass_{i}").apply_gm_pass(
                pattern_match_pass
            )
        logger.debug("Graph after pattern matching:")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(gm.print_readable(print_output=False))

    def apply_sequence_parallel_pass(self, gm: fx.GraphModule, inputs):
        """Rewrite TP `all_reduce + norm` patterns into SP
        `reduce_scatter + norm + all_gather`.

        Runs the norm on a seq-dim-sharded tensor so each rank processes only
        1/TP of the tokens, cutting norm compute/memory; the trade-off is an
        all_gather after the norm to restore the full sequence. Must run before
        freezing, otherwise CSE will break the all_reduce→norm match chain.
        """
        # GraphTransformObserver dumps per-pass rewrite results to `log_url`
        # for fusion-pass debugging; `subsystem` tags the log group.
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="sequence_parallel_pass",
            log_url=config.compilation.debug.graph_log_url,
        )
        # Config-gated: skip when SP is not enabled (e.g. TP=1 or SP disabled).
        if config.compilation.passes.enable_sequence_parallel:
            # SequenceParallelPass rewrites in P1/P2/P3 order (see that pass's
            # module docstring):
            #   P1: all_reduce → rms_norm / add_rms_norm
            #   P2: all_reduce → add_rms_norm2 (residual-aware)
            #   P3: add → norm on the residual path P2 leaves local
            GraphTransformObserver(gm, "sequence_parallel_pass").apply_gm_pass(
                SequenceParallelPass()
            )
            logger.debug("Graph after sequence parallel pass:")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(gm.print_readable(print_output=False))

    def apply_decompose_auto_functionalized_pass(self, gm: fx.GraphModule):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="decompose_auto_functionalized_pass",
            log_url=config.compilation.debug.graph_log_url,
        )
        GraphTransformObserver(gm, "decompose_auto_functionalized").apply_graph_pass(
            decompose_auto_functionalized
        )

    def apply_merge_linear_pass(self, gm: fx.GraphModule, inputs):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="merge_linear_pass",
            log_url=config.compilation.debug.graph_log_url,
        )
        if config.compilation.passes.enable_merge_linear:
            GraphTransformObserver(gm, "merge_linear_pass").apply_gm_pass(
                MergeLinearPass()
            )
            # TODO(jgong): make sure the merge linear pass is correct by shape propagation
            #              since explicitly adding shape info might be expensive
            fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
            logger.debug("Graph after the merge linear pass:")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(gm.print_readable(print_output=False))

    def apply_freezing_passes(self, gm: fx.GraphModule, inputs):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="freezing_passes",
            log_url=config.compilation.debug.graph_log_url,
        )
        if config.compilation.passes.enable_sink_split:
            GraphTransformObserver(gm, "sink_split_pass").apply_gm_pass(SinkSplitPass())
            # TODO(jgong): make sure the sink split pass is correct by shape propagation
            #              since explicitly adding shape info might be expensive
            fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
        if config.compilation.fusion_patterns.enable_matmul_allreduce:
            self.apply_freezing_pattern_passes(gm, inputs)
        if config.compilation.fusion_patterns.enable_grouped_matmul_swiglu:
            GraphTransformObserver(
                gm, "grouped_matmul_swiglu_fusion_pass"
            ).apply_gm_pass(GroupedMatmulSwigluPass())
            fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
        # DFC must run AFTER GroupedMatmulSwigluPass — it looks for
        # grouped_matmul_*_swiglu nodes that the swiglu pass produces.
        if config.compilation.fusion_patterns.enable_dispatch_ffn_combine:
            GraphTransformObserver(
                gm, "dispatch_ffn_combine_fusion_pass"
            ).apply_gm_pass(DispatchFFNCombinePass())
            fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
        logger.debug("Graph after freezing passes:")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(gm.print_readable(print_output=False))

    def apply_freezing_pattern_passes(self, gm: fx.GraphModule, inputs):
        freezing_patterns.lazy_init()
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="freezing_pattern_passes",
            log_url=config.compilation.debug.graph_log_url,
        )
        for i, freezing_pattern_pass in enumerate(freezing_patterns.all_passes):
            GraphTransformObserver(gm, f"freezing_pattern_pass_{i}").apply_gm_pass(
                freezing_pattern_pass
            )
            fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)

    def apply_peep_hole_pass(self, gm: fx.GraphModule, inputs):
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="peep_hole_pass",
            log_url=config.compilation.debug.graph_log_url,
        )
        GraphTransformObserver(gm, "peep_hole_pass").apply_gm_pass(PeepHolePass())
        logger.debug("Graph after peep hole pass:")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(gm.print_readable(print_output=False))

    def apply_multistream_pass(self, gm: fx.GraphModule, inputs):
        if not config.compilation.multistream.enable:
            return
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="multistream_pass",
            log_url=config.compilation.debug.graph_log_url,
        )
        fake_tensor_prop(gm, inputs, force_allow_non_fake_inputs=True)
        GraphTransformObserver(gm, "multistream_pass").apply_gm_pass(
            MultiStreamSchedulePass(device_name=self._multistream_device_name)
        )
        logger.debug("Graph after multistream pass:")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(gm.print_readable(print_output=False))
