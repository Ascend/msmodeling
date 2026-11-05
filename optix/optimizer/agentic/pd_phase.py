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


"""One agent-proposed P/D probe, isolated from the complete-PD experiment."""

import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .pd import ROLES, positive_integer
from .pd_state import check_workflow_budget, finish_interrupted_attempt, workflow_time_limit


def verify_phase_shutdown(record):
    """Confirm local launch process groups and the service listener have exited."""
    identity = record.get("cleanup_identity") or {}
    if (
        identity.get("host") != socket.gethostname()
        or identity.get("uid") != os.getuid()
        or identity.get("netns") != os.stat("/proc/self/ns/net").st_ino
        or not isinstance(identity.get("groups"), list)
    ):
        raise ValueError("Phase cleanup must be verified in its original host, user and network namespace")
    port = identity.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Phase cleanup requires a recorded service port")
    if identity.get("launch_pending"):
        raise ValueError("Phase launch identity was not recorded; inspect resources manually")
    for group in identity["groups"]:
        if type(group) is not int or group <= 0:
            raise ValueError("Invalid phase process group")
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            continue
        except OSError as error:
            raise ValueError("Phase process absence cannot be verified") from error
        raise ValueError(f"Phase process group is still alive: {group}")
    try:
        ss = shutil.which("ss")
        if not ss:
            raise ValueError("Phase endpoint absence cannot be verified: ss is unavailable")
        result = subprocess.run(
            [ss, "-H", "-ltn", f"sport = :{port}"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Phase endpoint absence cannot be verified") from error
    if result.returncode or result.stdout.strip():
        raise ValueError("Phase endpoint shutdown is unconfirmed")
    record["cleanup_confirmed"] = True


class PhaseLaunch:
    """Hold execution until its isolated process group is durably recorded."""

    def __init__(self, record, path):
        self.record, self.path = record, path
        self.gate = None

    def prepare(self, command):
        self.gate = self.path.parent / ("launch-" + uuid.uuid4().hex)
        self.record["cleanup_identity"]["launch_pending"] = True
        self._save()
        code = (
            "import os,sys,time\n"
            "gate=sys.argv[1]\n"
            "deadline=time.monotonic()+30\n"
            "while not os.path.exists(gate):\n"
            " if time.monotonic()>deadline: sys.exit(1)\n"
            " time.sleep(.05)\n"
            "os.execvpe(sys.argv[2],sys.argv[2:],os.environ)\n"
        )
        return [sys.executable, "-c", code, str(self.gate), *command]

    def started(self, process):
        identity = self.record["cleanup_identity"]
        identity["groups"].append(process.pid)
        identity["launch_pending"] = False
        self._save()
        self.gate.touch(exist_ok=False)

    def _save(self):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def recover_phase_attempts(root):
    for path in (Path(root) / "pd_tasks").glob("*/attempt.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("cleanup_identity"):
            raise ValueError(f"Phase cleanup identity is missing: {path.parent.name}; inspect the old attempt manually")
        if not record.get("cleanup_confirmed"):
            verify_phase_shutdown(record)
        if "finished_at" not in record:
            finish_interrupted_attempt(record)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def capacity_key(binding):
    """Bind capacity to actual phase config/resource/parameter/protocol facts."""
    return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()


def matching_capacity(record, binding):
    return (
        bool(binding.get("software"))
        and bool(binding.get("hardware"))
        and record.get("outcome") == "measured"
        and record.get("capacity_key") == capacity_key(binding)
        and isinstance(record.get("qps"), (int, float))
        and math.isfinite(record["qps"])
        and record["qps"] > 0
    )


def _option(tokens, name):
    values = []
    for index, token in enumerate(tokens):
        if token == name:
            values.append(tokens[index + 1] if index + 1 < len(tokens) else None)
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1:
        raise ValueError(f"Phase measurement requires exactly one {name}")
    return values[0]


def validate_protocol(role, phase, params):
    """First version accepts explicit reproducible vLLM random benchmark probes."""
    if role not in ROLES or phase.engine != "vllm" or phase.benchmark_policy != "vllm_benchmark":
        raise ValueError("Agent phase probes currently require vllm + vllm_benchmark")
    service = shlex.split(phase.simulator_command_overrides.get("others", ""))
    benchmark = shlex.split(phase.benchmark_command_overrides.get("others", ""))
    enabled = "--enable-prefix-caching" in service
    disabled = "--no-enable-prefix-caching" in service
    if role == "prefill" and (not disabled or enabled):
        raise ValueError("Prefill probe must explicitly disable prefix caching")
    if role == "decode" and (not enabled or disabled or phase.benchmark_run_count < 2):
        raise ValueError("Decode probe requires prefix caching and warmup before scoring")
    output = positive_integer(_option(benchmark, "--random-output-len"), "phase output length")
    if role == "prefill" and output != 1:
        raise ValueError("Prefill probe output must be one token")
    if role == "decode" and output <= 1:
        raise ValueError("Decode probe needs multiple output tokens")
    input_length = positive_integer(_option(benchmark, "--random-input-len"), "phase input length")
    requests = positive_integer(_option(benchmark, "--num-prompts"), "phase request count")
    seed = _option(benchmark, "--seed")
    int(seed)
    # The protocol must not be overridden by a candidate field.
    from .pd import OWNED_ENV

    for name in params:
        if name.upper() in OWNED_ENV or name.lower().replace("_", "-") in {
            "model",
            "host",
            "port",
            "served-model-name",
            "kv-transfer-config",
        }:
            raise ValueError("Phase device/network assignment cannot be overridden by candidates")
        normalized = name.lower().replace("_", "-")
        if any(
            key in normalized
            for key in (
                "prefix-caching",
                "random-input",
                "random-output",
                "seed",
                "num-prompts",
            )
        ):
            raise ValueError("Phase protocol fields are fixed, not agent search dimensions")
    return {
        "role": role,
        "input_tokens": input_length,
        "output_tokens": output,
        "requests": requests,
        "seed": seed,
        "benchmark_run_count": phase.benchmark_run_count,
        "capacity_kind": "short-output proxy" if role == "prefill" else "warm-prefix-cache proxy",
    }


def run_phase_task(root, task_id, role, resource, candidate, context, phase, runner):
    """Execute on the assigned device host using existing local OptiX plugins.

    The coordinator schedules this entry on that host; root must be the shared
    experiment directory. It never overwrites the parent's context/baseline.
    """
    from ..pd_disagg import _phase_settings
    from ..scheduler import Scheduler
    from ..store import DataStorage
    from ..performance_tunner import PerformanceTuner
    from ...config.config import DataStorageConfig
    from .strategies import AgentCandidateStrategy
    from .validation import validate_candidate

    root = Path(root)
    if not task_id or Path(task_id).name != task_id or task_id in {".", ".."}:
        raise ValueError("Phase task ID must be a filename component")
    experiment = json.loads((root / "pd_experiment.json").read_text(encoding="utf-8"))
    allowed = {tuple(d) for d in experiment["deployment"]["device_inventory"]}
    cards = resource["gpu_ids"]
    if not cards or len(set(cards)) != len(cards) or not {(resource["ssh_ip"], d) for d in cards} <= allowed:
        raise ValueError("Phase resources exceed the frozen device inventory")
    local_addresses = {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}
    local_addresses.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None))
    if resource["ssh_ip"] not in local_addresses:
        raise ValueError("Run the phase task on its assigned device host with the shared run directory")
    if context.get("deployment", {}).get("type") == "pd":
        raise ValueError("Phase task requires a separate single-service context")
    if context.get("model") != experiment.get("model") or context.get("model_info") != experiment.get("model_info"):
        raise ValueError("Phase model differs from the main experiment")
    if not runner.settings.manage_simulator_lifecycle or runner.settings.use_request_rate_calibration:
        raise ValueError("Phase tasks require managed lifecycle and no request-rate calibration")
    validation = validate_candidate(context, candidate)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    protocol = validate_protocol(role, phase, validation["params"])
    model = phase.simulator_command_overrides.get("model", runner.settings.vllm.command.model)
    if str(model) != str(experiment["model"].get("path")):
        raise ValueError("Actual phase service model differs from the experiment")
    dataset = phase.benchmark_command_overrides.get("dataset_name", runner.settings.vllm_benchmark.command.dataset_name)
    if dataset != "random":
        raise ValueError("Phase protocol currently requires the random benchmark dataset")
    workload = experiment.get("benchmark_spec", {})
    if protocol["input_tokens"] != workload.get("input_len_avg"):
        raise ValueError("Phase input length must match the frozen business workload")
    if role == "decode" and protocol["output_tokens"] != workload.get("output_len_avg"):
        raise ValueError("Decode output length must match the frozen business workload")
    if not (root / "baseline.json").exists() or not (root / "pd_instance_specs.json").exists():
        raise ValueError("Establish the business baseline and freeze instance specs before phase probing")
    specification = json.loads((root / "pd_instance_specs.json").read_text(encoding="utf-8"))[role]
    service_tokens = shlex.split(phase.simulator_command_overrides.get("others", ""))
    tp = positive_integer(_option(service_tokens, "--tensor-parallel-size"), "phase TP")
    if specification["dp"] != 1 or specification["tp"] != tp:
        raise ValueError("Phase probing requires the frozen host-local TP/DP=1 instance; otherwise skip probing")
    if tp != len(cards) or any("parallel" in name.lower() for name in validation["params"]):
        raise ValueError("Phase TP must match assigned cards; parallelism is fixed during probing")
    for token in service_tokens:
        if token.split("=", 1)[0] in {
            "--data-parallel-size",
            "--pipeline-parallel-size",
            "--prefill-context-parallel-size",
        }:
            raise ValueError("First-phase probes use one host-local TP instance only")
    # Reassignment is serial; verify all earlier full-PD launches have stopped.
    from ..pd_adapter import verify_saved_shutdown
    from .pd_state import read_attempts

    task_dir = root / "pd_tasks" / task_id
    if task_dir.exists():
        raise ValueError("Phase attempt IDs are immutable; use a new ID for retries")
    # The same controller lock also serializes phase tasks and config generation.
    from .pd import controller_session

    with controller_session(root):
        recover_phase_attempts(root)
        for record in read_attempts(root):
            if not record.get("cleanup_confirmed"):
                verify_saved_shutdown(record)
            if "finished_at" not in record:
                finish_interrupted_attempt(record)
            path = Path(record["archive"]) / "deployment.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        check_workflow_budget(root, role + "_probe")
        simulator_cls, benchmark_cls = runner._resolve_and_validate_plugins(role, phase)
        deploy_env = {
            **runner.deploy_env,
            "ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, cards)),
        }
        simulator = simulator_cls(bak_path=task_dir, runtime_ctx=runner.runtime_ctx, deploy_env=deploy_env)
        benchmark = benchmark_cls(bak_path=task_dir, runtime_ctx=runner.runtime_ctx, deploy_env=deploy_env)
        from ..plugins.simulate import VllmSimulator

        if type(simulator) is not VllmSimulator:
            raise ValueError("Phase cleanup verification requires the built-in local vllm simulator")
        runner._apply_simulator_command_overrides(simulator, role, phase)
        runner._apply_benchmark_command_overrides(benchmark, role, phase)
        runner._configure_phase_hook(simulator, role, phase)
        runner._configure_phase_hook(benchmark, role, phase)
        service_command = getattr(getattr(simulator, "config", None), "command", None)
        benchmark_command = getattr(getattr(benchmark, "config", None), "command", None)
        if service_command is not None and benchmark_command is not None:
            if str(service_command.port) != str(benchmark_command.port):
                raise ValueError("Phase benchmark endpoint must match the launched service")
        try:
            service_port = int(service_command.port)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Phase cleanup requires a configured service port") from error
        if not 1 <= service_port <= 65535:
            raise ValueError("Invalid phase service port")
        fields = runner._resolve_target_fields(simulator, benchmark, phase)
        binding = {
            "experiment_sha256": hashlib.sha256((root / "pd_experiment.json").read_bytes()).hexdigest(),
            "model": experiment["model"],
            "model_info": experiment.get("model_info"),
            "resource": resource,
            "params": validation["params"],
            "protocol": protocol,
            "phase_config": phase.model_dump(mode="json"),
            "software": context.get("software"),
            "hardware": context.get("hardware"),
        }
        task_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "purpose": role + "_probe",
            "started_at": time.time(),
            "outcome": "started",
            "binding": binding,
            "capacity_key": capacity_key(binding),
            "cleanup_confirmed": False,
            "cleanup_identity": {
                "host": socket.gethostname(),
                "uid": os.getuid(),
                "netns": os.stat("/proc/self/ns/net").st_ino,
                "port": service_port,
                "groups": [],
                "launch_pending": False,
            },
        }
        path = task_dir / "attempt.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        simulator._optix_phase_launch = PhaseLaunch(record, path)
        (task_dir / "context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
        (task_dir / "candidate.json").write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        storage = DataStorage(DataStorageConfig(store_dir=task_dir, pso_top_k=0), simulator, benchmark)
        scheduler = Scheduler(
            simulator,
            benchmark,
            storage,
            bak_path=task_dir,
            wait_start_time=runner.settings.wait_start_time,
            engine="vllm",
            benchmark_run_count=phase.benchmark_run_count,
        )
        scheduler.pd_workflow_root = root
        scheduler.pd_trial_purpose = role + "_probe"
        evaluator = PerformanceTuner(
            ttft_penalty=phase.ttft_penalty,
            tpot_penalty=phase.tpot_penalty,
            success_rate_penalty=phase.success_rate_penalty,
            ttft_slo=phase.ttft_slo,
            tpot_slo=phase.tpot_slo,
            success_rate_slo=phase.success_rate_slo,
            generate_speed_target=runner.settings.generate_speed_target,
        )
        strategy = AgentCandidateStrategy(task_dir, max_trials=1, fitness_evaluator=evaluator)
        try:
            with (
                workflow_time_limit(root, role + "_probe"),
                _phase_settings(runner.settings, phase),
            ):
                trial = strategy._run_trial(validation, scheduler, fields, context)
            record["trial"] = trial.to_dict()
            record["benchmark_pass"] = getattr(scheduler, "benchmark_pass", None)
            if trial.error or not math.isfinite(trial.fitness):
                raise ValueError(trial.error or "Phase evaluation has no finite score")
            qps = float(trial.performance.throughput)
            if not math.isfinite(qps) or qps <= 0:
                raise ValueError("Phase throughput must be finite and positive")
            record.update(outcome="measured", qps=qps)
        except Exception as error:
            record.update(outcome="failed", error=str(error))
            raise
        finally:
            try:
                scheduler.stop_target_server()
                verify_phase_shutdown(record)
            finally:
                record["finished_at"] = time.time()
                path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


def current_capacity_hint(record, experiment_sha256, role, params, common_options=""):
    """Use a current experiment's measured probe only when effective service inputs match.

    This is an approximate ranking hint, not a cross-experiment cache hit.
    """
    from ...config.base_config import CONCURRENCYS, REQUESTRATES

    binding = record.get("binding", {})
    qps = record.get("qps")
    if (
        record.get("outcome") != "measured"
        or not record.get("cleanup_confirmed")
        or binding.get("experiment_sha256") != experiment_sha256
        or binding.get("protocol", {}).get("role") != role
        or not isinstance(qps, (int, float))
        or not math.isfinite(qps)
        or qps <= 0
    ):
        return None
    if params.get("DATA_PARALLEL_SIZE_" + role, 1) != 1:
        return None
    if len(binding.get("resource", {}).get("gpu_ids", [])) != params.get("TENSOR_PARALLEL_SIZE_" + role):
        return None
    from .pd import merge_role_params, TOPOLOGY_FIELDS, FIXED_PARALLEL_FIELDS

    ignored = {*CONCURRENCYS, *REQUESTRATES, *TOPOLOGY_FIELDS, *FIXED_PARALLEL_FIELDS}
    effective = merge_role_params(params, role, excluded=ignored)
    measured = {name: value for name, value in binding.get("params", {}).items() if name not in ignored}
    if effective != measured:
        return None
    tokens = shlex.split(binding.get("phase_config", {}).get("simulator_command_overrides", {}).get("others", ""))
    static = []
    skip = False
    for argument in tokens:
        if skip:
            skip = False
            continue
        if argument == "--tensor-parallel-size":
            skip = True
        elif argument.startswith("--tensor-parallel-size=") or argument in {
            "--enable-prefix-caching",
            "--no-enable-prefix-caching",
        }:
            continue
        else:
            static.append(argument)
    if static != shlex.split(common_options):
        return None
    return qps
