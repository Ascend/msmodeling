"""Thin, resumable CLI orchestration; all graph inference stays in the project."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import queue
import threading

REPO = Path(__file__).resolve().parents[4]
PROFILE = Path(__file__).resolve().parents[1] / 'references' / 'scope.json'
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_path(path):
    absolute = str(Path(path).resolve())
    if os.name == 'nt' and not absolute.startswith('\\\\?\\'):
        absolute = '\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\' + absolute
    return Path(absolute)


def epoch():
    git = shutil.which('git')
    if git is None:
        raise ValueError('Git executable is required for source provenance')
    files = subprocess.check_output(
        [
            git,
            'ls-files',
            '--cached',
            '--others',
            '--exclude-standard',
            '--',
            'tensor_cast',
            'template_engine',
            'serving_cast',
            'cli',
            'template_generate.py',
        ],
        cwd=REPO,
        text=True,
    ).splitlines()
    return {
        'head': subprocess.check_output([git, 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'files': {
            f: sha(REPO / f)
            for f in files
            if Path(f).suffix in ('.py', '.yaml', '.json', '.toml') and (REPO / f).is_file()
        },
        'python': sys.version,
        'python_executable': sys.executable,
        'torch': importlib.metadata.version('torch'),
        'transformers': importlib.metadata.version('transformers'),
        'workflow_sha256': sha(__file__),
    }


def request_shape(input_length, output_length, decode, mtp=0):
    """Use the optimizer's own representative forward-shape contract."""
    from types import SimpleNamespace
    from serving_cast.service.base_throughput_optimizer import BaseThroughputOptimizer
    from serving_cast.service.utils import OptimizerData

    if input_length < 1 or output_length < 1:
        raise ValueError('Request input/output lengths must be positive')
    data = OptimizerData(input_length=input_length, output_length=output_length)
    return BaseThroughputOptimizer._resolve_forward_shape(SimpleNamespace(num_mtp_tokens=mtp), data, decode)


def plan(args):
    from tensor_cast.model_config import ParallelConfig
    from tensor_cast.core.quantization.datatypes import QuantizeLinearAction, QuantizeAttentionAction
    from template_engine.cache_io import build_key_and_config

    from tensor_cast.core.model_source_security import normalize_model_source

    args.model_id = normalize_model_source(args.model_id, 'huggingface', warn_remote=False).model_id
    root = run_path(args.run_dir)
    root.mkdir(parents=True, exist_ok=False)
    scope = read(args.scope or PROFILE)
    input_length, output_length = getattr(args, 'input_length', None), getattr(args, 'output_length', None)
    if (input_length is None) != (output_length is None):
        raise ValueError('Specify --input-length and --output-length together')
    if input_length is not None:
        request_shape(input_length, output_length, False)
        scope['sampling']['long_request'] = {'input_length': input_length, 'output_length': output_length}
    if getattr(args, 'decode_batch_coverage', None):
        scope['sampling']['decode_batch_coverage'] = args.decode_batch_coverage
    requested = scope['requested']
    if requested['PP'] != 1 or requested['compile'] is not True:
        raise ValueError('This template workflow requires PP=1 and compile=true')
    if args.model_config:
        raw = read(args.model_config)
    else:
        from tensor_cast.transformers.utils import AutoModelConfigLoader

        resolved = AutoModelConfigLoader().load_config(args.model_id)
        if resolved is None:
            raise ValueError('Official MS model configuration resolver returned no config')
        raw = resolved.to_dict()
    model = raw.get('text_config', raw)
    layers = int(model.get('num_hidden_layers', model.get('num_layers', 0)))
    if layers <= 0:
        raise ValueError('Cannot resolve native model depth; inspect the official config before planning')
    experts = max(
        int(model.get(k, 0) or 0) for k in ('num_experts', 'n_routed_experts', 'num_local_experts', 'moe_num_experts')
    )
    cap = min(requested['S'][1], int(model.get('max_position_embeddings', requested['S'][1]) or requested['S'][1]))
    if requested['S'][0] > cap:
        raise ValueError('Requested S range has no intersection with the model context limit')
    cells, rejected = {}, []
    linear_actions = [
        str(action)
        for action in QuantizeLinearAction
        if any(str(action) == family or str(action).startswith(family + '_') for family in requested['linear_families'])
    ]
    attention_actions = requested['attention_actions']
    default_attention = (
        attention_actions[0] if isinstance(attention_actions, list) and attention_actions else 'DISABLED'
    )
    if not linear_actions:
        raise ValueError('Requested linear quantization has no official CLI action')

    def add(layout, decode=False, dfc=False, mtp=0, linear=None, attention=None):
        linear = linear_actions[0] if linear is None else linear
        attention = default_attention if attention is None else attention
        tp, ep, md, dcp = layout
        try:
            phase = 'decode' if decode else 'prefill'
            if (
                phase not in requested['phase']
                or (decode and mtp not in requested['MTP'])
                or dfc not in requested['DFC']
            ):
                return
            if not decode and mtp:
                return
            if not any(
                str(linear) == family or str(linear).startswith(family + '_') for family in requested['linear_families']
            ):
                return
            if (
                isinstance(requested['attention_actions'], list)
                and str(attention) not in requested['attention_actions']
            ):
                return
            if not decode and dcp != 1:
                return
            world = max(tp, ep, dcp) * md
            pc = ParallelConfig(
                world_size=world,
                tensor_parallel_size=tp,
                expert_parallel_size=ep,
                moe_data_parallel_size=md,
                decode_context_parallel_size=dcp,
            )
            if not scope['requested']['world_size'][0] <= world <= scope['requested']['world_size'][1]:
                raise ValueError('world outside requested domain')
            if experts and not (ep >= tp and ep > 1 and ep >= pc.data_parallel_size):
                raise ValueError('outside inherited MoE EP constraints')
            config = dict(
                TP=tp,
                EP=ep,
                MOE_DP=md,
                DCP=dcp,
                B=1,
                S=1,
                do_compile=requested['compile'],
                is_decode=decode,
                num_mtp_tokens=mtp,
                enable_dispatch_ffn_combine=dfc,
                quantize_linear_action=str(linear),
                quantize_attention_action=str(attention),
            )
            key, _ = build_key_and_config(config)
            cells.setdefault(
                key,
                {'config': config, 'world': world, 'DP': pc.data_parallel_size, 'MOE_TP': pc.moe_tensor_parallel_size},
            )
        except (ValueError, AssertionError) as exc:
            rejected.append({'layout': layout, 'decode': decode, 'reason': str(exc)})

    layouts = scope['typical_moe_layouts' if experts else 'typical_dense_layouts']
    for layout in layouts:
        for decode in (False, True):
            add(layout, decode)
    representative = layouts[1] if len(layouts) > 1 else layouts[0]
    for decode in (False, True):
        if experts:
            for layout in (layouts[0], representative, layouts[max(0, len(layouts) - 2)]):
                add(layout, decode, dfc=True)
        for action in QuantizeLinearAction:
            add(representative, decode, linear=action)
        for action in QuantizeAttentionAction:
            add(representative, decode, attention=action)
        for mtp in scope['requested']['MTP']:
            add(representative, decode, mtp=mtp)
            if experts:
                add(representative, decode, dfc=True, mtp=mtp)
        if experts:
            for action in ('W8A8_DYNAMIC', 'MXFP4'):
                if action in {str(a) for a in QuantizeLinearAction}:
                    add(representative, decode, dfc=True, linear=action)
    result = []
    for index, cell in enumerate(cells.values()):
        config, dp = cell['config'], cell['DP']
        sampling = scope['sampling']

        def sequence_point(value):
            return min(max(int(value), requested['S'][0]), cap)

        def batch_point(value):
            return min(max(int(value), requested['B'][0]), requested['B'][1])

        lo, hi = [sequence_point(s) for s in sampling['S_train']]
        if lo >= hi:
            lo = max(requested['S'][0], hi // 2)
        if lo >= hi:
            raise ValueError('Typical two-axis sampling needs two legal S values; adapt the plan for a fixed-S request')
        bs = [batch_point(value) for value in (dp, dp + 1, 2 * dp + 1, 3 * dp + 1)]
        points = [
            (bs[0], lo, 'train'),
            (bs[0], hi, 'train'),
            (bs[1], lo, 'train'),
            (bs[1], hi, 'train'),
            (bs[2], hi, 'train'),
            (batch_point(dp + 2), sequence_point(sampling['S_development']), 'development'),
            (bs[2], sequence_point(sampling['S_holdout']), 'holdout'),
        ]
        if config['is_decode']:
            first_k = (requested['B'][0] + dp - 1) // dp
            last_k = (requested['B'][1] + dp - 1) // dp
            mode = sampling.get('decode_batch_coverage', 'typical')
            if mode not in ('typical', 'all'):
                raise ValueError('decode_batch_coverage must be typical or all')
            batches = (
                list(range(first_k, last_k + 1)) if mode == 'all' else sorted({(b + dp - 1) // dp for b in bs[:3]})
            )
            cell['decode_local_batches'] = batches
            cell['unplanned_decode_local_batches'] = sorted(set(range(first_k, last_k + 1)) - set(batches))
            points = []
            for local in batches:
                batch = batch_point(local * dp)
                points.extend((batch, s, 'train') for s in (lo, hi))
                points.append((batch, sequence_point(sampling['S_holdout']), 'holdout'))
            points.append((batch_point(batches[0] * dp), sequence_point(sampling['S_development']), 'development'))
        elif requested['S'][0] <= sampling['long_S_train'] <= cap:
            points.insert(5, (bs[0], sampling['long_S_train'], 'train'))
        if not config['is_decode']:
            for sequence in sampling.get('S_branch_train', []):
                points.extend((batch, sequence_point(sequence), 'train') for batch in bs[:2])
        long_request = sampling.get('long_request')
        long_sequence = None
        if long_request:
            _, long_sequence = request_shape(
                long_request['input_length'],
                long_request['output_length'],
                config['is_decode'],
                config['num_mtp_tokens'],
            )
            if not requested['S'][0] <= long_sequence <= cap:
                raise ValueError('Long-request target exceeds the requested/model sequence range; do not clamp it')
            points.append((bs[0], long_sequence, 'long_request'))
        samples, seen = [], set()
        training_geometry = {((batch + dp - 1) // dp, sequence) for batch, sequence, role in points if role == 'train'}
        for batch, sequence, role in points:
            target = {**config, 'B': batch, 'S': sequence, 'model_id': args.model_id}
            key, _ = build_key_and_config(target)
            if key in seen or sequence < config['num_mtp_tokens'] + 1:
                raise ValueError(
                    'Sample roles collide or query exceeds S; revise this cell rather than leak targets into training'
                )
            seen.add(key)
            if role != 'train' and ((batch + dp - 1) // dp, sequence) in training_geometry:
                raise ValueError('Independent target duplicates a training local-batch/S geometry')
            sample = {'id': f'c{index:03d}-{role}-{len(samples)}', 'role': role, 'key': key, 'config': target}
            if role == 'long_request':
                sample['request_lengths'] = long_request
            samples.append(sample)
        result.append({**cell, 'id': f'c{index:03d}', 'samples': samples})
    write(root / 'model-config.json', raw)
    write(root / 'epoch.json', epoch())
    document = {
        'schema': 'template_skill_plan/1',
        'model_id': args.model_id,
        'native_layers': layers,
        'model_config_sha256': sha(root / 'model-config.json'),
        'scope': scope,
        'context_cap': cap,
        'model_kind': 'moe' if experts else 'dense',
        'experts': experts,
        'cells': result,
        'candidate_not_model_validated': True,
        'rejected_layouts': rejected,
        'unsupported': scope['unsupported_must_be_reported'],
        'capture_upper_estimate': sum(len(c['samples']) for c in result),
    }
    write(root / 'plan.json', document)
    print(
        json.dumps(
            {
                'run_dir': str(root),
                'cells': len(result),
                'capture_upper_estimate': document['capture_upper_estimate'],
                'status': 'planned_not_captured',
            }
        )
    )


class BudgetReached(RuntimeError):
    pass


def bounded_read_call(callback, seconds, label):
    """Only read/compute stages run here; timed-out workers cannot publish files."""
    if seconds <= 0:
        raise BudgetReached("Time budget exhausted before " + label)
    results = queue.Queue(maxsize=1)

    def worker():
        try:
            results.put((True, callback()))
        except Exception as error:
            results.put((False, error))

    task = threading.Thread(target=worker, daemon=True)
    task.start()
    task.join(seconds)
    if task.is_alive():
        raise BudgetReached("Time budget exhausted during " + label)
    passed, result = results.get_nowait()
    if not passed:
        raise result
    return result


def model_snapshot_matches(expected, actual):
    """Frozen config is an expectation, never an override of MS's resolver."""
    if hasattr(actual, "to_dict"):
        actual = actual.to_dict()
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return False
    actual = {str(key): value for key, value in actual.items()}
    ignored = {"_name_or_path", "transformers_version", "_commit_hash"}
    for key, value in expected.items():
        if key in ignored:
            continue
        observed = actual.get(key)
        if isinstance(value, dict):
            if not model_snapshot_matches(value, observed):
                return False
        elif key in ("dtype", "torch_dtype"):
            if str(value).removeprefix("torch.") != str(observed).removeprefix("torch."):
                return False
        elif value != observed:
            return False
    return True


def run(args):
    started = time.monotonic()
    from template_engine.graph_codec import decode_capture, decode_document, encode_capture, read_artifact_bytes
    from template_engine.template_graph import load_template_graph, generate, TemplateGraphRefusal
    from template_engine.compare_ops import compare_performance_ops

    root = run_path(args.run_dir)
    document = read(root / 'plan.json')
    if epoch() != read(root / 'epoch.json') or sha(root / 'model-config.json') != document['model_config_sha256']:
        raise ValueError('Source/dependency/model config epoch changed; create a new run')
    state_path = root / 'state.json'
    if state_path.exists() and not args.resume:
        raise ValueError('Existing run state; inspect and pass --resume')
    state = (
        read(state_path)
        if state_path.exists()
        else {'plan_sha256': sha(root / 'plan.json'), 'samples': {}, 'comparisons': {}}
    )
    if state['plan_sha256'] != sha(root / 'plan.json'):
        raise ValueError('Plan changed; create a new run')
    for saved in state['samples'].values():
        if any(sha(saved[k]) != saved[k + '_sha256'] for k in ('graph', 'meta')):
            raise ValueError('Saved capture SHA changed')
    for saved in state['comparisons'].values():
        if saved.get('candidate') and sha(saved['candidate']) != saved['candidate_sha256']:
            raise ValueError('Saved candidate SHA changed')
    defaults = document['scope']['execution']
    limit = args.max_seconds if args.max_seconds is not None else defaults['max_seconds']
    capture_limit = args.max_captures if args.max_captures is not None else defaults['max_captures']
    previous_wall = max(
        float(state.get('elapsed_seconds', 0)),
        sum(float(read(p).get('wall_seconds', 0)) for p in (root / 'receipts').glob('*.json')),
    )

    def remaining_time():
        return limit - previous_wall - (time.monotonic() - started)

    def compute(label, callback):
        value = bounded_read_call(callback, remaining_time(), label)
        if remaining_time() <= 0:
            raise BudgetReached('Time budget exhausted during ' + label)
        return value

    def command(name, argv, *, capture=False):
        receipt = root / 'receipts' / (name + '.json')
        if receipt.exists():
            record = read(receipt)
            if record['argv'] != argv or record['cwd'] != str(REPO):
                raise ValueError('Command identity changed: ' + name)
            if record['status'] != 'complete' or record['returncode'] != 0:
                raise ValueError('Prior failed/unfinished attempt requires inspection: ' + name)
            return record['stdout']
        remaining = remaining_time()
        count = len(list((root / 'receipts').glob('capture-*.json')))
        if remaining <= 0 or (capture and count >= capture_limit):
            raise BudgetReached('Time/capture budget exhausted')
        if capture:
            import psutil

            captures = {}
            for proc in psutil.process_iter(['cmdline', 'pid', 'ppid']):
                try:
                    cmdline = proc.info['cmdline'] or []
                    if any(v in ('template_engine.capture_cli', 'template_engine.capture_ops') for v in cmdline) or (
                        'cli.inference.template_graph' in cmdline and 'capture' in cmdline
                    ):
                        captures[proc.info['pid']] = proc.info['ppid']
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            active = sum(parent not in captures for parent in captures.values())
            if active >= defaults['max_capture_processes']:
                raise BudgetReached(
                    'External captures occupy the configured concurrency limit; resume after they finish'
                )
        record = {'argv': argv, 'cwd': str(REPO), 'status': 'running'}
        write(receipt, record)
        env = {**os.environ, 'PYTHONPATH': str(REPO), 'PYTHONIOENCODING': 'utf-8'}
        begin = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=min(remaining, defaults['per_command_timeout']),
            )
        except subprocess.TimeoutExpired:
            record.update(status='timeout', wall_seconds=time.monotonic() - begin)
            write(receipt, record)
            raise BudgetReached('Command timed out; inspect its process state before resuming') from None
        record.update(
            status='complete',
            returncode=completed.returncode,
            wall_seconds=time.monotonic() - begin,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        write(receipt, record)
        (root / 'logs').mkdir(exist_ok=True)
        (root / 'logs' / (name + '.stdout')).write_text(completed.stdout, encoding='utf-8')
        (root / 'logs' / (name + '.stderr')).write_text(completed.stderr, encoding='utf-8')
        if completed.returncode:
            raise ValueError('Official CLI failed; see receipt: ' + name)
        return completed.stdout

    def capture(sample):
        if remaining_time() <= 0:
            raise BudgetReached('Time budget exhausted before capture')
        saved = state['samples'].get(sample['id'])
        if saved:
            if any(sha(saved[k]) != saved[k + '_sha256'] for k in ('graph', 'meta')):
                raise ValueError('Saved capture SHA changed')
            return saved
        c = sample['config']
        cache = root / 'captures' / sample['id']
        from template_engine.config_adapter import capture_cli_args

        argv = [
            sys.executable,
            '-m',
            'cli.inference.template_graph',
            'capture',
            *capture_cli_args(c),
            '--cache-dir',
            str(cache),
        ]
        stdout = command('capture-' + sample['id'], argv, capture=True)
        payload = next(json.loads(line) for line in reversed(stdout.splitlines()) if line.strip().startswith('{'))
        if payload['key'] != sample['key'] or not payload.get('path'):
            raise ValueError('Capture identity/path missing; do not guess a cache path')
        graph = Path(payload['path']).resolve()
        meta = Path(str(graph) + '.meta')
        info = compute('capture metadata decode', lambda: decode_document(read_artifact_bytes(meta), 'model_meta/1'))
        if (
            info.get('model_id') != document['model_id']
            or int(info.get('num_hidden_layers', 0)) != document['native_layers']
        ):
            raise ValueError('Capture model/native layer identity mismatch')
        captured_model = info.get('model_config')
        captured_config = getattr(captured_model, 'hf_config', None)
        if not model_snapshot_matches(read(root / 'model-config.json'), captured_config):
            raise ValueError('Captured model configuration differs from the frozen expectation')
        ops, _ = compute('capture graph decode', lambda: decode_capture(read_artifact_bytes(graph)))
        if not ops:
            raise ValueError('Capture is empty')
        saved = dict(
            graph=str(graph), meta=str(meta), graph_sha256=sha(graph), meta_sha256=sha(meta), key=sample['key']
        )
        state['samples'][sample['id']] = saved
        write(state_path, state)
        return saved

    cells = document['cells'][: args.max_cells] if args.max_cells is not None else document['cells']
    complete, errors, selected = [], [], []
    stop = None
    try:
        for cell in cells:
            try:
                records = [capture(s) for s in cell['samples'] if s['role'] == 'train']
                complete.append(cell['id'])
                selected.extend(records)
            except BudgetReached:
                raise
            except (ValueError, OSError, StopIteration) as exc:
                errors.append({'cell': cell['id'], 'stage': 'training', 'reason': str(exc)})
        token = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:16]
        training = root / 'training' / token
        training.mkdir(parents=True, exist_ok=True)
        expected = set()
        for record in selected:
            for key in ('graph', 'meta'):
                source = Path(record[key])
                expected.add(source.name)
                target = training / source.name
                if target.exists() and sha(target) != record[key + '_sha256']:
                    raise ValueError('Training file collision')
                shutil.copyfile(source, target)
        if {p.name for p in training.iterdir()} != expected:
            raise ValueError('Unexpected files in training directory')
        if not selected:
            raise ValueError('No complete training cells; no template can be built')
        token = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:16]
        template = root / 'templates' / (token + '.json.xz')
        template.parent.mkdir(exist_ok=True)
        compile_mode = 'on' if document['scope']['requested']['compile'] else 'off'
        command(
            'build-' + token,
            [
                sys.executable,
                '-m',
                'cli.inference.template_graph',
                'build',
                '--compile-mode',
                compile_mode,
                '--cache-dir',
                str(training),
                '--template',
                str(template),
            ],
        )
        if state.get('build_token') == token and state['template_sha256'] != sha(template):
            raise ValueError('Frozen template SHA changed')
        if state.get('build_token') and state['build_token'] != token:
            write(root / 'history' / (state['build_token'] + '.json'), state)
            state['comparisons'] = {}
        state['build_token'] = token
        state['template_sha256'] = sha(template)
        state['training_keys'] = [r['key'] for r in selected]
        shutil.copyfile(template, root / 'template.json.xz')
        write(state_path, state)
        graph = compute('template load', lambda: load_template_graph(template))
        denied = False

        def audit(event, arguments):
            if denied and event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
                path = run_path(os.fsdecode(arguments[0]))
                if any(path.is_relative_to(root / folder) for folder in ('captures', 'training')):
                    raise RuntimeError('Generation tried to read captured graphs')

        sys.addaudithook(audit)
        for cell in cells:
            if cell['id'] not in complete:
                continue
            for sample in cell['samples']:
                previous = state['comparisons'].get(sample['id'], {})
                if sample['role'] == 'train' or previous.get('status') in ('compared_pass', 'refused', 'mismatch'):
                    continue
                if sample['key'] in state['training_keys']:
                    raise ValueError('Target leaked into training')
                candidate = root / 'candidates' / token / (sample['id'] + '.json.xz')
                if previous.get('status') == 'generated_only':
                    candidate_ops, _ = compute(
                        'candidate decode', lambda: decode_capture(read_artifact_bytes(candidate))
                    )
                else:
                    try:
                        denied = True
                        result = compute(
                            'candidate generation',
                            lambda: generate(graph, sample['config'], return_ops=True),
                        )
                    except TemplateGraphRefusal as exc:
                        state['comparisons'][sample['id']] = {'status': 'refused', 'reason': str(exc)}
                        write(state_path, state)
                        continue
                    finally:
                        denied = False
                    candidate_ops = result['ops']
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_bytes(compute('candidate encoding', lambda: encode_capture(candidate_ops, 0.0)))
                    state['comparisons'][sample['id']] = {
                        'status': 'generated_only',
                        'candidate': str(candidate),
                        'candidate_sha256': sha(candidate),
                        'structure_bucket_id': result.get('structure_bucket_id'),
                    }
                    write(state_path, state)
                oracle = capture(sample)
                expected_ops, _ = compute(
                    'oracle decode', lambda: decode_capture(read_artifact_bytes(Path(oracle['graph'])))
                )
                comparison = compute(
                    'independent comparison', lambda: compare_performance_ops(expected_ops, candidate_ops, max_diffs=40)
                )
                status = 'compared_pass' if comparison['passed'] else 'mismatch'
                state['comparisons'][sample['id']].update(
                    status=status, comparison=comparison, oracle_sha256=oracle['graph_sha256']
                )
                write(root / 'comparisons' / token / (sample['id'] + '.json'), state['comparisons'][sample['id']])
                write(state_path, state)
    except BudgetReached as exc:
        stop = str(exc)
    except (ValueError, OSError, RuntimeError) as exc:
        errors.append({'stage': 'build_or_validation', 'reason': str(exc)})
    targets = [s for c in document['cells'] for s in c['samples'] if s['role'] != 'train']
    rows = [{**s, **state['comparisons'].get(s['id'], {'status': 'unrun'})} for s in targets]
    passed = sum(row['status'] == 'compared_pass' for row in rows)
    registration = {'status': 'disabled' if getattr(args, 'no_register', False) else 'not_registered'}
    if not getattr(args, 'no_register', False) and state.get('template_sha256') and remaining_time() > 0 and not stop:
        from template_engine.workflow_registration import register_validated

        training_records = [
            {'config': sample['config'], 'graph': state['samples'][sample['id']]['graph']}
            for cell in document['cells']
            for sample in cell['samples']
            if sample['role'] == 'train' and sample['id'] in state['samples']
        ]
        try:
            registration = register_validated(
                root / 'template.json.xz',
                document['model_id'],
                document['native_layers'],
                document['scope']['requested']['compile'],
                rows,
                training_records,
                root,
                read_call=compute,
                lock_timeout=max(0, remaining_time()),
            )
        except BudgetReached as exc:
            stop = str(exc)
            registration = {'status': 'not_registered', 'reason': str(exc)}
        except (ValueError, OSError, RuntimeError, TemplateGraphRefusal) as exc:
            registration = {'status': 'failed', 'reason': str(exc)}
            errors.append({'stage': 'registration', 'reason': str(exc)})
        state['registration'] = registration
        write(state_path, state)
    state['registration'] = registration
    state['elapsed_seconds'] = previous_wall + time.monotonic() - started
    write(state_path, state)
    report = {
        'status': 'passed' if passed == len(rows) and not errors and not stop else 'partial',
        'model_id': document['model_id'],
        'native_layers': document['native_layers'],
        'planned_cells': len(document['cells']),
        'complete_training_cells': complete,
        'training_keys': state.get('training_keys', []),
        'template_sha256': state.get('template_sha256'),
        'independent_targets_passed': passed,
        'template_generated_targets': sum(
            row['status'] in ('compared_pass', 'mismatch', 'generated_only') for row in rows
        ),
        'verified_hit_rate_over_planned': passed / len(rows) if rows else None,
        'mismatch_targets': sum(row['status'] == 'mismatch' for row in rows),
        'refused_targets': sum(row['status'] == 'refused' for row in rows),
        'targets_by_phase': {
            phase: {
                'planned': sum(bool(row['config']['is_decode']) == decode for row in rows),
                'passed': sum(
                    bool(row['config']['is_decode']) == decode and row['status'] == 'compared_pass' for row in rows
                ),
            }
            for phase, decode in (('prefill', False), ('decode', True))
        },
        'targets': rows,
        'errors': errors,
        'stop': stop,
        'unsupported': document['unsupported'],
        'new_wall_seconds': time.monotonic() - started,
        'runtime_optimizer_validated': False,
        'arbitrary_BS_proven': False,
        'registration': registration,
        'optimizer_template_ready': registration['status'] == 'registered',
    }
    write(root / 'report.json', report)
    print(
        json.dumps(
            {'status': report['status'], 'passed': passed, 'targets': len(rows), 'report': str(root / 'report.json')}
        )
    )
    return 0 if report['status'] == 'passed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    sub.add_parser('preflight')
    p = sub.add_parser('plan')
    p.add_argument('--model-id', required=True)
    p.add_argument('--model-config', type=Path)
    p.add_argument('--scope', type=Path)
    p.add_argument('--input-length', type=int, help='Long-request prompt length; use with --output-length')
    p.add_argument('--output-length', type=int, help='Long-request output length; use with --input-length')
    p.add_argument(
        '--decode-batch-coverage', choices=('typical', 'all'), help='Sample typical or all requested rank batches'
    )
    p.add_argument('--run-dir', required=True, type=Path)
    p = sub.add_parser('run')
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument(
        '--no-register', action='store_true', help='Generate/validate only; skip automatic optimizer registration'
    )
    p.add_argument('--max-cells', type=int)
    p.add_argument('--max-captures', type=int)
    p.add_argument('--max-seconds', type=float)
    args = parser.parse_args()
    if args.stage == 'preflight':
        for action in ('capture', 'build', 'generate'):
            subprocess.run(
                [sys.executable, '-m', 'cli.inference.template_graph', action, '--help'],
                cwd=REPO,
                env={**os.environ, 'PYTHONPATH': str(REPO)},
                check=True,
            )
        print(json.dumps({'repo': str(REPO), 'head': epoch()['head'], 'python': sys.executable, 'models_captured': 0}))
    elif args.stage == 'plan':
        plan(args)
    else:
        if any(v is not None and v <= 0 for v in (args.max_cells, args.max_captures, args.max_seconds)):
            parser.error('Budget limits must be positive')
        return run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
