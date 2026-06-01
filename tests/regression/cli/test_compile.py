from cli.inference import throughput_optimizer


def test_compile_flags_are_parsed_for_cli_optimizer(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "throughput_optimizer",
            "--input-length",
            "8",
            "--output-length",
            "4",
            "Qwen/Qwen3-32B",
            "--num-devices",
            "1",
            "--compile",
            "--compile-allow-graph-break",
        ],
    )
    args = throughput_optimizer.arg_parse()
    assert args.compile is True
    assert args.compile_allow_graph_break is True
