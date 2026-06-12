from web_ui.parsers import parse_result
from web_ui.schemas import ExperimentTask


def test_parse_result_text_api_contract():
    task = ExperimentTask(
        sim_type="text_generate",
        params={"decode": False},
        command=["python", "-m", "cli.inference.text_generate"],
        task_hash="h1",
        label="text-case",
    )
    log = "Total device memory: 80 GB\nMemory available: 10 GB\n"
    result = parse_result(task, log, "success")
    row = result.to_row()

    assert row["sim_type"] == "text_generate"
    assert row["status"] == "success"
    assert row["memory_fit_status"] == "fit"
