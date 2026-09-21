"""Tests for hal.agent_runner.load_raw_submissions (shared --continue_run replay)."""

import json

from hal.agent_runner import load_raw_submissions


def test_later_success_supersedes_earlier_error(tmp_path):
    submissions_file = tmp_path / "run_RAW_SUBMISSIONS.jsonl"
    lines = [
        {"3": "ERROR: Task timed out after 10 seconds"},
        {"3": {"answer": "ok-3"}},
        {"5": {"answer": "ok-5"}},
    ]
    submissions_file.write_text("".join(json.dumps(l) + "\n" for l in lines))

    previous_output, completed_tasks = load_raw_submissions(str(submissions_file))

    assert previous_output == {"3": {"answer": "ok-3"}, "5": {"answer": "ok-5"}}
    assert completed_tasks == {"3", "5"}


def test_error_only_task_is_kept_but_not_completed(tmp_path):
    submissions_file = tmp_path / "run_RAW_SUBMISSIONS.jsonl"
    submissions_file.write_text(
        json.dumps({"7": "ERROR: boom"}) + "\nnot json\n" + json.dumps({"8": 1}) + "\n"
    )

    previous_output, completed_tasks = load_raw_submissions(str(submissions_file))

    assert previous_output == {"7": "ERROR: boom", "8": 1}
    assert completed_tasks == {"8"}
