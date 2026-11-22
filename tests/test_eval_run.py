"""
tests/test_eval_run.py
-----------------------
Unit test ensuring the evaluation pipeline executes without errors in CI mode.
"""


from evals.run_eval import main


def test_eval_runner_ci_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--ci"])
    # Should run to completion and pass all assertions without raising
    main()
