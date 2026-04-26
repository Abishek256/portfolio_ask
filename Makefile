# Note: On Windows, run the Python commands directly instead of make targets.
# See README.md for Windows-equivalent commands.

.PHONY: setup run eval

setup:
	pip install -r requirements.txt
	python -m portfolio_ask.ingest

run:
	python -m portfolio_ask

eval:
	python evals/run_evals.py
