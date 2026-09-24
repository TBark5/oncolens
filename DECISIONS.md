# Decisions Log

Choices made without asking, with the reason for each. Newest last.

| # | Decision | Why |
|---|----------|-----|
| 1 | Use the existing `.venv` (Python 3.14.5). | It was already in the folder; every dependency installed cleanly from wheels. |
| 2 | Top-level deps in `requirements.in`, exact pins in `requirements.txt`. | Human-readable intent plus reproducible installs. |
| 3 | Removed PyCharm's template `main.py`. | Boilerplate, unrelated to the project. |
| 4 | Layout: `src/oncolens/` package, `scripts/` one script per phase, `run_all.py` runs them in order. | Hard rule 9: small modules, package plus scripts. |
