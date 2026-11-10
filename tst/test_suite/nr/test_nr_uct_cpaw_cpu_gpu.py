"""Isothermal HLLD+PLM UCT accuracy, geometry, and method-selection regression."""
from pathlib import Path

from test_suite.nr.uct_validation import run_regressions


def test_uct_cpaw(tmp_path):
    build = Path.cwd()
    run_regressions(build/"athena", build/"inputs/uct_cpaw.athinput", tmp_path/"uct")
