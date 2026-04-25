# Re-export build_scenario from the flat module for backwards compatibility
import importlib
import sys

# The original scenarios.py lives at app/scenarios.py but is shadowed
# by this package (app/scenarios/). Import from the flat module path.
from pathlib import Path
_parent = Path(__file__).resolve().parent.parent
_mod_path = _parent / "scenarios.py"

# Check if the flat file exists alongside the package
if _mod_path.exists():
    import importlib.util
    _spec = importlib.util.spec_from_file_location("app._scenarios_flat", str(_mod_path))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_scenario = _mod.build_scenario
else:
    # Fallback: try importing from scenario_base
    from .scenario_base import generate_basic_scenario as build_scenario
