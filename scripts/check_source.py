"""Lightweight source checks; does not validate training or paper scores."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = sorted(root.rglob("*.py"))
guidance = sorted((root / "llava1.6_results").glob("*.json"))
assert sources, "No Python sources found"
assert len(guidance) == 27, f"Expected 27 guidance files, found {len(guidance)}"
for path in sources:
    ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
for path in guidance:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data, f"Invalid guidance: {path.name}"
for required in ("README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY.md"):
    assert (root / required).is_file(), f"Missing release file: {required}"
print(f"PASS: {len(sources)} Python files parsed; {len(guidance)} guidance JSON files parsed.")
print("These checks do not run GPU training, evaluation, or reproduce paper results.")
