import ast
from pathlib import Path


def _has_pytest_test(module: ast.Module) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for node in module.body
    )


def test_research_paper_files_expose_pytest_tests():
    research_dir = Path(__file__).parent
    missing_tests = []

    for path in sorted(research_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _has_pytest_test(tree):
            continue
        missing_tests.append(path.name)

    assert not missing_tests, (
        "Each research_paper_tests/test_*.py file should expose at least one pytest "
        f"test function: {missing_tests}"
    )
