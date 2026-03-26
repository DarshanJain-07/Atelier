import ast
from pathlib import Path


def _has_pytest_test(module: ast.Module) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for node in module.body
    )


def _has_main_guard(module: ast.Module) -> bool:
    for node in module.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            continue
        left = test.left
        right = test.comparators[0]
        if (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        ):
            return True
    return False


def test_research_paper_files_are_collectable_or_explicit_scripts():
    research_dir = Path(__file__).parent
    missing_contract = []

    for path in sorted(research_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _has_pytest_test(tree) or _has_main_guard(tree):
            continue
        missing_contract.append(path.name)

    assert not missing_contract, (
        "Each research_paper_tests/test_*.py file should either expose a pytest "
        f"test function or be explicitly runnable as a script: {missing_contract}"
    )
