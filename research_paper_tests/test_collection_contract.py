import ast
from pathlib import Path

def _has_pytest_test(module: ast.Module) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        for node in module.body
    )

def _uses_statistical_validation(module: ast.Module) -> bool:
    """Checks if the file imports stats_utils or uses the n_seeds fixture."""
    uses_stats = any(
        (isinstance(node, ast.ImportFrom) and "stats_utils" in (node.module or "")) or
        (isinstance(node, ast.Import) and any(alias.name == "research_paper_tests.stats_utils" for alias in node.names))
        for node in module.body
    )
    
    uses_n_seeds = False
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            if any(arg.arg == "n_seeds" for arg in node.args.args):
                uses_n_seeds = True
                break

    return uses_stats or uses_n_seeds


SCHEMA_DATA_NAMES = {
    "DIMENSION_INDICES",
    "DIMENSIONS",
    "EMOTION_LABELS",
    "PERSONALITY_CORRELATIONS",
    "PERSONALITY_QUERY_MATRIX",
    "PSYCH_PROJECTION",
    "RUN_PROFILE_INTERNAL_ONLY_FIELDS",
    "SIM_CONFIG_DEFAULTS",
    "SIM_CONFIG_FIELDS",
    "SimConfig",
    "emotions_to_behavior_aware_sentiment_distribution",
    "emotions_to_sentiment_distribution",
    "emotions_to_valence",
}


def _schema_data_import_violations(module: ast.Module) -> list[str]:
    violations: list[str] = []
    for node in module.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in {"schema", "main"}:
            continue
        imported_names = {alias.name for alias in node.names}
        bad_names = sorted(imported_names & SCHEMA_DATA_NAMES)
        violations.extend(f"{node.module}.{name}" for name in bad_names)
    return violations

def test_research_paper_files_enforce_standards():
    research_dir = Path(__file__).parent
    missing_tests = []
    missing_stats = []
    schema_data_imports = {}

    # Files exempted from statistical validation requirements (contract, perf, or sanity tests)
    exempt_from_stats = {
        "test_collection_contract.py",
        "test_run_profile_contract.py",
        "test_paper_values_report.py",
        "test_research_paper_figures.py",
        "test_accuracy_metrics.py",
        "test_zero_tensor.py",
        "test_tensor_magnitudes.py",
        "test_ram_usage.py",
        "test_runtime_regressions.py",
        "test_trait_distribution.py",
        "test_personality_correlations.py",
        "test_response_boundaries.py",
        "test_sentiment_mapping.py",
        "test_trait_distribution_comparison.py",
    }

    for path in sorted(research_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        
        if not _has_pytest_test(tree):
            missing_tests.append(path.name)
            continue
            
        if path.name not in exempt_from_stats and not _uses_statistical_validation(tree):
            missing_stats.append(path.name)

        violations = _schema_data_import_violations(tree)
        if violations:
            schema_data_imports[path.name] = violations

    assert not missing_tests, (
        f"Files missing pytest functions: {missing_tests}"
    )
    
    assert not missing_stats, (
        "The following research tests must use statistical validation (stats_utils or n_seeds) "
        "to ensure real-world simulation robustness and avoid naive thresholds: "
        f"{missing_stats}"
    )
    assert not schema_data_imports, (
        "Research tests must import schema-derived data from "
        "research_paper_tests.config_schema, not schema.py or main.py: "
        f"{schema_data_imports}"
    )
