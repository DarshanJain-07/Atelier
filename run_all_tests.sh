#!/bin/bash

# Exit on error
set -e

# Set PYTHONPATH to the current directory so modules like 'schema' can be found
export PYTHONPATH=.

# Output file
OUTPUT_FILE="test_results_output.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

echo "===========================================" | tee -a "$OUTPUT_FILE"
echo "Running All Empirical Tests..."              | tee -a "$OUTPUT_FILE"
echo "===========================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Array of test files
TESTS=(
    "research_paper_tests/test_accuracy_metrics.py"
    "research_paper_tests/test_agent_memory.py"
    "research_paper_tests/test_algorithmic_filter_bubble.py"
    "research_paper_tests/test_bimodality_polarization.py"
    "research_paper_tests/test_cascade_power_law.py"
    "research_paper_tests/test_cluster_cohesion.py"
    "research_paper_tests/test_cognitive_gate.py"
    "research_paper_tests/test_echo_chambers.py"
    "research_paper_tests/test_endogenous_events.py"
    "research_paper_tests/test_ideological_influence_gini.py"
    "research_paper_tests/test_influence_susceptibility_ratio.py"
    "research_paper_tests/test_louvain_modularity.py"
    "research_paper_tests/test_maximum_virality.py"
    "research_paper_tests/test_network_topology.py"
    "research_paper_tests/test_personal.py"
    "research_paper_tests/test_r0_basic_reproduction.py"
    "research_paper_tests/test_ram_usage.py"
    "research_paper_tests/test_relative_deprivation.py"
    "research_paper_tests/test_semantic_alignment.py"
    "research_paper_tests/test_signal_distortion.py"
    "research_paper_tests/test_wealth_gini.py"
)

# Loop through and run each test, appending output to the file
for TEST in "${TESTS[@]}"; do
    echo "--------------------------------------------------------" | tee -a "$OUTPUT_FILE"
    echo "Running: $TEST" | tee -a "$OUTPUT_FILE"
    echo "--------------------------------------------------------" | tee -a "$OUTPUT_FILE"

    # Run the test and append both stdout and stderr to the output file
    python "$TEST" 2>&1 | tee -a "$OUTPUT_FILE"

    echo "" | tee -a "$OUTPUT_FILE"
done

echo "===========================================" | tee -a "$OUTPUT_FILE"
echo "All tests completed. Results saved to $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "===========================================" | tee -a "$OUTPUT_FILE"
