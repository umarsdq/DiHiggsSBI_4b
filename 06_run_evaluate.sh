#!/bin/bash

# Script to run 06a or 06b evaluation scripts
# Usage: ./run_evaluate.sh <script_type> <parameter_code> <number_features>
# Example: ./run_evaluate.sh 06a c0 5

script_type=$1
parameter_code=$2
number_features=$3

# Check if all arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <script_type> <parameter_code> <number_features>"
    echo "  script_type: 06a or 06b"
    echo "  parameter_code: e.g., c0, c1, c2"
    echo "  number_features: e.g., 1, 3, 5"
    exit 1
fi

# Validate script type
if [ "$script_type" != "06a" ] && [ "$script_type" != "06b" ]; then
    echo "Error: script_type must be either '06a' or '06b'"
    exit 1
fi

eval "$(/home/hep/us322/miniforge3/bin/conda shell.bash hook)"
conda activate madminer_env

# Set up environment
echo "Running ${script_type} evaluation with parameter_code=${parameter_code} and number_features=${number_features}"
echo "Started at: $(date)"

# Run the appropriate script
if [ "$script_type" == "06a" ]; then
    python3 06a_evaluate_test_statistic.py -p $parameter_code -n $number_features
elif [ "$script_type" == "06b" ]; then
    python3 06b_evaluate_coverage.py -p $parameter_code -n $number_features
fi

echo "Finished at: $(date)" 