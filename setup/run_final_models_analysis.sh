#!/bin/bash
# Complete workflow for testing and analyzing final models
# 
# This script:
# 1. Tests model compatibility with TPU
# 2. Runs batch TTFT/TPOT measurements
# 3. Analyzes results and generates visualizations
#
# Usage:
#   bash run_final_models_analysis.sh
#
# Run this inside the vLLM Docker container

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
CONFIG_FILE="${PROJECT_ROOT}/configs/final_models_batch_config.json"

echo "=========================================="
echo "Final Models Analysis Workflow"
echo "=========================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Step 1: Compatibility Test
echo -e "${YELLOW}Step 1: Testing model compatibility...${NC}"
echo ""

python3 "${SCRIPT_DIR}/test_final_models_compatibility.py"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Compatibility test failed!${NC}"
    echo "Please check TPU connectivity and model access before proceeding."
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Compatibility test passed!${NC}"
echo ""

# Step 2: Batch Measurements
echo -e "${YELLOW}Step 2: Running batch TTFT/TPOT measurements...${NC}"
echo ""
echo "This may take a while (loading models, running measurements)..."
echo ""

if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}✗ Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi

python3 "${SCRIPT_DIR}/batch_measure_ttft_tpot.py" --config "${CONFIG_FILE}"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Batch measurements failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Batch measurements complete!${NC}"
echo ""

# Step 3: Analysis and Visualization
echo -e "${YELLOW}Step 3: Analyzing results and generating visualizations...${NC}"
echo ""

# Find the most recent summary file
SUMMARY_FILE=$(find "${RESULTS_DIR}" -name "summary_*.json" -type f | sort -r | head -1)

if [ -z "${SUMMARY_FILE}" ]; then
    echo "⚠ No summary file found, analyzing all results in ${RESULTS_DIR}"
    python3 "${SCRIPT_DIR}/analyze_final_models_results.py" --results-dir "${RESULTS_DIR}" --output-dir "${RESULTS_DIR}"
else
    echo "Using summary file: ${SUMMARY_FILE}"
    python3 "${SCRIPT_DIR}/analyze_final_models_results.py" --summary "${SUMMARY_FILE}" --output-dir "${RESULTS_DIR}"
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Analysis failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Analysis complete!${NC}"
echo ""

# Final summary
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Generated files:"
echo "  - Individual measurement JSON files"
echo "  - Summary JSON file"
echo "  - Analysis plots (PNG images)"
echo "  - Analysis summary (TXT)"
echo ""
echo "To view results:"
echo "  cd ${RESULTS_DIR}"
echo "  ls -lh *.png *.txt *.json"
echo ""

