#!/bin/bash
# Quick start script for running agent with multiple vLLM servers
# Usage: ./run_agent.sh [output_dir] [base_port] [num_servers] [dataset_name] [browser_backend] [model_path]

OUTPUT_DIR=${1:-"results/browsecomp-plus/OpenResearcher_dense"}
BASE_PORT=${2:-8001}
NUM_SERVERS=${3:-2}
DATASET_NAME=${4:-"browsecomp-plus"}
BROWSER_BACKEND=${5:-"local"}
MODEL=${6:-"OpenResearcher/Nemotron-3-Nano-30B-A3B"}


SEARCH_URL="http://localhost:8000"

# Get script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build comma-separated server URLs
SERVER_URLS=""
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    URL="http://localhost:${PORT}/v1"

    if [ -n "$SERVER_URLS" ]; then
        SERVER_URLS="${SERVER_URLS},${URL}"
    else
        SERVER_URLS="${URL}"
    fi
done

echo "=========================================="
echo "Starting Agent with Multiple vLLM Servers"
echo "=========================================="
echo "Model: $MODEL"
echo "Number of Servers: $NUM_SERVERS"
echo "Server URLs:"
for i in $(seq 0 $((NUM_SERVERS-1))); do
    PORT=$((BASE_PORT + i))
    echo "  - http://localhost:${PORT}/v1"
done
echo "Search Service: $SEARCH_URL"
echo "Dataset: $DATASET_NAME"
echo "Browser Backend: $BROWSER_BACKEND"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check if using browsecomp-plus dataset (needs local data path)
if [ "$DATASET_NAME" = "browsecomp-plus" ]; then
    DATA_PATH="${SCRIPT_DIR}/Tevatron/browsecomp-plus/data/*.parquet"
    echo "Using local BrowseComp+ dataset: $DATA_PATH"
    echo ""

    python deploy_agent.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL" \
        --search_url "$SEARCH_URL" \
        --dataset_name "$DATASET_NAME" \
        --data_path "$DATA_PATH" \
        --browser_backend "$BROWSER_BACKEND" \
        --reasoning_effort high \
        --vllm_server_url "$SERVER_URLS" \
        --max_concurrency_per_worker 32
else
    # HuggingFace datasets or OpenAI BrowseComp (no local data_path needed)
    echo "Using dataset: $DATASET_NAME"
    echo "Available datasets: browsecomp, hle, gaia, webwalkerqa, webwalkerqa_ref, xbench, seal, seal_ref"
    echo ""

    python deploy_agent.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL" \
        --search_url "$SEARCH_URL" \
        --dataset_name "$DATASET_NAME" \
        --browser_backend "$BROWSER_BACKEND" \
        --reasoning_effort high \
        --vllm_server_url "$SERVER_URLS" \
        --max_concurrency_per_worker 32
fi
