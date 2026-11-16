# Script Parameters Reference

## scripts/start_search_service.sh

Start local search service for BrowseComp-Plus benchmark.

```bash
bash scripts/start_search_service.sh [searcher_type] [port] [cuda_visible_devices]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `searcher_type` | `dense` | `bm25` or `dense` |
| `port` | `8000` | Port for search service |
| `cuda_visible_devices` | `0` | GPU ID for dense searcher (ignored for BM25) |

**Important!! You should make sure there is sufficient GPU memory for Dense search -- which would cost about 15GB

**Examples:**
```bash
# BM25 search (CPU-based, lightweight)
bash scripts/start_search_service.sh bm25 8000

# Dense search (GPU-based, better quality)
bash scripts/start_search_service.sh dense 8000 0
```

---
## scripts/start_nemotron_servers.sh

Start multiple vLLM servers with tensor parallelism.

```bash
bash scripts/start_nemotron_servers.sh [tensor_parallel_size] [base_port] [cuda_visible_devices] [model_path]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tensor_parallel_size` | `2` | Number of GPUs per server |
| `base_port` | `8001` | Base port number for servers |
| `cuda_visible_devices` | `0,1,2,3,4,5,6,7` | Comma-separated GPU IDs to use |
| `model_path` | (see script) | Model name or path |

**Examples:**
```bash
# 2 servers with TP=2 on 4 GPUs
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3

# 4 servers with TP=2 on 8 GPUs
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3,4,5,6,7

# 1 server with TP=4 on 4 GPUs
bash scripts/start_nemotron_servers.sh 4 8001 0,1,2,3
```

---

## run_agent.sh

Main script for running the agent on benchmarks.

```bash
bash run_agent.sh [output_dir] [base_port] [num_servers] [dataset_name] [browser_backend] [model_path]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | `results/browsecomp/output` | Output directory for results |
| `base_port` | `8002` | Base port number for vLLM servers |
| `num_servers` | `3` | Number of vLLM servers to use |
| `dataset_name` | `browsecomp` | Dataset key (see benchmarks.md) |
| `browser_backend` | `local` | `local`, `serper`, or `specstream` |
| `model_path` | `OpenResearcher/OpenResearcher-30B-A3B` | Model name or path |

**Examples:**
```bash
# BrowseComp-Plus with local search
bash run_agent.sh results/bc 8001 2 browsecomp_plus local OpenResearcher/OpenResearcher-30B-A3B

# GAIA with Serper API
bash run_agent.sh results/gaia 8001 2 gaia serper OpenResearcher/OpenResearcher-30B-A3B

# GAIA with SpecStream API
bash run_agent.sh results/gaia 8001 2 gaia specstream OpenResearcher/OpenResearcher-30B-A3B
```

---
## scripts/stop_servers.sh

Stop all running vLLM servers.

```bash
bash scripts/stop_servers.sh
```

This script reads PIDs from `logs/server_pids.txt` and terminates all servers.
