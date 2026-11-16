# Evaluation Benchmarks

This document provides an overview of the benchmarks used for evaluation in this project.

## Available Benchmarks

| Benchmark | Dataset Key | Size | Language | Search Backend | Description |
|-----------|-------------|------|----------|----------------|-------------|
| **[BrowseComp-Plus](https://arxiv.org/abs/2508.06600)** | `browsecomp_plus` | 830 | EN | local | Deep-research benchmark from BrowseComp isolating retriever and LLM agent effects |
| **[BrowseComp](https://arxiv.org/abs/2504.12516)** | `browsecomp` | 1,266 | EN | serper | A Simple Yet Challenging Benchmark for Browsing Agents |
| **[GAIA-text](https://arxiv.org/abs/2311.12983)** | `gaia` | 103 | EN | serper | Text-only subset of GAIA benchmark (dev split) |
| **[xbench-DeepResearch](https://github.com/THUDM/xbench)** | `xbench` | 100 | ZH | serper | DeepSearch benchmark with encrypted test cases |


## Usage

To run evaluations on these benchmarks, use the dataset keys listed in the "Dataset Key" column:

```bash
# BrowseComp-Plus (requires local search service)
bash scripts/start_search_service.sh bm25 8000
bash run_agent.sh results/bc 8001 2 browsecomp_plus local <model>

# Other benchmarks (using Serper API)
bash run_agent.sh results/hle 8001 2 hle serper <model>
bash run_agent.sh results/gaia 8001 2 gaia serper <model>
bash run_agent.sh results/xbench 8001 2 xbench serper <model>
```

## Search Backend Notes

- **local**: Required for BrowseComp-Plus. Uses local BM25 or Dense search service.
- **serper**: Uses Serper Google Search API. Requires `SERPER_API_KEY` in `.env`.
- **specstream**: Uses SpecStream Search and Extract APIs. Requires `SPECSTREAM_API_KEY` in `.env`.
