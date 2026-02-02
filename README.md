<p align="center">
  <img src="./assets/docs/or-logo1.png" height="82" style="vertical-align: middle;">
  <img src="./assets/docs/openresearcher-title.svg" height="66" style="vertical-align: middle;"></p>

<div align="center" style="line-height: 1; margin-top: 16px;">

[![Models](https://img.shields.io/badge/Models-FFD966?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/OpenResearcher)
[![Datasets](https://img.shields.io/badge/Datasets-FFB7B2?style=for-the-badge&logo=huggingface&logoColor=ffffff)](#)
[![Blog](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://openresearcher.github.io/)
[![WandB Logs](https://img.shields.io/badge/WandB%20Logs-48B5A3?style=for-the-badge&logo=weightsandbiases&logoColor=white)](#)
[![Eval Logs](https://img.shields.io/badge/Eval%20Logs-755BB4?style=for-the-badge&logo=weightsandbiases&logoColor=white)](#)

</div>

<p align="center">
🤗 <a href="https://huggingface.co/OpenResearcher" target="_blank">HuggingFace</a> ｜
🤗 <a href="#" target="_blank">Datasets</a> ｜
📰 <a href="https://openresearcher.github.io/" target="_blank">Blog</a> ｜
📈 <a href="#" target="_blank">WandB Log</a> ｜
📊 <a href="#" target="_blank">Eval Logs</a>
</p>

## Introduction

Evaluation framework for deep research agents with local/remote search engines and multiple LLM backends. TODO
<img width="20639" height="14239" alt="teaser-v3 (1)" src="https://github.com/user-attachments/assets/5ae76a21-8ac9-46a1-94a8-6884d7a4a67b" />

## 🏆 Experiments

## 📋 Table of Contents

- [🛠 Environment Setup](#-environment-setup)
  - [Installation](#installation)
  - [DeepResearch Benchmarks Preparation](#deepresearch-benchmarks-preparation)
- [🔍 Configuration](#-configuration)
- [🚀 Get Started](#-get-started)
  - [Example 1: BrowseComp-Plus with Local Search Engine](#example-1-browsecomp-plus-with-local-search-engine)
  - [Example 2: Using Serper API (No Local Search Needed)](#example-2-using-serper-api-no-local-search-needed)
  - [Evaluation](#evaluation)
- [🔬 Benchmark OpenResearcher](#-benchmark-openresearcher)
  - [Quick Commands](#quick-commands)
- [🤝 Core Contributors](#-core-contributors)
- [🎓 Advisors](#-advisors)
- [🚀 Contributing](#-contributing)
- [📚 Citation](#-citation)

## 🛠 Environment Setup
We run this repo on the following setup:
+ 8 * A100 80G Nvidia GPUs
+ Linux operating system

Other hardware setups can also work, but remember to modify the corresponding parameters.
### Installation 
```bash
sudo apt install -y openjdk-21-jdk
uv venv --python 3.12
source .venv/bin/activate

# install tevatron for BrowserComp-plus 
git clone https://github.com/texttron/tevatron.git
cd tevatron
uv pip install -e .
cd ..

# install all dependencies automatically
uv pip install -e .
```

### DeepResearch Benchmarks Preparation

Run the setup script to automatically download the **[BrowserComp-Plus](https://arxiv.org/abs/2508.06600)** benchmark.  
Other benchmarks, including **[BrowserComp](https://arxiv.org/abs/2504.12516)**, **[GAIA](https://arxiv.org/abs/2311.12983)**, **[xbench-DeepResearch](https://github.com/THUDM/xbench)** and **[SEAL-0](https://arxiv.org/abs/2506.01062)**, will be set up automatically when they are first used.

```bash
bash setup.sh
```

**This script will:**
- ✅ Verify Python 3.12 virtual environment and automatically install any missing dependencies
- ✅ Downlaod BrowserComp-Plus dataset from HuggingFace and set up the directory structure

Please see [benchmarks.md](assets/docs/benchmarks.md) for more info about these deepresearch benchmarks.

## 🔍 Configuration

Copy the template and configure your API keys:

```bash
cp .env.template .env
```

Edit `.env`:
```bash
# Serper API (for web search when using browser_backend=serper)
SERPER_API_KEY=your_key        # Get from: https://serper.dev/

# OpenAI API (for evaluation scoring)
OPENAI_API_KEY=your_key        # Get from: https://platform.openai.com/api-keys
```

## 🚀 Get Started

### Example 1: BrowseComp-Plus with Local Search Engine

Complete evaluation using local Dense search (**note: only applicable for BrowserComp-Plus**):

```bash
# Terminal 1: Start local Dense search service on port 8000
# Embedding model (Qwen3-Embedding-8B) will be deployed on GPUs 7
bash scripts/start_search_service.sh dense 8000

# Terminal 2: Start vLLM servers (requires 4 GPUs)
# TP=2, deploy 2 servers starting from port 8001 on GPUs 0,1,2,3
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3

# Terminal 3: Run agent
bash run_agent.sh results/browsecomp_plus/OpenResearcher_dense 8001 2 browsecomp_plus local OpenResearcher/Nemotron-3-Nano-30B-A3B
```

What this does:
- Deploys Dense retriever service on port 8000 as search engine
- Launches 2 vLLM servers (ports 8001, 8002) with TP=2 across 4 GPUs
- Runs deepresearch agent with load balancing across both servers

### Example 2: Using Serper API (No Local Search Needed)

Run with Serper Google Search API (**note: applicable to all benchmarks except BrowserComp-Plus**):

```bash
# Terminal 1: Start vLLM servers (requires 4 GPUs)
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3

# Terminal 2: Run agent with serper search backend
bash run_agent.sh results/gaia/OpenResearcher_serper 8001 2 gaia serper OpenResearcher/Nemotron-3-Nano-30B-A3B
```

**Browser Backend Options:**
- `local` - Use local BM25/Dense search service (for BrowseComp-Plus)
- `serper` - Use Serper Google Search API (for all other benchmarks)

For other parameters, refer to [parameter.md](assets/docs/parameter.md).

### Evaluation

After running experiments, evaluate results:

```bash
# eval on browsecomp_plus
python eval.py --input_dir results/browsecomp_plus_dense/OpenResearcher

# eval on gaia
python eval.py --input_dir results/gaia/OpenResearcher_serper
```

## 🔬 Benchmark OpenResearcher
We benchmark our OpenResearcher-30B-A3B using below deepresearch benchmarks: 

| Benchmark | Dataset Key | Size | Language | Search Backend | Description |
|-----------|-------------|------|----------|----------------|-------------|
| **[BrowseComp-Plus](https://arxiv.org/abs/2508.06600)** | `browsecomp_plus` | 830 | EN | local | Deep-research benchmark from BrowseComp isolating retriever and LLM agent effects |
| **[BrowseComp](https://arxiv.org/abs/2504.12516)** | `browsecomp` | 103 | EN | serper | A Simple Yet Challenging Benchmark for Browsing Agents |
| **[GAIA-text](https://arxiv.org/abs/2311.12983)** | `gaia` | 103 | EN | serper | Text-only subset of GAIA benchmark (dev split) |
| **[xbench-DeepResearch](https://github.com/THUDM/xbench)** | `xbench` | 100 | ZH | serper | DeepSearch benchmark with encrypted test cases |
| **[SEAL-0](https://arxiv.org/abs/2506.01062)** | `seal` | 111 | EN | serper | Hardest subset of SealQA questions |

For more info about these deepresearch benchmarks, see [benchmarks.md](assets/docs/benchmarks.md) 


### Quick Commands

| Scenario | Command |
|----------|---------|
| **BrowseComp-Plus (BM25)** | `bash scripts/start_search_service.sh bm25 8000` then `bash run_agent.sh results/browsecomp-plus/OpenResearcher_bm25 8001 2 browsecomp-plus local OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **BrowseComp-Plus (Qwen3-8B Dense Embeddings)** | `bash scripts/start_search_service.sh dense 8000` then `bash run_agent.sh results/browsecomp-plus/OpenResearcher_dense 8001 2 browsecomp-plus local OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **BrowseComp** | `bash run_agent.sh results/browsecomp 8001 2 browsecomp serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **GAIA** | `bash run_agent.sh results/gaia 8001 2 gaia serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **XBench** | `bash run_agent.sh results/xbench 8001 2 xbench serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |

**Note:** Don't forget to evaluate your results using:  
```bash
python eval.py --input_dir [INPUT_DIR]
```
## 🤝 Core Contributors

##  🎓 Advisors

##  🚀 Contributing


##  📚 Citation


