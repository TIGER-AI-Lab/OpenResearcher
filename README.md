<p align="center">
  <img src="./assets/imgs/or-logo1.png" height="82" style="vertical-align: middle;">
  <img src="./assets/imgs/openresearcher-title.svg" height="66" style="vertical-align: middle;"></p>

<div align="center" style="line-height: 1; margin-top: 16px;">

[![Dataset](https://img.shields.io/badge/Dataset-FFB7B2?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/OpenResearcher)
[![Model](https://img.shields.io/badge/Model-FFD966?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/OpenResearcher/Nemotron-3-Nano-30B-A3B)
[![Blog](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://openresearcher.github.io/)
[![WandB Logs](https://img.shields.io/badge/WandB%20Logs-48B5A3?style=for-the-badge&logo=weightsandbiases&logoColor=white)](https://wandb.ai/dongfu/nano-v3-sft-search)
[![Eval Logs](https://img.shields.io/badge/Eval%20Logs-755BB4?style=for-the-badge&logo=weightsandbiases&logoColor=white)](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Eval-Logs/tree/main)

</div>

<p align="center">
🤗 <a href="https://huggingface.co/OpenResearcher" target="_blank">HuggingFace</a> ｜
<img src="./assets/imgs/notion.svg" width="15px" style="display:inline;"> <a href="https://openresearcher.github.io/" target="_blank">Blog</a> ｜
<img src="./assets/imgs/slack.png" width="14px" style="display:inline;"> <a href="https://join.slack.com/t/openresearcher/shared_invite/zt-3p0r32cky-PqtZkVjjWIAI14~XwcRMfQ" target="_blank">Slack</a> | <img src="./assets/imgs/wechat.svg" width="14px" style="display:inline;"> <a href="./assets/imgs/wechat_group.png" target="_blank">WeChat</a> |

</p>

## Introduction

**OpenResearcher** is a fully open agentic large language model (30B-A3B) designed for **long-horizon deep research** scenarios. It achieves an impressive **54.8%** accuracy on [BrowserComp-Plus](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus), surpassing performance of `GPT-4.1`, `Claude-Opus-4`, `Gemini-2.5-Pro`, `DeepSeek-R1`, `Tongyi-DeepResearch`. We **fully open-source** the complete training and evaluation recipe—including data, model, training methodology, and evaluation pipeline for everyone to progress deep research and develop state-of-the-art models efficiently.
<div align="center">
  <img src="assets/imgs/teaser.png" alt="OpenResearcher Teaser" width="100%" style="max-width: 850px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
</div>

<br>

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
- [🙏 Acknowledgements](#-acknowledgements)
- [✨ Contributing](#-contributing)
- [📚 Citation](#-citation)

## 🛠 Environment Setup
We run this repo on the following setup:
+ 8 * A100 80G Nvidia GPUs
+ Linux operating system

Other hardware setups can also work, but remember to modify the corresponding parameters.
### Installation 
```bash
sudo apt update 
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
Other benchmarks, including **[BrowserComp](https://arxiv.org/abs/2504.12516)**, **[GAIA](https://arxiv.org/abs/2311.12983)** and **[xbench-DeepResearch](https://github.com/THUDM/xbench)****, will be set up automatically when they are first used.

```bash
bash setup.sh
```

**This script will:**
- ✅ Verify Python 3.12 virtual environment and automatically install any missing dependencies
- ✅ Downlaod BrowserComp-Plus dataset from HuggingFace and set up the directory structure

For more info about these deepresearch benchmarks, see [benchmarks.md](assets/docs/benchmarks.md) 

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

For more info about these deepresearch benchmarks, see [benchmarks.md](assets/docs/benchmarks.md) 


### Quick Commands

| Scenario | Command |
|----------|---------|
| **BrowseComp-Plus (BM25)** | `bash scripts/start_search_service.sh bm25 8000` then `bash run_agent.sh results/browsecomp-plus/OpenResearcher_bm25 8001 2 browsecomp-plus local OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **BrowseComp-Plus (Qwen3-8B Dense Embeddings)** | `bash scripts/start_search_service.sh dense 8000` then `bash run_agent.sh results/browsecomp-plus/OpenResearcher_dense 8001 2 browsecomp-plus local OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **BrowseComp** | `bash run_agent.sh results/browsecomp 8001 2 browsecomp serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **GAIA** | `bash run_agent.sh results/gaia 8001 2 gaia serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **xbench-DeepResearch** | `bash run_agent.sh results/xbench 8001 2 xbench serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |

**Note:** Don't forget to evaluate your results using:  
```bash
python eval.py --input_dir [INPUT_DIR]
```
## 🤝 Core Contributors

<table>
<tr>
    <td align="center">
        <a href="https://zhuofeng-li.github.io/">
            <img src="https://github.com/Zhuofeng-Li.png" width="75px;" alt="Zhuofeng Li"/>
            <br />
            <sub><b>Zhuofeng Li</b></sub>
        </a>
    </td>
        <td align="center">
        <a href="https://github.com/jdf-prog">
            <img src="https://github.com/jdf-prog.png" width="75px;" alt="Dongfu Jiang"/>
            <br />
            <sub><b>Dongfu Jiang</b></sub>
        </a>
    </td>
    </td>
        <td align="center">
        <a href="https://mxueguang.github.io/">
            <img src="https://mxueguang.github.io/images/profile.jpg" width="75px;" alt="Xueguang"/>
            <br />
            <sub><b>Xueguang Ma</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://isaacghx.github.io/about/">
            <img src="https://github.com/IsaacGHX.png" width="75px;" alt="Haoxiang Zhang"/>
            <br />
            <sub><b>Haoxiang Zhang</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/erenup">
            <img src="https://media.licdn.com/dms/image/v2/C5603AQHqewKc14kzXg/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1648383444846?e=1771459200&v=beta&t=I6OTYpCSLp0LqclGi2nguJbEDZ_Aw01H1Y4kkyDYZC8" width="75px;" alt="Ping Nie"/>
            <br />
            <sub><b>Ping Nie</b></sub>
        </a>
    </td>
</tr>
</table>

## 🎓 Advisors

<table>
<tr>
      <td align="center">
        <a href="https://github.com/wenhuchen">
            <img src="https://github.com/wenhuchen.png" width="75px;" alt="Wenhu Chen"/>
            <br />
            <sub><b>Wenhu Chen</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://yuzhimanhua.github.io/">
            <img src="https://yuzhimanhua.github.io/profile_pic.jpg" width="75px;" alt="Yu Zhang"/>
            <br />
            <sub><b>Yu Zhang</b></sub>
        </a>
    </td>
</tr>
</table>

## 🙏 Acknowledgements
We thank [Lambda](https://lambda.ai), [Netmind AI](https://www.netmind.ai/), [Verdent AI](https://www.verdent.ai) for GPU and API support!


## ✨ Contributing
We are truly looking forward to open-source contributions to OpenResearcher! If you’re interested in contributing, collaborating, or reporting issues, please feel free to open an issue or submit a pull request (PR). You can also reach us at [zhuofengli12345@gmail.com](mailto:zhuofengli12345@gmail.com).

We are also looking forward to your feedback and suggestions!

##  📚 Citation
TODO: update 
```bibtex
@misc{deepscaler2025,
  title={DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL},
  author={Michael Luo and Sijun Tan and Justin Wong and Xiaoxiang Shi and William Y. Tang and Manan Roongta and Colin Cai and Jeffrey Luo and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2}},
  note={Notion Blog},
  year={2025}
}
```
