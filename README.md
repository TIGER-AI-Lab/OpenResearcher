# 🚀 Quick Start

**Prerequisites:** Install dependencies and configure Serper and OpenAI API keys (see [Configuration](#-configuration))

1. **Deploy OpenResearcher-30B-A3B**:

```bash
bash scripts/start_nemotron_servers.sh
```

2. **Run your first task**:

Here's a minimal example to demonstrate OpenResearcher's deep research capabilities:

```python
import asyncio
from deploy_agent import run_one, BrowserPool
from utils.openai_generator import OpenAIAsyncGenerator

async def main():
    # Initialize generator and browser
    generator = OpenAIAsyncGenerator(
        base_url="http://localhost:8001/v1",
        model_name="OpenResearcher/Nemotron-3-Nano-30B-A3B",
        use_native_tools=True
    )
    browser_pool = BrowserPool(browser_backend="serper")

    # Run deep research
    await run_one(
        question="What is the latest news about OpenAI?",
        qid="quick_start",
        generator=generator,
        browser_pool=browser_pool,
        verbose=True
    )

    browser_pool.cleanup("quick_start")

if __name__ == "__main__":
    asyncio.run(main())
```

The agent will automatically search the web, browse webpages, and extract relevant information. You'll see the final answer along with all intermediate reasoning steps.
