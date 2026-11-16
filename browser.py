import logging
from aiohttp import ClientSession
from typing import AsyncIterator, Any, List, Union
import time,json,html
import asyncio
import requests
from gpt_oss.tools.simple_browser.page_contents import (
    process_html,
)
from gpt_oss.tools.simple_browser.backend import (
    VIEW_SOURCE_PREFIX,
    BackendError,
    maybe_truncate,
)

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
)

from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool,maybe_get_function_args,function_the_model_can_call,handle_errors
from gpt_oss.tools.simple_browser.backend import BackendError, maybe_truncate

import os


logger = logging.getLogger(__name__)

def sanitize_dict_keys(d):
    """Remove None keys from dictionary."""
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if k is not None}

class BrowserTool(SimpleBrowserTool):
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        def make_error_message(error: str) -> Message:
            return self.make_response(
                content=TextContent(text=json.dumps({"error": error})),
                author=Author(role=Role.TOOL, name=message.recipient),
            )

        function_args = maybe_get_function_args(message, tool_name=self.name)
        if function_args is None:
            yield make_error_message("Invalid function arguments")
            return

        _, function_name = message.recipient.split(".")
        if function_name not in ["search", "open", "find"]:
            yield make_error_message(f"Unknown function: {function_name}")
            return
        try:
            if function_name == "search":
                async for msg in self.search(**function_args):
                    yield msg
            elif function_name == "open":
                async for msg in self.open(**function_args):
                    yield msg
            elif function_name == "find":
                async for msg in self.find(**function_args):
                    yield msg
        except TypeError as e:
            error_text = f"Error: Invalid arguments for function '{function_name}'. Please check the function signature. Details: {e}"
            error_content = TextContent(text=error_text)
            yield self.make_response(
                content=error_content,
                author=Author(role=Role.TOOL, name=message.recipient)
            )
        except Exception as e:
            error_text = f"An unexpected error occurred while executing function '{function_name}': {e}"
            error_content = TextContent(text=error_text)
            yield self.make_response(
                content=error_content,
                author=Author(role=Role.TOOL, name=message.recipient)
            )
    
    async def _open_url(self, url: str, direct_url_open: bool):
        """Use the cache, if available."""
        backend = self.backend
        # direct_url_open should be regarded as a refresh
        if not direct_url_open and (page := self.tool_state.get_page_by_url(url)):
            assert page.url == url
            return page

        try:
            async with ClientSession() as session:
                page = await backend.fetch(url, session=session)
            return page
        except Exception as e:
            msg = maybe_truncate(str(e))
            raise BackendError(
                f"Error fetching URL `{maybe_truncate(url)}`: {msg}"
            ) from e

class LocalServiceBrowserBackend:
    source = "web"
    
    def __init__(self,base_url):
        self.base_url = base_url
        
    async def _post(self, session: ClientSession, endpoint: str, payload: dict) -> dict:
        t0 = time.time()
        async with session.post(f"{self.base_url}{endpoint}", json=payload) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Search error {resp.status}: {await resp.text()}"
                )
            return await resp.json()
                
    async def _search_single(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> tuple[str, list]:
        """Execute a single search query and return query with results."""
        data = await self._post(
            session,
            "/search",
            {"query": query, "topn": topn},
        )
        results = data.get("results", [])
        if not results:
            logger.warning(f"No results returned for query: '{query}'")
            return query, []

        title_url_summary = []
        for result in results:
            title_url_summary.append((
                html.escape(result['title'], quote=True),
                html.escape(result['url'], quote=True),
                html.escape(result['summary'], quote=True)
            ))
        return query, title_url_summary

    async def search(
        self,
        query: Union[str, List[str]],
        topn: int = 5,
        session = None,
    ):
        """Search for one or more queries. If query is a list, searches are executed in parallel."""
        # Handle single query
        if isinstance(query, str):
            query_list = [query]
            title_str = query
        else:
            query_list = query
            # Create a title with all query names
            title_str = " | ".join(query_list)

        # Execute searches in parallel
        tasks = [self._search_single(q, topn, session) for q in query_list]
        all_results = await asyncio.gather(*tasks)

        # Merge all results
        title_url_summary_all = []
        for query_str, title_url_summary in all_results:
            # Add results from each query
            if title_url_summary:
                title_url_summary_all.extend(title_url_summary)

        # If no results from any query, raise error
        if not title_url_summary_all:
            raise BackendError(f"No results returned for any query: {query_list}")

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_url_summary_all])}
</ul>
</body></html>
"""

        pseudo_url = f"web-search://ts={int(time.time())}"
        return process_html(
            html=html_page,
            url=pseudo_url,
            title=title_str,
            display_urls=True,
            session=session,
        )
        
    async def fetch(self, url:str, session=None):
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX) :]
        
        data = await self._post(
            session,
            "/get_content",
            {"url": url},
        )
        
        if not data or not data.get("content"):
            raise BackendError(f"No content returned for {url}")
        return process_html(
            html=data.get("content", ""),
            url=url,
            title=data.get("title", ""),
            display_urls=True,
            session=session,
        )

class SerperServiceBrowserBackend:
    """Browser backend using Serper API for search and scraping."""
    source = "web"

    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.search_url = "https://google.serper.dev/search"
        self.scrape_url = "https://scrape.serper.dev/"

    async def _search_single(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> tuple[str, list]:
        """Execute a single search query and return query with results."""
        payload = {
            "q": query,
            "num": topn
        }
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        async with session.post(self.search_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Search error {resp.status}: {await resp.text()}"
                )
            data = await resp.json()

        results = data.get("organic", [])
        if not results:
            logger.warning(f"No results returned for query: '{query}'")
            return query, []

        title_url_summary = []
        for result in results:
            title_url_summary.append((
                html.escape(result.get('title', ''), quote=True),
                html.escape(result.get('link', ''), quote=True),
                html.escape(result.get('snippet', ''), quote=True)
            ))
        return query, title_url_summary

    async def search(
        self,
        query: Union[str, List[str]],
        topn: int = 5,
        session = None,
    ):
        """Search using Serper API. Supports single query or list of queries (parallel)."""
        # Handle single query
        if isinstance(query, str):
            query_list = [query]
            title_str = query
        else:
            query_list = query
            # Create a title with all query names
            title_str = " | ".join(query_list)

        # Execute searches in parallel
        tasks = [self._search_single(q, topn, session) for q in query_list]
        all_results = await asyncio.gather(*tasks)

        # Merge all results
        title_url_summary_all = []
        for query_str, title_url_summary in all_results:
            # Add results from each query
            if title_url_summary:
                title_url_summary_all.extend(title_url_summary)

        # If no results from any query, raise error
        if not title_url_summary_all:
            raise BackendError(f"No results returned for any query: {query_list}")

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_url_summary_all])}
</ul>
</body></html>
"""

        pseudo_url = f"web-search://ts={int(time.time())}"
        return process_html(
            html=html_page,
            url=pseudo_url,
            title=title_str,
            display_urls=True,
            session=session,
        )

    async def fetch(self, url: str, session=None):
        """Fetch and scrape a URL using Serper API."""
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX):]

        payload = {
            "url": url
        }
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        async with session.post(self.scrape_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Fetch error {resp.status}: {await resp.text()}"
                )
            data = await resp.json()

        # Sanitize the response data to remove any None keys
        data = sanitize_dict_keys(data)

        if not data:
            raise BackendError(f"No content returned for {url}")

        # Serper scrape API returns 'text' (extracted content) and 'metadata' (with title)
        text_content = data.get("text", "")
        metadata = data.get("metadata", {})
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""

        if not text_content:
            raise BackendError(f"No content returned for {url}")

        return process_html(
            html=text_content,
            url=url,
            title=title,
            display_urls=True,
            session=session,
        )


class SpecStreamServiceBrowserBackend:
    """Browser backend using SpecStream Search and Extract APIs."""
    source = "web"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else os.getenv("SPECSTREAM_API_KEY")
        self.search_url = "https://api.specstream.ai/v1/web-search"
        self.extract_url = "https://api.specstream.ai/v1/extract"

    def _headers(self) -> dict:
        if not self.api_key:
            raise BackendError("SPECSTREAM_API_KEY environment variable is not set")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _search_single(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> tuple[str, list]:
        """Execute a single search query and return query with results."""
        params = {
            "q": query,
            "num": max(1, min(int(topn), 100)),
            "speed": "fast",
        }

        async with session.get(self.search_url, params=params, headers=self._headers()) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Search error {resp.status}: {await resp.text()}"
                )
            data = await resp.json()

        results = (data.get("data") or {}).get("sources") or []
        if not results:
            logger.warning(f"No results returned for query: '{query}'")
            return query, []

        title_url_summary = [
            (
                html.escape(result.get("title") or "", quote=True),
                html.escape(result.get("url") or "", quote=True),
                html.escape(result.get("snippet") or "", quote=True),
            )
            for result in results
        ]
        return query, title_url_summary

    async def search(
        self,
        query: Union[str, List[str]],
        topn: int = 5,
        session=None,
    ):
        """Search using SpecStream API. Supports single query or list of queries (parallel)."""
        if isinstance(query, str):
            query_list = [query]
            title_str = query
        else:
            query_list = query
            title_str = " | ".join(query_list)

        tasks = [self._search_single(q, topn, session) for q in query_list]
        all_results = await asyncio.gather(*tasks)

        title_url_summary_all = []
        for _, title_url_summary in all_results:
            if title_url_summary:
                title_url_summary_all.extend(title_url_summary)

        if not title_url_summary_all:
            raise BackendError(f"No results returned for any query: {query_list}")

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_url_summary_all])}
</ul>
</body></html>
"""

        pseudo_url = f"web-search://ts={int(time.time())}"
        return process_html(
            html=html_page,
            url=pseudo_url,
            title=title_str,
            display_urls=True,
            session=session,
        )

    async def fetch(self, url: str, session=None):
        """Fetch and extract a URL using SpecStream Extract API."""
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX):]

        payload = {
            "urls": [url],
            "format": "html",
        }

        async with session.post(self.extract_url, json=payload, headers=self._headers()) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Fetch error {resp.status}: {await resp.text()}"
                )
            data = await resp.json()

        results = (data.get("data") or {}).get("results") or []
        if not results:
            raise BackendError(f"No content returned for {url}")

        result = sanitize_dict_keys(results[0])
        if not result:
            raise BackendError(f"No content returned for {url}")

        status = result.get("status")
        if status and status != "ok":
            raise BackendError(f"Fetch error for {url}: status={status}")

        text_content = result.get("text") or ""
        title = result.get("title") or ""

        if not text_content:
            raise BackendError(f"No content returned for {url}")

        return process_html(
            html=text_content,
            url=url,
            title=title,
            display_urls=True,
            session=session,
        )
