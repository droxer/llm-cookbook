import os
import httpx
from typing import *

import json

from openai import OpenAI


client = OpenAI(
    api_key=os.environ.get("MOONSHOT_API_KEY"),
    base_url=os.environ.get("MOONSHOT_API_URL"),
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": """ 
				通过搜索引擎搜索互联网上的内容。
 
				当你的知识无法回答用户提出的问题，或用户请求你进行联网搜索时，调用此工具。请从与用户的对话中提取用户想要搜索的内容作为 query 参数的值。
				搜索结果包含网站的标题、网站的地址（URL）以及网站简介。
			""",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": """
							用户搜索的内容，请从用户的提问或聊天上下文中提取。
						""",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crawl",
            "description": """
				根据网站地址（URL）获取网页内容。
			""",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": """
							需要获取内容的网站地址（URL），通常情况下从搜索结果中可以获取网站的地址。
						""",
                    }
                },
            },
        },
    },
]


def search_impl(query: str) -> List[Dict[str, Any]]:
    """
    search_impl 使用搜索引擎对 query 进行搜索，目前主流的搜索引擎（例如 Bing）都提供了 API 调用方式，你可以自行选择
    你喜欢的搜索引擎 API 进行调用，并将返回结果中的网站标题、网站链接、网站简介信息放置在一个 dict 中返回。

    这里只是一个简单的示例，你可能需要编写一些鉴权、校验、解析的代码。
    """
    r = httpx.get("https://serpapi.com/", params={"search": query})
    return r.json()


def search(arguments: Dict[str, Any]) -> Any:
    query = arguments["query"]
    result = search_impl(query)
    return {"result": result}


def crawl_impl(url: str) -> str:
    """
    crawl_url 根据 url 获取网页上的内容。

    这里只是一个简单的示例，在实际的网页抓取过程中，你可能需要编写更多的代码来适配复杂的情况，例如异步加载的数据等；同时，在获取
    网页内容后，你可以根据自己的需要对网页内容进行清洗，只保留文本或移除不必要的内容（例如广告信息等）。
    """
    r = httpx.get(url)
    return r.text


def crawl(arguments: dict) -> str:
    url = arguments["url"]
    content = crawl_impl(url)
    return {"content": content}


tool_map = {
    "search": search,
    "crawl": crawl,
}

messages = [
    {
        "role": "system",
        "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
    },
    {
        "role": "user",
        "content": "请联网搜索 Context Caching，并告诉我它是什么。",
    },
]

finish_reason = None

while finish_reason is None or finish_reason == "tool_calls":
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0.3,
        tools=tools,  #
    )
    choice = completion.choices[0]
    finish_reason = choice.finish_reason
    if finish_reason == "tool_calls":
        messages.append(choice.message)
        for tool_call in choice.message.tool_calls:
            tool_call_name = tool_call.function.name
            tool_call_arguments = json.loads(tool_call.function.arguments)
            tool_function = tool_map[tool_call_name]
            tool_result = tool_function(tool_call_arguments)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result),
                }
            )

print(choice.message.content)
