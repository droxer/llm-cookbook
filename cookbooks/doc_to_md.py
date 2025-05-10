import os

from markitdown import MarkItDown
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
md = MarkItDown(llm_client=client, llm_model="gpt-4o-mini")
result = md.convert("images/1280X1280.JPEG")
print(result.text_content)
