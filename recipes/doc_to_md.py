import os

from markitdown import MarkItDown
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url=os.getenv("BAILIAN_API_URL"),
)
md = MarkItDown(llm_client=client, llm_model="qwen-vl-max")
result = md.convert("images/1280X1280.JPEG")
print(result.text_content)
