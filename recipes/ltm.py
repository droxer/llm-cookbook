import os
from icecream import ic

from mem0 import MemoryClient


from dotenv import load_dotenv

load_dotenv()

client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

# messages = [
#     {
#         "role": "user",
#         "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts.",
#     },
#     {
#         "role": "assistant",
#         "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions.",
#     },
# ]

# client.add(messages, user_id="alex")


query = "What can I cook for dinner tonight?"
results = client.search(query, user_id="alex")
ic(results)