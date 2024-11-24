import os

# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI

from langchain.agents import load_tools
from langchain.agents import initialize_agent

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
tools = load_tools(["google-serper", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("奥利维亚·王尔德的男朋友是谁?他现在的年龄的0.23次方是多少?")
