import os

from autogen import ConversableAgent

cathy = ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians.",
    llm_config={
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }
        ]
    },
    human_input_mode="NEVER",
)

joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config={
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "api_key": os.environ.get("OPENAI_API_KEY"),
            }
        ]
    },
    human_input_mode="NEVER",
)

result = joe.initiate_chat(cathy, message="Cathy, 给我讲一个笑话.", max_turns=5)
