import os
from typing import Dict, List

from mem0 import MemoryClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from dotenv import load_dotenv

load_dotenv()

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "db",
        }
    }
}



mem0 = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
llm = ChatOpenAI(model="qwen-max", api_key=os.getenv("BAILIAN_API_KEY"), base_url=os.getenv("BAILIAN_API_URL"))

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful travel agent AI. Use the provided context to personalize your responses and remember user preferences and past interactions. 
    Provide travel recommendations, itinerary suggestions, and answer questions about destinations. 
    If you don't have specific information, you can make general suggestions based on common travel knowledge."""),
    MessagesPlaceholder(variable_name="context"),
    HumanMessage(content="{input}")
])


def retrieve_context(query: str, user_id: str) -> List[Dict]:
    """Retrieve relevant context from Mem0"""
    memories = mem0.search(query, user_id=user_id, api_version="v1.1")
    seralized_memories = ' '.join([mem["memory"] for mem in memories])
    context = [
        {
            "role": "system", 
            "content": f"Relevant information: {seralized_memories}"
        },
        {
            "role": "user",
            "content": query
        }
    ]
    return context

def generate_response(input: str, context: List[Dict]) -> str:
    """Generate a response using the language model"""
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "input": input
    })
    return response.content

def save_interaction(user_id: str, user_input: str, assistant_response: str):
    """Save the interaction to Mem0"""
    interaction = [
        {
          "role": "user",
          "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    mem0.add(interaction, user_id=user_id, api_version="v1.1")


def chat_turn(user_input: str, user_id: str) -> str:
    # Retrieve context
    context = retrieve_context(user_input, user_id)
    
    # Generate response
    response = generate_response(user_input, context)
    
    # Save interaction
    save_interaction(user_id, user_input, response)
    
    return response

if __name__ == "__main__":
    print("Welcome to your personal Travel Agent Planner! How can I assist you with your travel plans today?")
    user_id = "Michael"
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Travel Agent: Thank you for using our travel planning service. Have a great trip!")
            break
        
        response = chat_turn(user_input, user_id)
        print(f"Travel Agent: {response}")    