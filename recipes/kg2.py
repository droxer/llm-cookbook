from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from icecream import ic


llm = ChatOllama(
    model="sciphi/triplex",
    temperature=0.1,
)

prompts = """
Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.

        **Entity Types:**
        {entity_types}

        **Predicates:**
        {predicates}

        **Text:**
        {text}
"""

text = """
San Francisco,[24] officially the City and County of San Francisco, is a commercial, financial, and cultural center in Northern California. 

With a population of 808,437 residents as of 2022, San Francisco is the fourth most populous city in the U.S. state of California behind Los Angeles, San Diego, and San Jose.
"""

prompt_template = PromptTemplate(
    template=prompts,
    partial_variables={
        "entity_types": [
            "LOCATION",
            "POSITION",
            "DATE",
            "CITY",
            "COUNTRY",
            "NUMBER",
        ],
        "predicates": ["POPULATION", "AREA"],
    },
)


chain = prompt_template | llm

output = chain.invoke({"text": text})
ic(output)
