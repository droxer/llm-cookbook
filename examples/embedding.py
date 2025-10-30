from sentence_transformers import SentenceTransformer, util
from rich import print as rprint

from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer(
    # "BAAI/bge-large-zh-v1.5",
    "jinaai/jina-embeddings-v4",
    trust_remote_code=True,
)

# Corpus with example sentences
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]


# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

corpus_embeddings = corpus_embeddings.to("mps")
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)


for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    rprint("\nQuery:", query)
    rprint("Top 5 most similar sentences in corpus:")

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        rprint(corpus[hit["corpus_id"]], "(Score: {:.4f})".format(hit["score"]))


# 使用 jinaai/jina-embeddings-v3模型
"""
Query: A man is eating pasta.
Top 5 most similar sentences in corpus:
A man is eating food. (Score: 0.8442)
A man is eating a piece of bread. (Score: 0.7457)
A man is riding a horse. (Score: 0.4423)
A monkey is playing drums. (Score: 0.3901)
A man is riding a white horse on an enclosed ground. (Score: 0.3779)

Query: Someone in a gorilla costume is playing a set of drums.
Top 5 most similar sentences in corpus:
A monkey is playing drums. (Score: 0.8406)
A man is riding a horse. (Score: 0.5040)
A woman is playing violin. (Score: 0.4966)
A cheetah is running behind its prey. (Score: 0.4669)
A man is riding a white horse on an enclosed ground. (Score: 0.4631)

Query: A cheetah chases prey on across a field.
Top 5 most similar sentences in corpus:
A cheetah is running behind its prey. (Score: 0.9227)
A monkey is playing drums. (Score: 0.5145)
A man is riding a white horse on an enclosed ground. (Score: 0.4276)
Two men pushed carts through the woods. (Score: 0.4155)
A man is riding a horse. (Score: 0.4144)
"""

# 使用 BAAI/bge-large-zh-v1.5 模型
"""
Query: A man is eating pasta.
Top 5 most similar sentences in corpus:
A man is eating food. (Score: 0.8750)
A man is eating a piece of bread. (Score: 0.8173)
A man is riding a horse. (Score: 0.6221)
A man is riding a white horse on an enclosed ground. (Score: 0.5911)
A woman is playing violin. (Score: 0.4883)

Query: Someone in a gorilla costume is playing a set of drums.
Top 5 most similar sentences in corpus:
A monkey is playing drums. (Score: 0.6211)
A woman is playing violin. (Score: 0.5694)
A man is riding a white horse on an enclosed ground. (Score: 0.5404)
A man is riding a horse. (Score: 0.5376)
A man is eating a piece of bread. (Score: 0.5231)

Query: A cheetah chases prey on across a field.
Top 5 most similar sentences in corpus:
A cheetah is running behind its prey. (Score: 0.8325)
Two men pushed carts through the woods. (Score: 0.6193)
A monkey is playing drums. (Score: 0.5686)
A man is riding a horse. (Score: 0.5437)
A man is riding a white horse on an enclosed ground. (Score: 0.5280)
"""
