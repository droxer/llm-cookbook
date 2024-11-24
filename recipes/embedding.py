from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from icecream import ic

model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-zh",
    # "BAAI/bge-large-zh-v1.5",
    # automodel_args={"torch_dtype": "auto"},
    # trust_remote_code=True,
)

# control your input sequence length up to 8192
model.max_seq_length = 1024

embeddings = model.encode(["How is the weather today?", "今天天气怎么样?"])
ic(cos_sim(embeddings[0], embeddings[1]))

similarities = model.similarity(embeddings[0], embeddings[1])
ic(similarities)
