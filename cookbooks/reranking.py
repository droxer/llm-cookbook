from sentence_transformers import CrossEncoder
from rich import print as rprint

model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    # "jinaai/jina-reranker-v2-base-multilingual",
    model_kwargs={"torch_dtype": "auto"},
    trust_remote_code=True,
)

# Example query and documents
query = "Organic skincare products for sensitive skin"
documents = [
    "Organic skincare for sensitive skin with aloe vera and chamomile.",
    "New makeup trends focus on bold colors and innovative techniques",
    "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
    "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
    "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
    "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
    "针对敏感肌专门设计的天然有机护肤产品",
    "新的化妆趋势注重鲜艳的颜色和创新的技巧",
    "敏感肌のために特別に設計された天然有機スキンケア製品",
    "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
]

# construct sentence pairs
sentence_pairs = [[query, doc] for doc in documents]

scores = model.predict(sentence_pairs, convert_to_tensor=True).tolist()
"""
[0.828125, 0.0927734375, 0.6328125, 0.08251953125, 0.76171875, 0.099609375, 0.92578125, 0.058349609375, 0.84375, 0.111328125]
"""

rankings = model.rank(query, documents, return_documents=True, convert_to_tensor=True)
rprint(f"Query: {query}")
for ranking in rankings:
    rprint(
        f"ID: {ranking['corpus_id']}, Score: {ranking['score']:.4f}, Text: {ranking['text']}"
    )

# 使用 jinaai/jina-reranker-v2-base-multilingual 模型
"""
Query: Organic skincare products for sensitive skin
ID: 6, Score: 0.9258, Text: 针对敏感肌专门设计的天然有机护肤产品
ID: 8, Score: 0.8438, Text: 敏感肌のために特別に設計された天然有機スキンケア製品
ID: 0, Score: 0.8281, Text: Organic skincare for sensitive skin with aloe vera and chamomile.
ID: 4, Score: 0.7617, Text: Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla
ID: 2, Score: 0.6289, Text: Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille
ID: 9, Score: 0.1128, Text: 新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています
ID: 5, Score: 0.0996, Text: Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras
ID: 1, Score: 0.0952, Text: New makeup trends focus on bold colors and innovative techniques
ID: 3, Score: 0.0825, Text: Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken
ID: 7, Score: 0.0593, Text: 新的化妆趋势注重鲜艳的颜色和创新的技巧
"""

# 使用 bge-reranker-v2-m3 模型
"""
Query: Organic skincare products for sensitive skin
ID: 8, Score: 0.9988, Text: 敏感肌のために特別に設計された天然有機スキンケア製品
ID: 6, Score: 0.9970, Text: 针对敏感肌专门设计的天然有机护肤产品
ID: 0, Score: 0.9882, Text: Organic skincare for sensitive skin with aloe vera and chamomile.
ID: 4, Score: 0.9786, Text: Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla
ID: 2, Score: 0.7314, Text: Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille
ID: 9, Score: 0.0000, Text: 新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています
ID: 7, Score: 0.0000, Text: 新的化妆趋势注重鲜艳的颜色和创新的技巧
ID: 5, Score: 0.0000, Text: Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras
ID: 1, Score: 0.0000, Text: New makeup trends focus on bold colors and innovative techniques
ID: 3, Score: 0.0000, Text: Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken
"""