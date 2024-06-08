from dotenv import load_dotenv

load_dotenv()


from langchain_openai import OpenAIEmbeddings
import numpy as np

from numpy import dot
from numpy.linalg import norm


def cos_sim(a, b):
    """余弦距离 -- 越大越相似"""
    return dot(a, b) / (norm(a) * norm(b))


def l2(a, b):
    """欧式距离 -- 越小越相似"""
    x = np.asarray(a) - np.asarray(b)
    return norm(x)


embedding = OpenAIEmbeddings()


if __name__ == "__main__":
    sentence1_chinese = "我喜欢狗"
    sentence2_chinese = "我喜欢犬科动物"
    sentence3_chinese = "外面的天气很糟糕"

    embedding1_chinese = embedding.embed_query(sentence1_chinese)
    embedding2_chinese = embedding.embed_query(sentence2_chinese)
    embedding3_chinese = embedding.embed_query(sentence3_chinese)

    print(l2(embedding1_chinese, embedding2_chinese))
    print(l2(embedding1_chinese, embedding3_chinese))
