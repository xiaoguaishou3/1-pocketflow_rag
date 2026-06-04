"""
@File   : utils
@Author : 74775
@Date   : 2026/5/19 13:39
"""
import os
import numpy as np
from openai import OpenAI

def call_llm(prompt):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"),
        base_url="https://api.xiaomimimo.com/v1"
    )
    r = client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[{"role": "user", "content": prompt}]
    )
    return  r.choices[0].message.content

def get_embedding(text):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    embedding = response.data[0].embedding

    return np.array(embedding, dtype=np.float32)

def get_embeddings_2(texts):
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    vec = _model.encode(texts, normalize_embeddings=True)  # 归一化后 L2 距离 ≈ 余弦相似度
    return np.asarray(vec, dtype=np.float32)

if __name__ == '__main__':
    print("=== Testing embedding function ===")

    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Python is a popular programming language for data science."

    # oai_emb1= get_embedding(text1)
    # print(f"可以获得什么结果：{oai_emb1}")

    res = get_embeddings_2(text1)
    print(f"可以获得什么结果：{res}")
