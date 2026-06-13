"""
@File   : utils
@Author : 74775
@Date   : 2026/5/19 13:39
"""
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(r"D:\Project\practice_many\PocketFlow\1-pocketflow_rag\.env")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE")
BGE_MODEL_REPO = os.getenv("BGE_MODEL_REPO")
DEFAULT_BGE_MODEL = "BAAI/bge-small-zh-v1.5"
_embedding_model = None


def _resolve_bge_model_path():
    """Prefer a local Hugging Face cache snapshot; fall back to model repo id."""
    if HF_HUB_CACHE and BGE_MODEL_REPO:
        refs_main = os.path.join(HF_HUB_CACHE, BGE_MODEL_REPO, "refs", "main")
        with open(refs_main, encoding="utf-8") as f:
            revision = f.read().strip()
        return os.path.join(HF_HUB_CACHE, BGE_MODEL_REPO, "snapshots", revision)
    return DEFAULT_BGE_MODEL


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        model_path = _resolve_bge_model_path()
        _embedding_model = SentenceTransformer(model_path)
    return _embedding_model

def call_llm(prompt):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
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
    _model = _get_embedding_model()
    vec = _model.encode(texts, normalize_embeddings=True)  # 归一化后 L2 距离 ≈ 余弦相似度
    return np.asarray(vec, dtype=np.float32)

def fix_size_chunk(text, chunk_size=2000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

if __name__ == '__main__':
    # client = OpenAI(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url="https://api.xiaomimimo.com/v1"
    # )
    # res = client.models.list()
    # print(res)
    # print("="*20)

    # print("=== Testing call_llm ===")
    # prompt = "In a few words, what is the meaning of life?"
    # print(f"Prompt: {prompt}")
    # response = call_llm(prompt)
    # print(f"Response: {response}")

    print("=== Testing embedding function ===")

    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Python is a popular programming language for data science."

    emb1 = get_embeddings_2(text1)
    emb2 = get_embeddings_2(text2)
    print(f"shape: {emb1.shape}, dtype: {emb1.dtype}")
    print(f"similarity (dot product): {float(np.dot(emb1, emb2)):.4f}")
