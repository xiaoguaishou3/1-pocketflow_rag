"""
@File   : utils
@Author : 74775
@Date   : 2026/5/19 13:39
"""
import os
import numpy as np
from openai import OpenAI

def call_llm(prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
    r = client.chat.completions.create(
        model="gpt-4o",
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

    return np.array(embedding, dytpe=np.float32)

if __name__ == '__main__':
    print("=== Testing embedding function ===")

    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Python is a popular programming language for data science."

    oai_emb1= get_embedding(text1)
    print(f"可以获得什么结果：{oai_emb1}")
