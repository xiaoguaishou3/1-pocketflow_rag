"""
@File   : nodes
@Author : 74775
@Date   : 2026/5/16 21:32
"""
from pocketflow import Node, BatchNode
import numpy as np
import faiss
from .utils import get_embeddings_2, fix_size_chunk, call_llm


# Nodes for the offline flow
class ChunkDocumentsNode(BatchNode):
    def prep(self, shared):
        return shared["texts"]

    def exec(self, text):
        """Chunk a single text into smaller pieces"""
        return fix_size_chunk(text)

    def post(self, shared, prep_res, exec_res_list):
        all_chunks = []
        for chunks in exec_res_list:
            all_chunks.extend(chunks)

        shared["texts"] =  all_chunks

        print(f"✅ Create {len(all_chunks)} chunks from {len(prep_res)} documents")
        return "default"

class EmbedDocumentsNode(BatchNode):
    def prep(self, shared):
        return shared["texts"]

    def exec(self, text):
        """Embed a single text"""
        return get_embeddings_2(text)

    def post(self, shared, prep_res, exec_res_list):
        embeddings = np.array(exec_res_list, dtype=np.float32)
        shared["embeddings"] = embeddings
        print(f"✅ Create {len(embeddings)}")
        print(f"✅ Create {len(embeddings)} document embeddings")
        return "default"

class CreateIndexNode(Node):
    def prep(self, shared):
        return shared["embeddings"]

    def exec(self, embeddings):
        print("🔍 Creating search index ...")
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        return index

    def post(self, shared, prep_res, exec_res):
        shared["index"] = exec_res
        print(f"✅ Index created with {exec_res.ntotal}")
        return "default"

# Nodes for the online flow
class EmbedQueryNode(Node):
    def prep(self, shared):
        return shared["query"]

    def exec(self, query):
        print(f"🔍 Embedding query: {query}")
        query_embedding = get_embeddings_2(query)
        return np.array([query_embedding])

    def post(self, shared, prep_res, exec_res):
        shared["query_embedding"] = exec_res
        return "default"

class RetrieveDocumentNode(Node):
    def prep(self, shared):
        return  shared["query_embedding"], shared["index"], shared["texts"]

    def exec(self, inputs):
        print("🔍 Search for relevant documents...")
        query_embedding, index, texts = inputs

        distances, indices = index.search(query_embedding, k=1)

        best_idx = indices[0][0]
        distance = distances[0][0]

        most_relevant_text = texts[best_idx]

        return {
            "text": most_relevant_text,
            "index": best_idx,
            "distance": distance
        }

    def post(self, shared, prep_res, exec_res):
        shared["retrieved_document"] = exec_res
        print(f"📄 Retrieved document (index: {exec_res['index']}, distance: {exec_res['distance']:.4f})")
        print(f"📄 Most relevant text: \"{exec_res['text']}\"")
        return "default"

class GenerateAnswerNode(Node):
    def prep(self, shared):
        return shared["query"], shared["retrieved_document"]

    def exec(self, inputs):
        query, retrieved_doc = inputs

        prompt = f"""
        Briefly answer the following question based on the context provided:
        Question: {query}
        Context: {retrieved_doc['text']}
        Answer:
        """

        answer = call_llm(prompt)
        return  answer

    def post(self, shared, prep_res, exec_res):
        shared["generated_answer"] = exec_res
        print("\n🤖 Generated Answer:")
        print(exec_res)
        return "default"

