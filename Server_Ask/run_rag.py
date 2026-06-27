from common.defaults import DEFAULT_DOCUMENTS


def offline_get_shared_store():
    shared = {
        "texts": list(DEFAULT_DOCUMENTS),
        "embeddings": None,
        "query": None,
        "query_embedding": None,
        "retrieved_document": None,
        "generated_answer": None
    }
    return shared


def online_get_shared_store(query):
    shared = {
        "texts": list(DEFAULT_DOCUMENTS),
        "query": query,
        "query_embedding": None,
        "retrieved_document": None,
        "generated_answer": None
    }
    return shared
