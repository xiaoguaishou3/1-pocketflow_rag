from rag_logic.flow import online_flow

DEFAULT_DOCUMENTS = [
    """Pocket Flow is a 100-line minimalist LLM framework
    Lightweight: Just 100 lines. Zero bloat, zero dependencies, zero vendor lock-in.
    Expressive: Everything you love—(Multi-)Agents, Workflow, RAG, and more.
    Agentic Coding: Let AI Agents (e.g., Cursor AI) build Agents—10x productivity boost!
    To install, pip install pocketflow or just copy the source code (only 100 lines).""",

    """NeurAlign M7 is a revolutionary non-invasive neural alignment device.
    Targeted magnetic resonance technology increases neuroplasticity in specific brain regions.
    Clinical trials showed 72% improvement in PTSD treatment outcomes.
    Developed by Cortex Medical in 2024 as an adjunct to standard cognitive therapy.
    Portable design allows for in-home use with remote practitioner monitoring.""",

    """The Velvet Revolution of Caldonia (1967-1968) ended Generalissimo Verak's 40-year rule.
    Led by poet Eliza Markovian through underground literary societies.
    Culminated in the Great Silence Protest with 300,000 silent protesters.
    First democratic elections held in March 1968 with 94% voter turnout.
    Became a model for non-violent political transitions in neighboring regions.""",

    """Q-Mesh is QuantumLeap Technologies' instantaneous data synchronization protocol.
    Utilizes directed acyclic graph consensus for 500,000 transactions per second.
    Consumes 95% less energy than traditional blockchain systems.
    Adopted by three central banks for secure financial data transfer.
    Released in February 2024 after five years of development in stealth mode.""",

    """Harlow Institute's Mycelium Strain HI-271 removes 99.7% of PFAS from contaminated soil.
    Engineered fungi create symbiotic relationships with native soil bacteria.
    Breaks down "forever chemicals" into non-toxic compounds within 60 days.
    Field tests successfully remediated previously permanently contaminated industrial sites.
    Deployment costs 80% less than traditional chemical extraction methods."""
]


def get_default_documents():
    return list(DEFAULT_DOCUMENTS)


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
