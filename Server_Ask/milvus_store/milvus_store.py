"""
@File   : milvus_store
@Author : 74775
@Date   : 2026/7/9 22:41
"""
from pymilvus import MilvusClient, Collection, CollectionSchema, DataType, FieldSchema
from .config import COLLECTION_NAME, VECTOR_DIM


class MilvusStore:
    def __init__(self, uri: str):
        self.client = MilvusClient(uri)
        self.collection_schema = None


    def get_milvus_version(self) -> str:
        """get Milvus version"""
        return self.client.get_server_version()


    def is_connected(self) -> bool:
        """check connection"""
        try:
            self.get_milvus_version()
            return True
        except Exception:
            return False


    def create_schema(self, fields_config: dict = None):
        if not fields_config:
            fields_config = [
                {
                    "name": "id",
                    "dtype": DataType.INT64,
                    "is_primary": True,
                    "auto_id": True
                },
                {
                    "name": "text",
                    "dtype": DataType.VARCHAR,
                    "max_length": 65535,
                },
                {
                    "name": "embedding",
                    "dtype": DataType.FLOAT_VECTOR,
                    "dim": VECTOR_DIM,
                },
                {
                    "name": "doc_hash",
                    "dtype": DataType.VARCHAR,
                    "max_length": 64,
                },
                {
                    "name": "chunk_index",
                    "dtype": DataType.INT64,
                },
                {
                    "name": "session_ids",
                    "dtype": DataType.VARCHAR,
                    "max_length": 4096,
                },
            ]
        fields = []
        for field in fields_config:
            field_schema = FieldSchema(
                name=field["name"],
                dtype=field["dtype"],
                **{k: v for k, v in field.items() if k not in ["name", "dtype"]}
            )
            fields.append(field_schema)
        collection_schema = CollectionSchema(
            fields=fields,
            enable_dynamic_field=False,
        )
        return collection_schema


    def create_collection(self, name: str = COLLECTION_NAME, dim: int = VECTOR_DIM):
        if self.client.has_collection(name):
            print(f"the collection named {name} is existed")

        else:
            self.collection_schema = self.create_schema()
            self.client.create_collection(
                collection_name=name,
                schema=self.collection_schema,
            )

    def get_collection(self, name: str = COLLECTION_NAME) -> bool:
        return self.client.has_collection(name)


    def drop_collection(self, name: str = COLLECTION_NAME) -> bool:
        if self.get_collection(name):
            self.client.drop_collection(name)


    def insert_vectors(
        self,
        texts,
        embeddings,
        session_id: str,
        chunk_indices,
        doc_hash,
        collection_name: str = COLLECTION_NAME,
    ):
        data = [texts, embeddings, doc_hash, [session_id] * len(texts), chunk_indices]
        res = self.client.insert(collection_name, data)

        print(f"insert result:\n{res}")


    def search_vectors(self, collection_name: str, query_embedding, session_id = ""):
        search_params = {
            "metric_type": "L2",
            # IVF_FLAT 索引的搜索参数
            "params": {"nprobe": 16}
        }
        con1 = f'session_ids == "{session_id}"' if session_id else None

        res = self.client.search(
            collection_name,
            query_embedding,
            search_params,
            limit=3,
            filter=con1,
            output_fields=["text", "session_ids", "chunk_index"]
        )

        formatted_results = []
        for hits in res:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "text": hit.entity.get("text"),
                    "session_ids": hit.entity.get("session_ids"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "distance": hit.distance
                })

        return formatted_results


    def delete_vectors(self, collection_name, vector_ids):
        res = self.client.delete(collection_name, vector_ids)
        print(f"delete_count: {res['delete_count']}")


    def count_vectors(self, collection_name):
        stats = self.client.get_collection_stats(collection_name)
        return stats.get("row_count", None)
