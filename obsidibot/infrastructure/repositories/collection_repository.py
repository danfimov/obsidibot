import chromadb


class CollectionRepository:
    def __init__(self, collection: chromadb.Collection) -> None:
        self._collection = collection

    def query(self, query: str, n_results: int = 1) -> chromadb.QueryResult:
        return self._collection.query(query_texts=[query], n_results=n_results)
