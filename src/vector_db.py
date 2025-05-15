from pymilvus import MilvusException, connections, db, utility, Collection
from langchain_milvus import BM25BuiltInFunction, Milvus
from loguru import logger
from uuid import uuid4
from langchain_core.documents import Document


class VectorDBManager:
    def __init__(
        self,
        embedder,
        collection_name,
        db_name="milvus_demo",
        host="127.0.0.1",
        port=19530,
        token="root:Milvus",
        drop_old=False,
        dense_index_param=None,
        sparse_index_param=None,
    ):
        self.db_name = db_name
        self.collection_name = collection_name
        self.embedder = embedder
        self.host = host
        self.port = port
        self.token = token
        self.drop_old = drop_old

        self.dense_index_param = dense_index_param or {
            "metric_type": "COSINE",
            "index_type": "HNSW",
        }
        self.sparse_index_param = sparse_index_param or {
            "metric_type": "BM25",
            "index_type": "AUTOINDEX",
        }

        connections.connect(host=self.host, port=self.port)
        try:
            existing_databases = db.list_database()
            if self.db_name not in existing_databases:
                db.create_database(self.db_name)
            db.using_database(self.db_name)
        except MilvusException as e:
            logger.error(f"Error creating or using database: {e}")
            raise

        # Warn if drop_old is True and collection exists
        all_collections = utility.list_collections()
        if self.drop_old and self.collection_name in all_collections:
            logger.warning(
                f"drop_old=True: Collection '{self.collection_name}' will be dropped and recreated. All previous data will be lost."
            )

        vectordb_config = {
            "uri": f"http://{self.host}:{self.port}",
            "token": self.token,
            "db_name": self.db_name,
        }

        self.vector_db = Milvus(
            embedding_function=self.embedder,
            connection_args=vectordb_config,
            consistency_level="Strong",
            drop_old=self.drop_old,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            index_params=[self.dense_index_param, self.sparse_index_param],
            collection_name=self.collection_name,
        )

    def add_documents(self, docs: list[Document], uuids: list[str] = None):
        """
        Adds documents to the vector database.
        If uuids are not provided, generates new ones.
        """
        if uuids is None:
            uuids = [str(uuid4()) for _ in range(len(docs))]
        self.vector_db.add_documents(
            documents=docs,
            ids=uuids,
        )
        logger.info(
            f"Added {len(docs)} documents to the collection '{self.collection_name}'."
        )

    def retrieve_similar(self, query, k=2, method="weighted", ranker_params=None):
        """
        Retrieves similar documents from the vector database using different methods.
        Supported methods:
            - "weighted": Uses a weighted ranker.
            - "rrf": Uses reciprocal rank fusion.
        """
        if method == "weighted":
            params = ranker_params or {"weights": [0.5, 0.5]}
            return self.vector_db.similarity_search(
                query,
                k=k,
                ranker_type="weighted",
                ranker_params=params,
            )
        elif method == "rrf":
            params = ranker_params or {"k": 100}
            return self.vector_db.similarity_search(
                query,
                k=k,
                ranker_type="rrf",
                ranker_params=params,
            )
        else:
            raise ValueError(f"Unsupported retrieval method: {method}")

    def check_collections(self):
        """
        Checks if the specified collection exists in the current database.
        Returns True if it exists, False otherwise.
        """
        db.using_database(self.db_name)
        all_collections = utility.list_collections()
        logger.info(f"All collections in '{self.db_name}': {all_collections}")
        if self.collection_name in all_collections:
            logger.info(f"Collection '{self.collection_name}' exists.")
            return True
        else:
            logger.warning(f"Collection '{self.collection_name}' does not exist.")
            return False

    def drop_collections(self, collections_to_drop):
        """
        Drops one or more collections from the current database after checking their existence.
        Logs the collections that were dropped and lists remaining collections.
        """
        db.using_database(self.db_name)
        all_collections = utility.list_collections()
        if isinstance(collections_to_drop, str):
            collections_to_drop = [collections_to_drop]

        for collection_name in collections_to_drop:
            if collection_name in all_collections:
                collection = Collection(name=collection_name)
                collection.drop()
                logger.info(f"Collection '{collection_name}' has been dropped.")
            else:
                logger.warning(
                    f"Collection '{collection_name}' does not exist and cannot be dropped."
                )

        remaining_collections = utility.list_collections()
        logger.info(f"Remaining collections: {remaining_collections}")
