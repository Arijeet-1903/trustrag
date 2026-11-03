import json
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from trustrag.chunking.chunk_schema import Chunk
from trustrag.indexing.embeddings import BaseEmbeddingModel, get_embedding_model


class VectorStore:
    """
    Unified vector store interface supporting both ChromaDB and an in-memory
    NumPy store for fast, deterministic unit testing and production execution.
    """

    def __init__(
        self,
        collection_name: str = "trustrag_chunks",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        use_in_memory: bool = False
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or get_embedding_model("deterministic_mock")
        self.use_in_memory = use_in_memory

        self._in_memory_chunks: Dict[str, Chunk] = {}
        self._in_memory_embeddings: Dict[str, np.ndarray] = {}

        self._chroma_client = None
        self._chroma_collection = None

        if not self.use_in_memory:
            self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings
            if self.persist_directory:
                self._chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._chroma_client = chromadb.Client()
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # Fall back to in-memory store if Chroma is unavailable
            self.use_in_memory = True

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """
        Idempotently insert or update chunks in the vector store.
        Returns the number of chunks added/updated.
        """
        if not chunks:
            return 0

        # Ensure embeddings are populated
        texts_to_embed = []
        indices_to_embed = []
        for idx, chunk in enumerate(chunks):
            if chunk.embedding is None or len(chunk.embedding) == 0:
                texts_to_embed.append(chunk.text)
                indices_to_embed.append(idx)

        if texts_to_embed:
            vectors = self.embedding_model.embed_batch(texts_to_embed)
            for chunk_idx, vec in zip(indices_to_embed, vectors):
                chunks[chunk_idx].embedding = vec

        if self.use_in_memory or self._chroma_collection is None:
            for chunk in chunks:
                self._in_memory_chunks[chunk.chunk_id] = chunk
                self._in_memory_embeddings[chunk.chunk_id] = np.array(chunk.embedding, dtype=np.float32)
            return len(chunks)

        # Chroma upsert
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = []
        for c in chunks:
            meta = c.to_metadata()
            # Store full chunk JSON string for lossless reconstruction
            meta["raw_chunk_json"] = c.model_dump_json(exclude={"embedding"})
            metadatas.append(meta)

        self._chroma_collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(chunks)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a specific chunk by its chunk_id."""
        if self.use_in_memory or self._chroma_collection is None:
            return self._in_memory_chunks.get(chunk_id)

        res = self._chroma_collection.get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])
        if not res or not res.get("ids") or len(res["ids"]) == 0:
            return None

        meta = res["metadatas"][0]
        if "raw_chunk_json" in meta:
            chunk_data = json.loads(meta["raw_chunk_json"])
            if res.get("embeddings") is not None and len(res["embeddings"]) > 0:
                chunk_data["embedding"] = list(res["embeddings"][0])
            return Chunk(**chunk_data)

        # Reconstruct from metadata if raw_chunk_json is missing
        return Chunk(
            chunk_id=chunk_id,
            document_id=meta.get("document_id", ""),
            version=meta.get("version", "1.0"),
            chunk_type=meta.get("chunk_type", "paragraph"),
            section_path=meta.get("section_path", ""),
            page_start=int(meta.get("page_start", 1)),
            page_end=int(meta.get("page_end", 1)),
            table_id=meta.get("table_id") or None,
            text=res["documents"][0],
            content_hash=meta.get("content_hash", ""),
            source_uri=meta.get("source_uri") or None,
            embedding=res["embeddings"][0] if res.get("embeddings") is not None else None
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for relevant chunks using cosine similarity.
        Returns a list of (Chunk, similarity_score) tuples, highest score first.
        """
        if top_k <= 0:
            return []

        query_vec = np.array(self.embedding_model.embed_text(query), dtype=np.float32)

        if self.use_in_memory or self._chroma_collection is None:
            results = []
            for cid, chunk in self._in_memory_chunks.items():
                # Apply filters
                if filters:
                    match = True
                    for k, v in filters.items():
                        chunk_val = getattr(chunk, k, None)
                        if chunk_val != v:
                            match = False
                            break
                    if not match:
                        continue

                c_vec = self._in_memory_embeddings[cid]
                # Cosine similarity
                norm_q = np.linalg.norm(query_vec)
                norm_c = np.linalg.norm(c_vec)
                sim = float(np.dot(query_vec, c_vec) / (norm_q * norm_c)) if (norm_q > 0 and norm_c > 0) else 0.0
                results.append((chunk, sim))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        # Chroma query
        where_clause = None
        if filters:
            if len(filters) == 1:
                k, v = next(iter(filters.items()))
                where_clause = {k: v}
            elif len(filters) > 1:
                where_clause = {"$and": [{k: v} for k, v in filters.items()]}

        res = self._chroma_collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "embeddings", "distances"]
        )

        matched = []
        if res and res.get("ids") and len(res["ids"][0]) > 0:
            for idx in range(len(res["ids"][0])):
                cid = res["ids"][0][idx]
                meta = res["metadatas"][0][idx]
                dist = res["distances"][0][idx] if res.get("distances") else 0.0
                # Chroma cosine distance = 1 - cosine_similarity
                similarity = max(0.0, 1.0 - dist)

                if "raw_chunk_json" in meta:
                    chunk_data = json.loads(meta["raw_chunk_json"])
                    if res.get("embeddings") is not None and len(res["embeddings"]) > 0:
                        chunk_data["embedding"] = list(res["embeddings"][0][idx])
                    chunk = Chunk(**chunk_data)
                else:
                    chunk = Chunk(
                        chunk_id=cid,
                        document_id=meta.get("document_id", ""),
                        version=meta.get("version", "1.0"),
                        chunk_type=meta.get("chunk_type", "paragraph"),
                        section_path=meta.get("section_path", ""),
                        page_start=int(meta.get("page_start", 1)),
                        page_end=int(meta.get("page_end", 1)),
                        table_id=meta.get("table_id") or None,
                        text=res["documents"][0][idx],
                        content_hash=meta.get("content_hash", ""),
                        source_uri=meta.get("source_uri") or None
                    )
                matched.append((chunk, float(similarity)))

        return matched

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """Delete specific chunks by ID."""
        if not chunk_ids:
            return 0

        if self.use_in_memory or self._chroma_collection is None:
            removed = 0
            for cid in chunk_ids:
                if cid in self._in_memory_chunks:
                    del self._in_memory_chunks[cid]
                    del self._in_memory_embeddings[cid]
                    removed += 1
            return removed

        self._chroma_collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def delete_document(self, document_id: str, version: Optional[str] = None) -> int:
        """Remove all chunks associated with a document (and optionally a specific version)."""
        if self.use_in_memory or self._chroma_collection is None:
            to_delete = [
                cid for cid, chunk in self._in_memory_chunks.items()
                if chunk.document_id == document_id and (version is None or chunk.version == version)
            ]
            for cid in to_delete:
                del self._in_memory_chunks[cid]
                del self._in_memory_embeddings[cid]
            return len(to_delete)

        where = {"document_id": document_id}
        if version:
            where = {"$and": [{"document_id": document_id}, {"version": version}]}

        # Find matching chunk IDs first
        matched = self._chroma_collection.get(where=where, include=["metadatas"])
        if matched and matched.get("ids"):
            self._chroma_collection.delete(ids=matched["ids"])
            return len(matched["ids"])
        return 0

    def list_chunks(self, filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """List chunks matching filters."""
        if self.use_in_memory or self._chroma_collection is None:
            if not filters:
                return list(self._in_memory_chunks.values())
            res = []
            for chunk in self._in_memory_chunks.values():
                match = True
                for k, v in filters.items():
                    if getattr(chunk, k, None) != v:
                        match = False
                        break
                if match:
                    res.append(chunk)
            return res

        where = None
        if filters:
            if len(filters) == 1:
                k, v = next(iter(filters.items()))
                where = {k: v}
            elif len(filters) > 1:
                where = {"$and": [{k: v} for k, v in filters.items()]}

        res = self._chroma_collection.get(where=where, include=["metadatas", "documents"])
        chunks = []
        if res and res.get("ids"):
            for idx, cid in enumerate(res["ids"]):
                meta = res["metadatas"][idx]
                if "raw_chunk_json" in meta:
                    chunk = Chunk(**json.loads(meta["raw_chunk_json"]))
                else:
                    chunk = Chunk(
                        chunk_id=cid,
                        document_id=meta.get("document_id", ""),
                        version=meta.get("version", "1.0"),
                        chunk_type=meta.get("chunk_type", "paragraph"),
                        section_path=meta.get("section_path", ""),
                        page_start=int(meta.get("page_start", 1)),
                        page_end=int(meta.get("page_end", 1)),
                        table_id=meta.get("table_id") or None,
                        text=res["documents"][idx],
                        content_hash=meta.get("content_hash", ""),
                        source_uri=meta.get("source_uri") or None
                    )
                chunks.append(chunk)
        return chunks

    def count(self) -> int:
        """Return total number of chunks stored."""
        if self.use_in_memory or self._chroma_collection is None:
            return len(self._in_memory_chunks)
        return self._chroma_collection.count()

    def clear(self):
        """Empty the vector store."""
        if self.use_in_memory or self._chroma_collection is None:
            self._in_memory_chunks.clear()
            self._in_memory_embeddings.clear()
        else:
            try:
                self._chroma_client.delete_collection(self.collection_name)
                self._chroma_collection = self._chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
