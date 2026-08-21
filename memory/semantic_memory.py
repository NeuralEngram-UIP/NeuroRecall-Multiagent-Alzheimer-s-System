# semantic_memory.py

"""
FINAL ENTERPRISE-GRADE
SEMANTIC MEMORY STORE

Qdrant-backed semantic cognition layer
for autonomous multi-agent systems.

Features
────────
- persistent vector storage
- semantic retrieval
- reinforcement-aware ranking
- metadata filtering
- observability metrics
- thread-safe operations
- structured logging
- embedding validation
- immutable memory objects
- production-safe exception handling
- global memory IDs
- cognitive retrieval scoring
"""

import logging
import math
import threading
import time
import tempfile

from qdrant_client.local.qdrant_local import QdrantLocal
from dataclasses import dataclass

from typing import (
    Any,
    Dict,
    List,
    Optional
)

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams
)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

VECTOR_SIZE = 384

COLLECTION_NAME = "semantic_memory"

DEFAULT_TOP_K = 5


# retrieval weighting

SIMILARITY_WEIGHT = 0.80

REINFORCEMENT_WEIGHT = 0.20


# ─────────────────────────────────────────────────────────────
# Semantic Memory Object
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SemanticMemory:
    """
    Immutable semantic memory object.
    """

    memory_id: str

    content: str

    metadata: Dict[str, Any]

    score: Optional[float] = None

    embedding: Optional[
        List[float]
    ] = None

    reinforcement_count: int = 0


# ─────────────────────────────────────────────────────────────
# Semantic Memory Store
# ─────────────────────────────────────────────────────────────

class SemanticMemoryStore:
    """
    Enterprise-grade semantic memory.

    Responsibilities:
    - vector persistence
    - semantic retrieval
    - reinforcement-aware ranking
    - metadata filtering
    - observability
    """

    def __init__(
        self,
        collection_name: str = (
            COLLECTION_NAME
        ),
        vector_size: int = (
            VECTOR_SIZE
        ),
        storage_path: str = None
       
    ):

        self.collection_name = (
            collection_name
        )

        self.vector_size = (
            vector_size
        )

        # IMPORTANT:
        # RLock is intentionally used
        # because internal helper methods
        # may be called from already-
        # locked contexts.
        #
        # Replacing this with Lock
        # can introduce deadlocks.

        self.lock = threading.RLock()

        # observability metrics

        self.total_searches = 0

        self.total_inserts = 0

        self.total_deletes = 0

        self.total_reinforcements = 0

        self.total_failures = 0

        self.total_latency = 0.0

        # IMPORTANT (Alzheimer's deployment fix):
        # The original code used location=":memory:" here, which is a
        # non-persistent, in-process-only Qdrant instance -- every
        # restart silently wiped all semantic memory. For a system
        # whose entire purpose is long-term retention for a cognitively
        # vulnerable patient, that is not an acceptable failure mode.
        # storage_path (constructor param, previously accepted but
        # unused) now actually controls where data is written on disk.
        self.storage_path = storage_path or "qdrant_data_new"

        try:
            self.client = QdrantClient(
                path=self.storage_path
            )
            self._create_collection()

        except Exception:
            logger.exception(
                "Failed initializing "
                "Qdrant client "
                "path=%s",
                self.storage_path
            )
            raise
     

        logger.info(
            "SemanticMemoryStore initialized "
            "collection=%s",
            self.collection_name
        )

    # ─────────────────────────────────────────────────────────
    # Collection Setup
    # ─────────────────────────────────────────────────────────

    def _create_collection(
        self
    ):
        """
        Create Qdrant collection
        if missing.
        """

        try:

            with self.lock:

                collections = (
                    self.client
                    .get_collections()
                )

                existing = {
                    collection.name
                    for collection
                    in collections.collections
                }

                if (
                    self.collection_name
                    in existing
                ):

                    logger.info(
                        "Collection exists "
                        "collection=%s",
                        self.collection_name
                    )

                    return

                self.client.create_collection(
                    collection_name=(
                        self.collection_name
                    ),

                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )

                logger.info(
                    "Created collection "
                    "collection=%s",
                    self.collection_name
                )

        except Exception:

            with self.lock:

                self.total_failures += 1

            logger.exception(
                "Collection creation failed "
                "collection=%s",
                self.collection_name
            )

            raise

    # ─────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────

    def _validate_embedding(
        self,
        embedding: List[float]
    ):
        """
        Validate embedding integrity.
        """

        if not embedding:

            raise ValueError(
                "embedding cannot be empty"
            )

        if len(embedding) != (
            self.vector_size
        ):

            raise ValueError(
                f"embedding size must "
                f"be {self.vector_size}"
            )

        for value in embedding:

            if not isinstance(
                value,
                (float, int)
            ):

                raise ValueError(
                    "embedding values "
                    "must be numeric"
                )

    # ─────────────────────────────────────────────────────────
    # Reinforcement Scoring
    # ─────────────────────────────────────────────────────────

    def _reinforcement_score(
        self,
        reinforcement_count: int
    ) -> float:
        """
        Logarithmic reinforcement scaling.

        Prevents runaway dominance while
        still rewarding repeated recall.
        """

        return min(
            math.log1p(
                reinforcement_count
            ) / 3.0,
            1.0
        )

    # ─────────────────────────────────────────────────────────
    # Store
    # ─────────────────────────────────────────────────────────

    def add(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[
            Dict[str, Any]
        ] = None
    ) -> str:
        """
        Store semantic memory vector.
        """

        if not content.strip():

            raise ValueError(
                "content cannot be empty"
            )

        self._validate_embedding(
            embedding
        )

        metadata = metadata or {}

        payload = {

            "memory_id": memory_id,

            "agent_id": metadata.get(
                "agent_id"
            ),

            "role": metadata.get(
                "role"
            ),

            "shared": metadata.get(
                "shared",
                False
            ),

            "content": content,

            "metadata": metadata,

            "reinforcement_count": 0
        }

        try:

            with self.lock:

                self.client.upsert(
                    collection_name=(
                        self.collection_name
                    ),

                    points=[
                        PointStruct(
                            id=memory_id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )

                self.total_inserts += 1

            logger.info(
                "stored semantic memory "
                "memory_id=%s",
                memory_id
            )

            return memory_id

        except Exception:

            with self.lock:

                self.total_failures += 1

            logger.exception(
                "Failed storing semantic memory "
                "memory_id=%s",
                memory_id
            )

            raise

    # ─────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────

    def search(
        self,
        embedding: List[float],
        limit: int = DEFAULT_TOP_K,
        metadata_filter: Optional[
            Dict[str, Any]
        ] = None,
        with_vectors: bool = False
    ) -> List[SemanticMemory]:
        """
        Semantic similarity retrieval.
        """

        self._validate_embedding(
            embedding
        )

        start_time = time.time()

        query_filter = None

        if metadata_filter:

            conditions = []

            for key, value in (
                metadata_filter.items()
            ):

                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(
                            value=value
                        )
                    )
                )

            query_filter = Filter(
                must=conditions
            )

        try:

            with self.lock:

                results = (
                    self.client
                    .query_points(
                        collection_name=(
                            self.collection_name
                        ),

                        query=embedding,

                        limit=limit,

                        query_filter=(
                            query_filter
                        ),

                        with_vectors=(
                            with_vectors
                        )
                    )
                    .points
                )

                self.total_searches += 1

            memories = []

            for result in results:

                payload = (
                    result.payload
                    or {}
                )

                reinforcement_count = (
                    payload.get(
                        "reinforcement_count",
                        0
                    )
                )

                similarity_score = (
                    float(result.score)
                )

                reinforcement_score = (
                    self._reinforcement_score(
                        reinforcement_count
                    )
                )

                final_score = (
                    similarity_score
                    * SIMILARITY_WEIGHT
                    +
                    reinforcement_score
                    * REINFORCEMENT_WEIGHT
                )

                memories.append(
                    SemanticMemory(
                        memory_id=str(
                            result.id
                        ),

                        content=payload.get(
                            "content",
                            ""
                        ),

                        metadata=payload.get(
                            "metadata",
                            {}
                        ),

                        score=final_score,

                        embedding=(
                            result.vector
                            if with_vectors
                            else None
                        ),

                        reinforcement_count=(
                            reinforcement_count
                        )
                    )
                )

            memories.sort(
                key=lambda x: (
                    x.score or 0.0
                ),
                reverse=True
            )

            latency = (
                time.time()
                - start_time
            )

            with self.lock:

                self.total_latency += latency

            logger.info(
                "semantic search complete "
                "results=%s "
                "latency=%.4fs",
                len(memories),
                latency
            )

            return memories

        except Exception:

            with self.lock:

                self.total_failures += 1

            logger.exception(
                "Semantic search failed"
            )

            raise

    # ─────────────────────────────────────────────────────────
    # Multi-Agent Search
    # ─────────────────────────────────────────────────────────

    def search_multi(
        self,
        embedding: List[float],
        limit: int = DEFAULT_TOP_K,
        agent_id: Optional[str] = None
    ) -> List[SemanticMemory]:
        """
        Multi-agent semantic retrieval.
        """

        metadata_filter = None

        if agent_id is not None:

            metadata_filter = {
                "agent_id": agent_id
            }

        return self.search(
            embedding=embedding,
            limit=limit,
            metadata_filter=metadata_filter
        )

    # ─────────────────────────────────────────────────────────
    # Get
    # ─────────────────────────────────────────────────────────

    def get(
        self,
        memory_id: str,
        with_vectors: bool = False
    ) -> Optional[
        SemanticMemory
    ]:
        """
        Retrieve semantic memory by ID.
        """

        try:

            with self.lock:

                results = (
                    self.client.retrieve(
                        collection_name=(
                            self.collection_name
                        ),

                        ids=[memory_id],

                        with_vectors=(
                            with_vectors
                        )
                    )
                )

            if not results:
                return None

            result = results[0]

            payload = (
                result.payload
                or {}
            )

            return SemanticMemory(
                memory_id=str(
                    result.id
                ),

                content=payload.get(
                    "content",
                    ""
                ),

                metadata=payload.get(
                    "metadata",
                    {}
                ),

                embedding=(
                    result.vector
                    if with_vectors
                    else None
                ),

                reinforcement_count=(
                    payload.get(
                        "reinforcement_count",
                        0
                    )
                )
            )

        except Exception:

            with self.lock:

                self.total_failures += 1

            logger.exception(
                "Failed retrieving memory "
                "memory_id=%s",
                memory_id
            )

            raise

    # ─────────────────────────────────────────────────────────
    # Reinforcement
    # ─────────────────────────────────────────────────────────

    def reinforce(
        self,
        memory_id: str
    ):
        """
        Reinforce semantic memory.
        """

        try:

            with self.lock:

                memory = self.get(
                    memory_id
                )

                if memory is None:

                    logger.warning(
                        "reinforcement target "
                        "not found memory_id=%s",
                        memory_id
                    )

                    return

                updated_count = (
                    memory.reinforcement_count
                    + 1
                )

                payload = {
                    "reinforcement_count": (
                        updated_count
                    )
                }

                self.client.set_payload(
                    collection_name=(
                        self.collection_name
                    ),

                    payload=payload,

                    points=[memory_id]
                )

                self.total_reinforcements += 1

            logger.info(
                "reinforced semantic memory "
                "memory_id=%s "
                "count=%s",
                memory_id,
                updated_count
            )

        except Exception:

            with self.lock:

                self.total_failures += 1

            logger.exception(
                "Semantic reinforcement failed "
                "memory_id=%s",
                memory_id
            )

            raise

    # ─────────────────────────────────────────────────────────
    # Delete
    # ─────────────────────────────────────────────────────────

    def delete(
        self,
        memory_id: str
    ):
        """
        Delete semantic memory.
        """

        try:

            with self.lock:

                self.client.delete(
                    collection_name=(
                        self.collection_name
                    ),

                    points_selector=(
                        PointIdsList(
                            points=[
                                memory_id
                            ]
                        )
                    )
                )

                self.total_deletes += 1

            logger.info(
                "deleted semantic memory "
                "memory_id=%s",
                memory_id
            )

        except Exception:

            with self.lock:

                self.total_failures += 1

            logger.exception(
                "Failed deleting memory "
                "memory_id=%s",
                memory_id
            )

            raise

    # ─────────────────────────────────────────────────────────
    # Unsafe Count
    # ─────────────────────────────────────────────────────────

    def _count_unsafe(
        self
    ) -> int:
        """
        Lock-free count.

        MUST only be called from
        contexts that already hold
        self.lock.
        """

        result = (
            self.client.count(
                collection_name=(
                    self.collection_name
                )
            )
        )

        return result.count

    # ─────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────

    def metrics(
        self
    ) -> Dict[str, Any]:
        """
        Observability metrics.

        NOTE:
        count() performs a live
        Qdrant query.
        """

        with self.lock:

            avg_latency = 0.0

            if self.total_searches > 0:

                avg_latency = (
                    self.total_latency
                    / self.total_searches
                )

            return {

                "total_memories": (
                    self._count_unsafe()
                ),

                "total_searches": (
                    self.total_searches
                ),

                "total_inserts": (
                    self.total_inserts
                ),

                "total_reinforcements": (
                    self.total_reinforcements
                ),

                "total_deletes": (
                    self.total_deletes
                ),

                "total_failures": (
                    self.total_failures
                ),

                "average_search_latency": (
                    avg_latency
                )
            }

    # ─────────────────────────────────────────────────────────
    # Count
    # ─────────────────────────────────────────────────────────

    def count(
        self
    ) -> int:
        """
        Total semantic memories.
        """

        try:

            with self.lock:

                return (
                    self._count_unsafe()
                )

        except Exception:

            with self.lock:

                self.total_failures += 1

            logger.exception(
                "Failed counting memories"
            )

            raise