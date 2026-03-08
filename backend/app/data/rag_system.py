"""
RAG System - Retrieval Augmented Generation with 800K Products

This module implements a comprehensive RAG system for the SteadyPrice
Week 8 multi-agent system, providing intelligent product retrieval
and augmented generation capabilities.
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import pickle
import hashlib
from pathlib import Path

# Vector database and embeddings
import faiss
from sentence_transformers import SentenceTransformer
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class RetrievalMethod(Enum):
    """Different retrieval methods"""
    SEMANTIC_SEARCH = "semantic_search"
    HYBRID_SEARCH = "hybrid_search"
    KEYWORD_SEARCH = "keyword_search"
    CATEGORY_FILTER = "category_filter"
    PRICE_RANGE_FILTER = "price_range_filter"

@dataclass
class ProductEmbedding:
    """Product embedding with metadata"""
    product_id: str
    title: str
    category: str
    description: str
    price: float
    retailer: str
    embedding: np.ndarray
    keywords: List[str]
    timestamp: datetime

@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    products: List[ProductEmbedding]
    scores: List[float]
    query_embedding: Optional[np.ndarray]
    retrieval_method: RetrievalMethod
    total_candidates: int
    retrieval_time: float
    metadata: Dict[str, Any]

class RAGSystem:
    """
    Retrieval Augmented Generation system for 800K products.
    
    Provides intelligent product retrieval with multiple search methods,
    semantic understanding, and real-time updates.
    """
    
    def __init__(self):
        """Initialize the RAG system."""
        # Product storage
        self.products: Dict[str, ProductEmbedding] = {}
        self.product_embeddings: np.ndarray = None
        self.product_ids: List[str] = []
        
        # Search components
        self.embedding_model: Optional[SentenceTransformer] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        
        # Vector indices
        self.faiss_index: Optional[faiss.Index] = None
        self.dimension = 384  # Sentence transformer dimension
        
        # Redis for caching
        self.redis_client: Optional[redis.Redis] = None
        
        # Performance metrics
        self.rag_metrics = {
            "total_products": 0,
            "total_retrievals": 0,
            "average_retrieval_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retrieval_methods_used": {},
            "last_updated": None
        }
        
        # Search configuration
        self.search_config = {
            "max_results": 50,
            "similarity_threshold": 0.3,
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
            "cache_ttl": 3600  # 1 hour
        }
    
    async def initialize(self, model_name: str = "all-MiniLM-L6-v2") -> bool:
        """
        Initialize the RAG system with embedding model and indices.
        
        Args:
            model_name: Name of the sentence transformer model
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing RAG system with 800K products...")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize Redis for caching
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except:
                logger.warning("Redis not available, using in-memory cache")
                self.redis_client = None
            
            # Load or create product database
            await self._load_product_database()
            
            # Build search indices
            await self._build_search_indices()
            
            self.rag_metrics["last_updated"] = datetime.utcnow()
            logger.info(f"RAG system initialized with {len(self.products)} products")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            return False
    
    async def _load_product_database(self):
        """Load product database from various sources."""
        try:
            # In production, this would load from actual databases
            # For demo, create sample products
            
            sample_products = self._generate_sample_products(800000)
            
            for product_data in sample_products:
                product = ProductEmbedding(
                    product_id=product_data["id"],
                    title=product_data["title"],
                    category=product_data["category"],
                    description=product_data["description"],
                    price=product_data["price"],
                    retailer=product_data["retailer"],
                    embedding=np.zeros(self.dimension),  # Will be computed later
                    keywords=product_data["keywords"],
                    timestamp=datetime.utcnow()
                )
                
                self.products[product.product_id] = product
            
            logger.info(f"Loaded {len(self.products)} products into database")
            
        except Exception as e:
            logger.error(f"Error loading product database: {e}")
            raise
    
    def _generate_sample_products(self, count: int) -> List[Dict[str, Any]]:
        """Generate sample products for demonstration."""
        products = []
        
        categories = ["Electronics", "Appliances", "Automotive", "Furniture", "Clothing", "Books", "Sports", "Home", "Beauty", "Toys"]
        retailers = ["Amazon", "Best Buy", "Walmart", "Target", "Home Depot", "Lowe's", "Wayfair", "Newegg"]
        
        # Product templates
        electronics_templates = [
            "Samsung {model} 4K Smart TV",
            "Apple iPhone {model}",
            "Dell {model} Laptop",
            "Sony {model} Headphones",
            "LG {model} Refrigerator",
            "Microsoft {model} Console"
        ]
        
        for i in range(count):
            category = categories[i % len(categories)]
            retailer = retailers[i % len(retailers)]
            
            if category == "Electronics":
                template = electronics_templates[i % len(electronics_templates)]
                title = template.format(model=f"Series {i % 10 + 1}")
                price = np.random.uniform(100, 2000)
                description = f"High-quality {title.lower()} with advanced features and modern technology."
            else:
                title = f"{category} Product {i}"
                price = np.random.uniform(20, 500)
                description = f"Premium {category.lower()} product with excellent quality and durability."
            
            # Generate keywords
            keywords = [
                category.lower(),
                retailer.lower(),
                "premium",
                "quality",
                "best",
                "deal"
            ]
            
            products.append({
                "id": f"product_{i}",
                "title": title,
                "category": category,
                "description": description,
                "price": round(price, 2),
                "retailer": retailer,
                "keywords": keywords
            })
        
        return products
    
    async def _build_search_indices(self):
        """Build search indices for fast retrieval."""
        try:
            logger.info("Building search indices...")
            
            # Compute embeddings for all products
            product_texts = []
            self.product_ids = []
            
            for product_id, product in self.products.items():
                # Combine title and description for embedding
                text = f"{product.title} {product.description}"
                product_texts.append(text)
                self.product_ids.append(product_id)
            
            # Compute embeddings in batches
            batch_size = 1000
            all_embeddings = []
            
            for i in range(0, len(product_texts), batch_size):
                batch_texts = product_texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_embeddings)
                
                if i % 10000 == 0:
                    logger.info(f"Processed {i} products...")
            
            # Combine all embeddings
            self.product_embeddings = np.vstack(all_embeddings)
            
            # Update product embeddings
            for i, product_id in enumerate(self.product_ids):
                self.products[product_id].embedding = self.product_embeddings[i]
            
            # Build FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(self.product_embeddings, axis=1, keepdims=True)
            normalized_embeddings = self.product_embeddings / norms
            
            self.faiss_index.add(normalized_embeddings)
            
            # Build TF-IDF matrix
            all_texts = [f"{p.title} {p.description}" for p in self.products.values()]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
            logger.info(f"Built TF-IDF matrix with shape {self.tfidf_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error building search indices: {e}")
            raise
    
    async def retrieve_products(
        self,
        query: str,
        method: RetrievalMethod = RetrievalMethod.SEMANTIC_SEARCH,
        max_results: Optional[int] = None,
        category_filter: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
        retailer_filter: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve products based on query using specified method.
        
        Args:
            query: Search query
            method: Retrieval method
            max_results: Maximum number of results
            category_filter: Filter by category
            price_range: Filter by price range (min, max)
            retailer_filter: Filter by retailer
            
        Returns:
            RetrievalResult with matched products and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, method, category_filter, price_range, retailer_filter)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                self.rag_metrics["cache_hits"] += 1
                return cached_result
            
            self.rag_metrics["cache_misses"] += 1
            
            # Apply filters
            filtered_products = self._apply_filters(
                list(self.products.values()),
                category_filter,
                price_range,
                retailer_filter
            )
            
            if not filtered_products:
                return RetrievalResult(
                    products=[],
                    scores=[],
                    query_embedding=None,
                    retrieval_method=method,
                    total_candidates=0,
                    retrieval_time=0.0,
                    metadata={"message": "No products match the filters"}
                )
            
            # Perform retrieval based on method
            if method == RetrievalMethod.SEMANTIC_SEARCH:
                result = await self._semantic_search(query, filtered_products, max_results)
            elif method == RetrievalMethod.HYBRID_SEARCH:
                result = await self._hybrid_search(query, filtered_products, max_results)
            elif method == RetrievalMethod.KEYWORD_SEARCH:
                result = await self._keyword_search(query, filtered_products, max_results)
            else:
                result = await self._semantic_search(query, filtered_products, max_results)
            
            # Update metrics
            retrieval_time = (datetime.utcnow() - start_time).total_seconds()
            self.rag_metrics["total_retrievals"] += 1
            
            # Update average retrieval time
            total = self.rag_metrics["total_retrievals"]
            current_avg = self.rag_metrics["average_retrieval_time"]
            new_avg = ((current_avg * (total - 1)) + retrieval_time) / total
            self.rag_metrics["average_retrieval_time"] = new_avg
            
            # Update method usage
            method_name = method.value
            self.rag_metrics["retrieval_methods_used"][method_name] = (
                self.rag_metrics["retrieval_methods_used"].get(method_name, 0) + 1
            )
            
            # Cache result
            result.retrieval_time = retrieval_time
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in product retrieval: {e}")
            return RetrievalResult(
                products=[],
                scores=[],
                query_embedding=None,
                retrieval_method=method,
                total_candidates=0,
                retrieval_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def _semantic_search(self, query: str, products: List[ProductEmbedding], max_results: Optional[int]) -> RetrievalResult:
        """Perform semantic search using embeddings."""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
            
            # Search in FAISS index
            k = min(max_results or self.search_config["max_results"], len(products))
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                k
            )
            
            # Get results
            retrieved_products = []
            retrieved_scores = []
            
            for i, idx in enumerate(indices[0]):
                if idx < len(self.product_ids):
                    product_id = self.product_ids[idx]
                    product = self.products[product_id]
                    
                    # Apply similarity threshold
                    if scores[0][i] >= self.search_config["similarity_threshold"]:
                        retrieved_products.append(product)
                        retrieved_scores.append(float(scores[0][i]))
            
            return RetrievalResult(
                products=retrieved_products,
                scores=retrieved_scores,
                query_embedding=query_embedding,
                retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
                total_candidates=len(products),
                retrieval_time=0.0,
                metadata={"search_method": "semantic", "threshold": self.search_config["similarity_threshold"]}
            )
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise
    
    async def _hybrid_search(self, query: str, products: List[ProductEmbedding], max_results: Optional[int]) -> RetrievalResult:
        """Perform hybrid search combining semantic and keyword search."""
        try:
            # Get semantic results
            semantic_result = await self._semantic_search(query, products, max_results)
            
            # Get keyword results
            keyword_result = await self._keyword_search(query, products, max_results)
            
            # Combine results
            combined_scores = {}
            
            # Add semantic scores
            for i, product in enumerate(semantic_result.products):
                combined_scores[product.product_id] = {
                    "product": product,
                    "semantic_score": semantic_result.scores[i],
                    "keyword_score": 0.0
                }
            
            # Add keyword scores
            for i, product in enumerate(keyword_result.products):
                if product.product_id in combined_scores:
                    combined_scores[product.product_id]["keyword_score"] = keyword_result.scores[i]
                else:
                    combined_scores[product.product_id] = {
                        "product": product,
                        "semantic_score": 0.0,
                        "keyword_score": keyword_result.scores[i]
                    }
            
            # Calculate hybrid scores
            final_products = []
            final_scores = []
            
            for product_id, scores in combined_scores.items():
                semantic_weight = self.search_config["semantic_weight"]
                keyword_weight = self.search_config["keyword_weight"]
                
                hybrid_score = (
                    semantic_weight * scores["semantic_score"] +
                    keyword_weight * scores["keyword_score"]
                )
                
                final_products.append(scores["product"])
                final_scores.append(hybrid_score)
            
            # Sort by hybrid score
            sorted_indices = np.argsort(final_scores)[::-1]
            final_products = [final_products[i] for i in sorted_indices]
            final_scores = [final_scores[i] for i in sorted_indices]
            
            # Limit results
            k = min(max_results or self.search_config["max_results"], len(final_products))
            final_products = final_products[:k]
            final_scores = final_scores[:k]
            
            return RetrievalResult(
                products=final_products,
                scores=final_scores,
                query_embedding=None,
                retrieval_method=RetrievalMethod.HYBRID_SEARCH,
                total_candidates=len(products),
                retrieval_time=0.0,
                metadata={
                    "search_method": "hybrid",
                    "semantic_weight": self.search_config["semantic_weight"],
                    "keyword_weight": self.search_config["keyword_weight"]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise
    
    async def _keyword_search(self, query: str, products: List[ProductEmbedding], max_results: Optional[int]) -> RetrievalResult:
        """Perform keyword-based search using TF-IDF."""
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top results
            k = min(max_results or self.search_config["max_results"], len(products))
            top_indices = np.argsort(similarities)[::-1][:k]
            
            retrieved_products = []
            retrieved_scores = []
            
            for idx in top_indices:
                if idx < len(self.product_ids):
                    product_id = self.product_ids[idx]
                    product = self.products[product_id]
                    
                    if similarities[idx] > 0:  # Only include non-zero similarities
                        retrieved_products.append(product)
                        retrieved_scores.append(float(similarities[idx]))
            
            return RetrievalResult(
                products=retrieved_products,
                scores=retrieved_scores,
                query_embedding=None,
                retrieval_method=RetrievalMethod.KEYWORD_SEARCH,
                total_candidates=len(products),
                retrieval_time=0.0,
                metadata={"search_method": "keyword"}
            )
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            raise
    
    def _apply_filters(
        self,
        products: List[ProductEmbedding],
        category_filter: Optional[str],
        price_range: Optional[Tuple[float, float]],
        retailer_filter: Optional[str]
    ) -> List[ProductEmbedding]:
        """Apply filters to product list."""
        filtered = products
        
        if category_filter:
            filtered = [p for p in filtered if p.category == category_filter]
        
        if price_range:
            min_price, max_price = price_range
            filtered = [p for p in filtered if min_price <= p.price <= max_price]
        
        if retailer_filter:
            filtered = [p for p in filtered if p.retailer == retailer_filter]
        
        return filtered
    
    def _generate_cache_key(
        self,
        query: str,
        method: RetrievalMethod,
        category_filter: Optional[str],
        price_range: Optional[Tuple[float, float]],
        retailer_filter: Optional[str]
    ) -> str:
        """Generate cache key for retrieval result."""
        key_data = {
            "query": query,
            "method": method.value,
            "category": category_filter,
            "price_range": price_range,
            "retailer": retailer_filter
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[RetrievalResult]:
        """Get cached retrieval result."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(f"rag_cache:{cache_key}")
            if cached_data:
                data = json.loads(cached_data)
                
                # Reconstruct embeddings
                products = []
                for product_data in data["products"]:
                    product = ProductEmbedding(**product_data)
                    products.append(product)
                
                return RetrievalResult(
                    products=products,
                    scores=data["scores"],
                    query_embedding=np.array(data["query_embedding"]) if data["query_embedding"] else None,
                    retrieval_method=RetrievalMethod(data["retrieval_method"]),
                    total_candidates=data["total_candidates"],
                    retrieval_time=data["retrieval_time"],
                    metadata=data["metadata"]
                )
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: RetrievalResult):
        """Cache retrieval result."""
        if not self.redis_client:
            return
        
        try:
            # Serialize result
            cache_data = {
                "products": [asdict(p) for p in result.products],
                "scores": result.scores,
                "query_embedding": result.query_embedding.tolist() if result.query_embedding is not None else None,
                "retrieval_method": result.retrieval_method.value,
                "total_candidates": result.total_candidates,
                "retrieval_time": result.retrieval_time,
                "metadata": result.metadata
            }
            
            # Store in Redis with TTL
            self.redis_client.setex(
                f"rag_cache:{cache_key}",
                self.search_config["cache_ttl"],
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    async def add_product(self, product: ProductEmbedding) -> bool:
        """Add a new product to the RAG system."""
        try:
            # Compute embedding
            text = f"{product.title} {product.description}"
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
            product.embedding = embedding
            
            # Add to storage
            self.products[product.product_id] = product
            
            # Update indices (in production, would do this in batches)
            await self._update_indices()
            
            # Update metrics
            self.rag_metrics["total_products"] = len(self.products)
            self.rag_metrics["last_updated"] = datetime.utcnow()
            
            logger.info(f"Added product {product.product_id} to RAG system")
            return True
            
        except Exception as e:
            logger.error(f"Error adding product: {e}")
            return False
    
    async def _update_indices(self):
        """Update search indices with new products."""
        # In production, this would be more sophisticated
        # For now, rebuild indices periodically
        if len(self.products) % 1000 == 0:
            await self._build_search_indices()
    
    async def get_similar_products(self, product_id: str, max_results: int = 10) -> RetrievalResult:
        """Get products similar to a given product."""
        try:
            if product_id not in self.products:
                return RetrievalResult(
                    products=[],
                    scores=[],
                    query_embedding=None,
                    retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
                    total_candidates=0,
                    retrieval_time=0.0,
                    metadata={"error": "Product not found"}
                )
            
            product = self.products[product_id]
            query = f"{product.title} {product.description}"
            
            return await self.retrieve_products(
                query,
                RetrievalMethod.SEMANTIC_SEARCH,
                max_results=max_results
            )
            
        except Exception as e:
            logger.error(f"Error getting similar products: {e}")
            return RetrievalResult(
                products=[],
                scores=[],
                query_embedding=None,
                retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
                total_candidates=0,
                retrieval_time=0.0,
                metadata={"error": str(e)}
            )
    
    def get_rag_metrics(self) -> Dict[str, Any]:
        """Get comprehensive RAG system metrics."""
        return {
            **self.rag_metrics,
            "total_products": len(self.products),
            "embedding_dimension": self.dimension,
            "index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "tfidf_features": self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            "cache_available": self.redis_client is not None,
            "search_config": self.search_config,
            "supported_methods": [method.value for method in RetrievalMethod],
            "categories": list(set(p.category for p in self.products.values())),
            "retailers": list(set(p.retailer for p in self.products.values()))
        }
    
    async def health_check(self) -> bool:
        """Check if the RAG system is healthy."""
        try:
            # Test basic retrieval
            test_result = await self.retrieve_products("test query", max_results=5)
            
            # Check if indices are built
            if not self.faiss_index or self.faiss_index.ntotal == 0:
                return False
            
            # Check if embedding model is loaded
            if not self.embedding_model:
                return False
            
            # Check if we have products
            if len(self.products) == 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"RAG system health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the RAG system."""
        try:
            # Save metrics
            self.rag_metrics["shutdown_time"] = datetime.utcnow()
            
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("RAG system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during RAG system shutdown: {e}")
