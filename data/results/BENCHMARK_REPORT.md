# RAG Vector Database Benchmark Report

## Objective

Build high-performance RAG pipeline using PostgreSQL (pgvector) + FAISS with metadata pre-filtering for ML-ArXiv-Papers dataset. Target: metadata filtering latency <20ms, reduce search space by >=80%.

## Configuration

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Dataset**: ML-ArXiv-Papers (100,000 documents)
- **Target Latency**: Metadata filtering <20ms

## Results Summary

### Vector Search Performance (100K documents)

| Engine | Index | Latency | Recall@5 | Speedup vs pgvector-flat |
|--------|-------|---------|----------|--------------------------|
| pgvector | flat | 245.22ms | 100% | 1x |
| pgvector | ivfflat-100 | 97.44ms | 56% | 2.5x |
| pgvector | hnsw | 257.82ms | 100% | 0.95x |
| FAISS | flat | 8.16ms | 100% | 30x |
| FAISS | ivf-100 | 4.09ms | 100% | 60x |
| FAISS | hnsw | **0.38ms** | 100% | **645x** |

### Metadata Filtering Performance (100K documents)

| Filter Type | Avg Latency | P95 Latency | Results |
|-------------|-------------|-------------|---------|
| No filter (baseline) | 339.10ms | 417.84ms | 5 |
| Category filter (cs.AI) | 324.17ms | 421.17ms | 5 |
| Date range filter | 126.32ms | 216.73ms | 0 |

**Note**: Date filter returns 0 results because date metadata was not loaded into the database.

### Key Findings

1. **FAISS HNSW is fastest**: 0.38ms latency (645x faster than pgvector flat)
2. **pgvector overhead**: PostgreSQL adds ~240ms overhead for connection, query parsing, and result serialization
3. **Metadata filtering latency**: ~320ms for category filter on 100K documents (far from 20ms target)
4. **IVFFlat trade-off**: pgvector IVFFlat reduces latency but drops recall to 56%
5. **FAISS IVF**: Maintains 100% recall while providing 60x speedup

### Why pgvector is Slower

pgvector latency includes:
- SQL query parsing and planning
- Network round-trip to PostgreSQL
- Row-level locking and transaction overhead
- Result serialization (Psycopg2 cursor fetching)

FAISS runs in-process with no network overhead, making it significantly faster for vector operations.

### Achieving <20ms Metadata Filtering

To achieve the <20ms target for metadata filtering:

1. **Add B-tree index on category column**:
   ```sql
   CREATE INDEX idx_category ON documents(category);
   ```

2. **Pre-partition data by category** to reduce scan scope

3. **Use hybrid approach**: Filter first (B-tree index), then vector search on subset

4. **Denormalize**: Store category counts to estimate filter selectivity before querying

### Conclusions

1. FAISS outperforms pgvector by 30-645x for vector search operations
2. pgvector is suitable when SQL queries and metadata filtering are required
3. For <20ms filtering, additional indexing strategies are needed beyond pgvector's default configuration
4. FAISS HNSW offers best-in-class performance with 100% recall

## Files

- `benchmark_20260330_110111.json` - 100K vector search benchmark
- `metadata_filter_20260330_111649.json` - Metadata filtering benchmark
