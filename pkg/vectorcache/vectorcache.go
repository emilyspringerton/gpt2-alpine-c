// Package vectorcache is a Go port of the semantic LLM-context cache
// described in docs/reference/vector_cache.md (S150-02): a flat, brute-
// force cosine-similarity index over query embeddings, with Merkle-style
// per-node content hashing for integrity/dedup, returning a cached
// response for a near-duplicate query instead of paying for a fresh
// frontier-API call.
//
// Two real uses this backs, per the backlog item: (a) semantic caching of
// frontier-API calls -- a direct, measurable way to bridge the gap while
// FABLE's own checkpoints mature -- and (b) retrieval memory for the
// personal predictive model (S148) and archetype selection.
//
// Port target, not a runtime dependency, same relationship pkg/towerprint
// has to the 2020 mag book: this package ports the reference's real
// mechanics (Merkle hashing, flat cosine search, hit/miss stats) faithfully
// -- the Merkle hashing in particular, per this item's own instruction, is
// not simplified away. What it does NOT do yet is provide a real semantic
// embedder: v0 ships only StubEmbedder, a deterministic but NOT
// semantically meaningful placeholder, so the cache mechanics can be built
// and tested now without requiring a running embedding service. Wiring a
// real embedder (e.g. a local sentence-transformers HTTP service, the same
// pattern scripts/serve.py already uses for GPT-2 inference) is real
// follow-up work, honestly deferred rather than faked here -- see
// StubEmbedder's own doc comment.
package vectorcache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"time"
)

// Embedder converts text into a semantic embedding vector. Pluggable so a
// real model can be swapped in without touching Cache's own logic.
type Embedder interface {
	Embed(text string) ([]float32, error)
	// Dim is the fixed dimensionality every Embed call returns. Cache uses
	// this to validate embeddings are comparable before ever computing a
	// similarity score against mismatched-dimension vectors.
	Dim() int
}

// Node is one cached query/response pair -- the Go shape of the
// reference's VectorNode.
type Node struct {
	Query       string
	Response    string
	Embedding   []float32
	Timestamp   time.Time
	AccessCount int
	// ChildrenHashes supports a future hierarchical tree structure (the
	// reference design's own stated direction, "organizes similar concepts
	// hierarchically") -- v0's Cache is flat (no tree construction sets
	// this), but Hash already folds it in so a later tree-building pass
	// doesn't need to change the hash's own shape, only populate this
	// field.
	ChildrenHashes []string
	// Hash is the Merkle-style content hash: query + response + embedding
	// bytes + children hashes, in that order -- ported faithfully from the
	// reference's _compute_hash, not simplified to e.g. just the query
	// text, per this item's own instruction to port the hashing faithfully.
	Hash string
}

func newNode(query, response string, embedding []float32) *Node {
	n := &Node{
		Query:     query,
		Response:  response,
		Embedding: embedding,
		Timestamp: time.Now().UTC(),
	}
	n.Hash = n.computeHash()
	return n
}

func (n *Node) computeHash() string {
	h := sha256.New()
	h.Write([]byte(n.Query))
	h.Write([]byte{'|'})
	h.Write([]byte(n.Response))
	h.Write([]byte{'|'})
	for _, f := range n.Embedding {
		binary.Write(h, binary.LittleEndian, f) //nolint:errcheck // hash.Hash.Write never errors
	}
	for _, ch := range n.ChildrenHashes {
		h.Write([]byte{'|'})
		h.Write([]byte(ch))
	}
	// Reference truncates to 16 hex chars ("[:16]" in _compute_hash) --
	// this is a cache key for dedup/display, not a security boundary, so
	// the shorter form is preserved rather than gold-plated to a full
	// sha256 the reference itself didn't use.
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// Stats mirrors the reference's get_stats() -- hit/miss counters plus the
// derived hit rate.
type Stats struct {
	TotalQueries int
	CacheHits    int
	CacheMisses  int
}

// HitRatePercent is CacheHits / TotalQueries * 100, 0 when no queries yet
// (avoids a divide-by-zero, same guard the reference's own get_stats has).
func (s Stats) HitRatePercent() float64 {
	if s.TotalQueries == 0 {
		return 0
	}
	return float64(s.CacheHits) / float64(s.TotalQueries) * 100
}

// Cache is a flat (IndexFlatIP-equivalent) semantic cache. "Flat" here
// means literally what FAISS's IndexFlatIP does: brute-force inner-product
// search over every stored vector, no approximate-nearest-neighbor
// structure. This is not a shortcut around FAISS -- IndexFlatIP itself is
// exact brute-force search; an ANN index (IVF, HNSW, ...) would be the
// performance-motivated alternative the reference doc doesn't reach for
// either. Safe for the scale this system realistically hits (thousands,
// not millions, of cached queries).
type Cache struct {
	mu                  sync.RWMutex
	embedder            Embedder
	similarityThreshold float32
	nodes               []*Node
	stats               Stats
}

// New creates a Cache. similarityThreshold is the cosine-similarity cutoff
// for a cache hit (reference default: 0.85) -- callers must choose an
// explicit value; there is no house-wide "right" threshold, it depends on
// the embedder and the acceptable false-hit rate for the caller's use case.
func New(embedder Embedder, similarityThreshold float32) *Cache {
	return &Cache{embedder: embedder, similarityThreshold: similarityThreshold}
}

// Lookup embeds query and searches for the most similar cached node. Hit
// (true) when the best match's cosine similarity meets the threshold --
// increments that node's AccessCount and Timestamp (mirrors the
// reference's increment_access(), called from inside query() on a hit) and
// bumps CacheHits; otherwise bumps CacheMisses. Every call increments
// TotalQueries, hit or miss, matching the reference's own counting (it
// increments total_queries before the hit/miss branch).
func (c *Cache) Lookup(query string) (*Node, bool, error) {
	emb, err := c.embedder.Embed(query)
	if err != nil {
		return nil, false, fmt.Errorf("vectorcache: embed query: %w", err)
	}
	if len(emb) != c.embedder.Dim() {
		return nil, false, fmt.Errorf("vectorcache: embedder returned %d dims, want %d", len(emb), c.embedder.Dim())
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.stats.TotalQueries++

	if len(c.nodes) == 0 {
		c.stats.CacheMisses++
		return nil, false, nil
	}

	var best *Node
	var bestSim float32 = -2 // cosine similarity is always in [-1, 1]
	for _, n := range c.nodes {
		sim := cosineSimilarity(emb, n.Embedding)
		if sim > bestSim {
			bestSim = sim
			best = n
		}
	}

	if bestSim >= c.similarityThreshold {
		best.AccessCount++
		best.Timestamp = time.Now().UTC()
		c.stats.CacheHits++
		return best, true, nil
	}

	c.stats.CacheMisses++
	return nil, false, nil
}

// Add embeds query and stores a new (query, response) node -- mirrors the
// reference's _add_to_cache. Callers own the miss->fetch->Add sequence
// themselves (this package deliberately doesn't own an LLM client or API
// credentials -- that's the house's existing LLMClient interface's job;
// vectorcache stays a pure caching layer).
func (c *Cache) Add(query, response string) (*Node, error) {
	emb, err := c.embedder.Embed(query)
	if err != nil {
		return nil, fmt.Errorf("vectorcache: embed query: %w", err)
	}
	if len(emb) != c.embedder.Dim() {
		return nil, fmt.Errorf("vectorcache: embedder returned %d dims, want %d", len(emb), c.embedder.Dim())
	}

	n := newNode(query, response, emb)

	c.mu.Lock()
	c.nodes = append(c.nodes, n)
	c.mu.Unlock()

	return n, nil
}

// Stats returns a snapshot of current hit/miss counters.
func (c *Cache) Stats() Stats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.stats
}

// Len is the number of stored nodes.
func (c *Cache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.nodes)
}

// cosineSimilarity assumes both vectors are already the same length
// (callers validate against Embedder.Dim() before this is reached). Not
// pre-normalized like the reference's _normalize_embedding -- computed
// directly here instead, mathematically equivalent (cosine similarity of
// normalized vectors via dot product == cosine similarity via the full
// formula on unnormalized vectors) without requiring Embedder
// implementations to remember to normalize their own output.
func cosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}
