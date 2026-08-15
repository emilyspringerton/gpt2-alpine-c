package vectorcache

import (
	"errors"
	"math"
	"testing"
)

func TestStubEmbedder_Deterministic(t *testing.T) {
	e := NewStubEmbedder(32)
	a, err := e.Embed("hello world")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	b, err := e.Embed("hello world")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(a) != 32 || len(b) != 32 {
		t.Fatalf("expected 32-dim vectors, got %d and %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same text produced different embeddings at index %d: %v vs %v", i, a[i], b[i])
		}
	}
}

func TestStubEmbedder_DifferentTextDifferentVector(t *testing.T) {
	e := NewStubEmbedder(32)
	a, _ := e.Embed("hello world")
	b, _ := e.Embed("goodbye world")
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different text produced identical embeddings")
	}
}

func TestCosineSimilarity_IdenticalVectorsIsOne(t *testing.T) {
	v := []float32{1, 2, 3, 4}
	sim := cosineSimilarity(v, v)
	if math.Abs(float64(sim)-1.0) > 1e-5 {
		t.Errorf("cosineSimilarity(v, v) = %v, want ~1.0", sim)
	}
}

func TestCosineSimilarity_OrthogonalVectorsIsZero(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	sim := cosineSimilarity(a, b)
	if math.Abs(float64(sim)) > 1e-5 {
		t.Errorf("cosineSimilarity(orthogonal) = %v, want ~0.0", sim)
	}
}

func TestCosineSimilarity_OppositeVectorsIsNegativeOne(t *testing.T) {
	a := []float32{1, 1}
	b := []float32{-1, -1}
	sim := cosineSimilarity(a, b)
	if math.Abs(float64(sim)+1.0) > 1e-5 {
		t.Errorf("cosineSimilarity(opposite) = %v, want ~-1.0", sim)
	}
}

func TestCosineSimilarity_ZeroVectorReturnsZeroNotNaN(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	sim := cosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("cosineSimilarity with a zero vector = %v, want 0 (not NaN)", sim)
	}
}

func TestCache_Lookup_MissOnEmptyCache(t *testing.T) {
	c := New(NewStubEmbedder(32), 0.85)
	node, hit, err := c.Lookup("anything")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if hit {
		t.Error("expected a miss on an empty cache")
	}
	if node != nil {
		t.Error("expected nil node on a miss")
	}
	if c.Stats().CacheMisses != 1 {
		t.Errorf("CacheMisses = %d, want 1", c.Stats().CacheMisses)
	}
	if c.Stats().TotalQueries != 1 {
		t.Errorf("TotalQueries = %d, want 1", c.Stats().TotalQueries)
	}
}

func TestCache_AddThenLookup_ExactTextIsHit(t *testing.T) {
	c := New(NewStubEmbedder(32), 0.85)
	_, err := c.Add("what is the capital of France", "Paris")
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	node, hit, err := c.Lookup("what is the capital of France")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if !hit {
		t.Fatal("expected a hit for the exact same query text (embeds identically)")
	}
	if node.Response != "Paris" {
		t.Errorf("Response = %q, want Paris", node.Response)
	}
	if node.AccessCount != 1 {
		t.Errorf("AccessCount = %d, want 1 after one hit", node.AccessCount)
	}
}

func TestCache_Lookup_MultipleHitsIncrementAccessCount(t *testing.T) {
	c := New(NewStubEmbedder(32), 0.85)
	_, _ = c.Add("query one", "response one")

	for i := 0; i < 3; i++ {
		_, hit, err := c.Lookup("query one")
		if err != nil {
			t.Fatalf("Lookup: %v", err)
		}
		if !hit {
			t.Fatalf("iteration %d: expected a hit", i)
		}
	}

	node, _, _ := c.Lookup("query one")
	if node.AccessCount != 4 {
		t.Errorf("AccessCount = %d, want 4 (4 total lookups)", node.AccessCount)
	}
}

func TestCache_Lookup_DissimilarQueryIsMiss(t *testing.T) {
	c := New(NewStubEmbedder(32), 0.999) // very tight threshold
	_, _ = c.Add("what is the capital of France", "Paris")

	_, hit, err := c.Lookup("completely unrelated text about spacecraft engineering")
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	if hit {
		t.Error("expected a miss for dissimilar text at a tight threshold")
	}
}

func TestCache_Stats_HitRateComputedCorrectly(t *testing.T) {
	c := New(NewStubEmbedder(32), 0.85)
	_, _ = c.Add("query one", "response one")

	_, _, _ = c.Lookup("query one")                                             // hit
	_, _, _ = c.Lookup("query one")                                             // hit
	_, _, _ = c.Lookup("something totally different and unrelated to anything") // likely miss at 0.85

	stats := c.Stats()
	if stats.TotalQueries != 3 {
		t.Errorf("TotalQueries = %d, want 3", stats.TotalQueries)
	}
	if stats.CacheHits+stats.CacheMisses != stats.TotalQueries {
		t.Errorf("hits(%d)+misses(%d) != total(%d)", stats.CacheHits, stats.CacheMisses, stats.TotalQueries)
	}
	wantRate := float64(stats.CacheHits) / float64(stats.TotalQueries) * 100
	if math.Abs(stats.HitRatePercent()-wantRate) > 1e-9 {
		t.Errorf("HitRatePercent() = %v, want %v", stats.HitRatePercent(), wantRate)
	}
}

func TestStats_HitRatePercent_ZeroQueriesNoDivideByZero(t *testing.T) {
	var s Stats
	if s.HitRatePercent() != 0 {
		t.Errorf("HitRatePercent() on zero queries = %v, want 0", s.HitRatePercent())
	}
}

func TestNode_Hash_DeterministicForSameContent(t *testing.T) {
	n1 := newNode("q", "r", []float32{1, 2, 3})
	n2 := newNode("q", "r", []float32{1, 2, 3})
	if n1.Hash != n2.Hash {
		t.Errorf("identical content produced different hashes: %s vs %s", n1.Hash, n2.Hash)
	}
}

func TestNode_Hash_DiffersOnResponseChange(t *testing.T) {
	n1 := newNode("q", "response A", []float32{1, 2, 3})
	n2 := newNode("q", "response B", []float32{1, 2, 3})
	if n1.Hash == n2.Hash {
		t.Error("different responses produced the same hash")
	}
}

func TestNode_Hash_DiffersOnEmbeddingChange(t *testing.T) {
	n1 := newNode("q", "r", []float32{1, 2, 3})
	n2 := newNode("q", "r", []float32{1, 2, 4})
	if n1.Hash == n2.Hash {
		t.Error("different embeddings produced the same hash")
	}
}

func TestNode_Hash_DiffersOnChildrenHashes(t *testing.T) {
	n1 := newNode("q", "r", []float32{1, 2, 3})
	n2 := newNode("q", "r", []float32{1, 2, 3})
	n2.ChildrenHashes = []string{"child-hash-1"}
	n2.Hash = n2.computeHash()
	if n1.Hash == n2.Hash {
		t.Error("adding a children hash did not change the node's own hash")
	}
}

func TestCache_Len(t *testing.T) {
	c := New(NewStubEmbedder(32), 0.85)
	if c.Len() != 0 {
		t.Errorf("Len() on empty cache = %d, want 0", c.Len())
	}
	_, _ = c.Add("a", "1")
	_, _ = c.Add("b", "2")
	if c.Len() != 2 {
		t.Errorf("Len() = %d, want 2", c.Len())
	}
}

type errorEmbedder struct{}

func (errorEmbedder) Embed(string) ([]float32, error) { return nil, errors.New("boom") }
func (errorEmbedder) Dim() int                        { return 8 }

func TestCache_Lookup_PropagatesEmbedderError(t *testing.T) {
	c := New(errorEmbedder{}, 0.85)
	_, _, err := c.Lookup("anything")
	if err == nil {
		t.Fatal("expected an error when the embedder fails")
	}
}

func TestCache_Add_PropagatesEmbedderError(t *testing.T) {
	c := New(errorEmbedder{}, 0.85)
	_, err := c.Add("q", "r")
	if err == nil {
		t.Fatal("expected an error when the embedder fails")
	}
}

type wrongDimEmbedder struct{}

func (wrongDimEmbedder) Embed(string) ([]float32, error) { return []float32{1, 2, 3}, nil }
func (wrongDimEmbedder) Dim() int                        { return 8 }

func TestCache_Lookup_RejectsMismatchedEmbeddingDimension(t *testing.T) {
	c := New(wrongDimEmbedder{}, 0.85)
	_, _, err := c.Lookup("anything")
	if err == nil {
		t.Fatal("expected an error when the embedder's output length doesn't match Dim()")
	}
}
