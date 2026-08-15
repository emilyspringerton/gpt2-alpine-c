package vectorcache

import (
	"crypto/sha256"
	"math"
)

// StubEmbedder is a deterministic but NOT semantically meaningful
// placeholder Embedder. It exists so Cache's real mechanics (Merkle
// hashing, flat cosine search, hit/miss stats) can be built and tested
// today without a running embedding service -- it does NOT provide real
// semantic similarity: two texts with completely different meaning but
// similar byte content can score as "similar," and two paraphrases of the
// same idea will not. Wiring a real embedder (e.g. a local
// sentence-transformers HTTP service, the same pattern scripts/serve.py
// already uses for GPT-2 inference) is real follow-up work -- do not use
// StubEmbedder for an actual semantic-cache deployment; it is a test/demo
// fixture, not a "good enough for now" embedder.
type StubEmbedder struct {
	dim int
}

// NewStubEmbedder creates a StubEmbedder producing dim-dimensional
// vectors. dim must be a positive multiple of 8 (each sha256 round
// contributes 8 float32 slots); a small dim like 32 is plenty for tests.
func NewStubEmbedder(dim int) *StubEmbedder {
	if dim <= 0 {
		dim = 32
	}
	return &StubEmbedder{dim: dim}
}

func (e *StubEmbedder) Dim() int { return e.dim }

// Embed derives a fixed-length pseudo-vector from text's own sha256 bytes,
// re-hashed as many times as needed to fill Dim() slots, each byte pair
// mapped into [-1, 1]. Deterministic (same text -> same vector, always)
// and cheap, which is all a mechanics-testing fixture needs -- it is
// explicitly not a claim about semantic content, see the type's own doc
// comment.
func (e *StubEmbedder) Embed(text string) ([]float32, error) {
	out := make([]float32, 0, e.dim)
	block := sha256.Sum256([]byte(text))
	for len(out) < e.dim {
		for i := 0; i+1 < len(block) && len(out) < e.dim; i += 2 {
			v := int16(uint16(block[i])<<8 | uint16(block[i+1]))
			out = append(out, float32(v)/float32(math.MaxInt16))
		}
		next := sha256.Sum256(block[:])
		block = next
	}
	return out, nil
}
