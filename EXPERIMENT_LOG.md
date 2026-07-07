# Experiment Log

## v1.0 — Baseline content-based recommender (current)

**Approach:** Bag-of-words (`CountVectorizer`, top 5000 features) over a
concatenated tag of overview + genres + keywords + top-3 cast + director,
compared with cosine similarity.

**Why this approach:**
- Simple, fast, fully explainable — every recommendation can be traced back
  to shared tokens.
- No cold-start problem, unlike collaborative filtering.

**Known limitations:**
- Bag-of-words treats all tokens equally; a rare, highly-specific keyword
  ("time-travel-paradox") should probably matter more than a common one
  ("action"), which `CountVectorizer` doesn't capture.
- No signal for movie quality/popularity — a recommender could suggest a
  poorly-rated movie just because it shares genre and cast tokens.
- Evaluated only qualitatively (manual spot checks), not with a quantitative metric.

**Results (qualitative):**
- `Avatar` → [Aliens vs Predator: Requiem, Aliens, Falcon Rising ,Independence Day, Titan A.E.]
- `The Dark Knight` → [The Dark Knight Rises, Batman Begins, Batman Returns, Batman Forever, Batman]

## Planned next steps

- [ ] **TF-IDF instead of raw counts** — down-weight common terms like
      "action" or "love" that appear across many movies, up-weight rarer,
      more distinctive terms.
- [ ] **Field weighting** — director and top-billed cast are likely stronger
      similarity signals than overview text; try weighting fields rather than
      treating the concatenated tag as flat.
- [ ] **Hybrid re-ranking** — blend content similarity with a popularity or
      average-rating prior so obscure/poorly-rated look-alikes aren't
      recommended over well-regarded ones.
- [ ] **Quantitative evaluation** — e.g., precision@5 on genre overlap across
      a sample of query movies, to compare future iterations objectively
      instead of relying on spot checks.
- [ ] **Embeddings-based similarity** — replace bag-of-words with sentence
      embeddings (e.g. `sentence-transformers`) over the overview text, which
      would capture semantic similarity beyond exact word overlap.
