# Metrics

Definitions live in `rag_pipeline/metrics.py` — this is the plain-English version.
All worked numbers below are computed from that module.

---

## The relevance judgement

Every retrieval and reranking metric is built on one yes/no decision per retrieved
passage: **does this passage contain a gold answer?** (`has_answer`)

- Whitespace-normalised and case-insensitive.
- **Token-bounded**: `"art"` does *not* match inside `"Bart"`, `"1998"` does *not*
  match inside `"11998"`. A plain substring test inflates every metric below.
- Boundaries are only applied where the answer starts/ends alphanumeric, so `"$5"`
  and `"(1998)"` still match.

This is the DPR criterion for NQ. It is a proxy: a passage that answers the question
in different words scores as irrelevant, and a passage that mentions the answer string
by coincidence scores as relevant.

A query's retrieved list becomes a **relevance list** — ordered, one bool per passage,
index 0 = rank 1.

---

## Running example

Query `"when was google founded"`, gold answer `"September 4, 1998"`.
10 passages retrieved; relevant ones at ranks 2 and 5. Metric cutoff `k = 3`.

```
rank      1  2  3 │ 4  5  6  7  8  9 10
relevant  .  T  . │ .  T  .  .  .  .  .
                  └── k = 3
```

---

## Retrieval — `stages.retrieval`

| key | plain English | example |
|---|---|---|
| `recall_at_k` | Did **any** of the top-k contain the answer? Averaged over queries. | `1.0` — rank 2 is relevant |
| `precision_at_k` | What fraction of the top-k contained the answer? | `0.333` — 1 of 3 |
| `mrr` | 1 ÷ rank of the *first* relevant passage in the top-k. | `0.5` — first hit at rank 2 |
| `ndcg_at_k` | Like precision, but credits high ranks more than low ones. | `0.387` |
| `pool_recall_at_k` | Of the relevant passages the retriever found *at all*, how many reached the top-k? | `0.5` — 1 of the 2 found |
| `k` | The cutoff used. Set by `RETRIEVAL_EVAL_K`. | `3` |

### `recall_at_k` is the headline number

This is DPR's **top-k retrieval accuracy** — the fraction of questions whose top-k
passages contain an answer span. Later work (RAG, FiD, Atlas) publishes the same
quantity as recall@k, so it is the one directly comparable to published tables.

It is a per-query 0-or-1. Averaged over 1,000 queries, `0.62` means "62% of questions
had at least one usable passage in the top-k".

### `ndcg_at_k`, worked

Each relevant passage contributes `1 / log₂(rank + 1)`, then divide by the best
arrangement possible:

```
actual  = 0/log₂2 + 1/log₂3 + 0/log₂4  = 0.631
ideal   = 1/log₂2 + 1/log₂3            = 1.631   (both relevant at ranks 1-2)
ndcg@3  = 0.631 / 1.631                = 0.387
```

### `pool_recall_at_k` is not coverage — read it carefully

Its denominator is *what this retriever returned*, not what exists in the corpus.
True corpus recall would need all 35M passages labelled per query.

Because the denominator is self-referential, **a worse retriever can score higher**:

| | relevant found in top-100 | in top-10 | `pool_recall` | `recall_at_k` |
|---|---|---|---|---|
| Retriever A | 10 | 5 | **0.50** | 1.0 |
| Retriever B | 2 | 2 | **1.00** | 1.0 |

B found five times fewer relevant passages and scores twice as high. Read it as *rank
concentration* — "of what it found, how much did it push to the top" — and never quote
it as recall.

---

## Reranking — `stages.reranking`

Same four metrics, measured twice over the same window of size `reranker_top_k`:
**before** = the top-k you'd have kept from the dense ranking, **after** = the top-k
the reranker chose from the full candidate pool.

Reranked top-3 = ranks `[T, ., T]` (both relevant surfaced):

| key | before → after | reading |
|---|---|---|
| `recall_*` | `1.0 → 1.0` (Δ `0.0`) | already had a hit; can't improve |
| `mrr_*` | `0.5 → 1.0` (Δ `0.5`) | pulled the first hit from rank 2 to rank 1 |
| `precision_*` | `0.333 → 0.667` (Δ `0.333`) | 2 of 3 relevant instead of 1 of 3 |
| `ndcg_*` | `0.387 → 0.920` (Δ `0.533`) | both hits, both ranked high |

**Both sides share one ideal**, taken from the full candidate pool. If each side were
normalised against itself, a reranker could gain by *discarding* relevant passages —
dropping them shrinks its own denominator. Concretely: a pool with 5 relevant, reranked
to a top-10 keeping only 1 (at rank 1), scores **1.00 self-normalised** vs **0.339**
against the shared ideal. The first is nonsense.

A negative delta means the reranker made things worse. `PassthroughReranker` gives
Δ = 0 by construction — a useful sanity check that the harness is wired correctly.

---

## Generation — `stages.generation`

| key | plain English | note |
|---|---|---|
| `avg_exact_match` | Answer string is *identical* to the gold answer after normalisation. | brutal; near-0 for conversational output |
| `avg_rouge_l` | Word overlap with the gold answer, in order. | rewards partial credit |
| `avg_token_f1` | Token-overlap F1 against the gold answer, bag-of-words. | the SQuAD/NQ standard, reported here for comparability — not a headline |
| `avg_bert_score` | Semantic similarity via a BERT model — catches correct answers worded differently. | baseline-rescaled, so it spreads over [0, 1]; not comparable to raw published figures |
| `bert_score_hash` | The configuration that produced the score: model, layer, idf, library versions, `-rescaled`. | **quote this in the write-up** — the score is meaningless without it |
| `bert_score_error` | `null` normally. A string means the BERTScore pass failed (typically CUDA OOM) and `avg_bert_score` is null for that reason, not for want of answers. | every other metric in the block is still valid |

### Exact match

Normalisation is the SQuAD/NQ standard: lowercase → strip punctuation → drop the
articles *a/an/the* → collapse whitespace. Deviating makes the number incomparable to
published NQ results.

| prediction | gold | EM |
|---|---|---|
| `September 4, 1998` | `September 4, 1998` | `1.0` |
| `september 4 1998` | `September 4, 1998` | `1.0` — punctuation and case ignored |
| `It was September 4, 1998.` | `September 4, 1998` | `0.0` — extra words fail it |
| `the moon` | `moon` | `1.0` — article dropped |

The third row is why EM alone is misleading for a chat-style generator: the answer is
correct and scores zero. Read it alongside ROUGE-L and BERTScore.

### ROUGE-L, worked

Longest common subsequence of tokens, as an F1:

```
gold  "September 4, 1998"        -> 3 tokens
pred  "it was September 4 1998"  -> 5 tokens
LCS   september / 4 / 1998       -> 3

precision = 3/5 = 0.60    recall = 3/3 = 1.00
F1        = 2(0.60)(1.00) / (0.60 + 1.00) = 0.75
```

Same answer that scored `0.0` on exact match scores `0.75` here.

### Token F1, and why it is here at all

Bag-of-tokens overlap after the same normalisation exact match uses:

```
gold  "September 4, 1998"                     -> {september, 4, 1998}
pred  "It was September 4 1998 in California" -> {it, was, september, 4, 1998, in, california}
shared 3   precision = 3/7 = 0.43   recall = 3/3 = 1.00   F1 = 0.60
```

Published open-domain NQ results are ranked on exact match **and** this number. Neither of
the metrics this pipeline prefers appears in those tables — EM is ~0 here by construction,
and `answer_accuracy` is a containment criterion that published work does not use — so
without token F1 the generation results have no column that lines up with the literature.

Expect it to sit well below extractive published figures: the precision term divides by
every token in the answer, so prose around a three-word gold span is penalised even when
the answer is correct. Read it beside `answer_accuracy`, not instead of it.

### BERTScore, and why it is rescaled

Raw BERTScore is cosine similarity between contextual embeddings, and two *unrelated*
English sentences already score ~0.85 — so raw values bunch into [0.80, 0.90] and two
configs look identical when they are not. `rescale_with_baseline=True` subtracts that
baseline and rescales, spreading the same ordering across [0, 1]; a bad answer can go
slightly negative. Two consequences worth stating in the thesis: these numbers are
**not** comparable to raw BERTScore figures in published tables, and they are not
comparable to any earlier RAGNAR run made before this was turned on (2026-08-12).

Because the number depends entirely on which model and layer produced it, the library
emits a hash — `roberta-large_L17_no-idf_version=0.3.12(hug_trans=…)-rescaled` — and the
authors ask for it to be quoted in the paper. It is captured in `bert_score_hash` and
printed in the run summary. Report it next to the score.

BERTScore is also the one metric that runs a model on the GPU at report time. If it
OOMs, `avg_bert_score` is `null` and `bert_score_error` carries the exception — the
rest of the report is still written, and BERTScore can be recomputed from the trace
file, which holds every answer.

---

## Latency and throughput

Recorded per stage (`embed`, `retrieve`, `rerank`, `generate`).

| key | meaning |
|---|---|
| `avg_latency_ms` | Mean wall-clock time for that stage, per query. |
| `p95_latency_ms` | 95th-percentile, **nearest-rank** — an observed value, never interpolated. For 20 latencies of 1…20 ms this is `19.0`, not the max. |
| `throughput_qps` | Embedding only. **This is 1 ÷ mean latency, not a real throughput measurement** — queries run one at a time, unbatched. Do not present it as system capacity. |
| `avg_ttft_ms` | Time to first token — how long before the answer starts appearing. |
| `avg_tokens_per_sec` | Generation speed once started. 118 tokens in 1.486 s → `79.4`. |
| `avg_prompt_tokens` | Context size actually processed. **Also a truncation alarm**: if it plateaus near `num_ctx − num_predict`, passages are being silently dropped from the prompt. |
| `avg_completion_tokens` | Answer length in tokens. |

Total per-query latency ≈ embed + retrieve + rerank + generate; generation dominates
by roughly an order of magnitude.

### Warm-up — why the first query is not in these numbers

Every component does work on its first call that it never repeats: the embedder and
the cross-encoder move weights onto the GPU and compile kernels, FAISS touches index
pages, and Ollama loads the model into VRAM. Charged to query 1, that cost is tens of
seconds sitting inside a distribution whose median is milliseconds — it distorts the
mean, and at small sample sizes it *is* the p95.

So `online.eval.warmup_queries` (default 2) queries run before the measured loop and
are discarded. They are fixed built-in questions, not dataset samples: a warm-up query
that was also a measured query would come back warm the second time (Ollama prefix
cache, same IVF cells, same pages in the OS cache), which trades a cold outlier for an
artificially fast one. Their raw per-stage latencies are kept — in `warmup.latency_ms`
in the report and in the trace `_meta.json` — because they are the measurement of how
large the excluded cost was.

| key | meaning |
|---|---|
| `warmup.n_queries` / `queries` | How many were discarded, and which. |
| `warmup.latency_ms` | One dict per discarded query, raw. Query 1 vs query 2 is the size of the cold start on that node. |
| `warmup.failures` | Warm-up failures warn rather than abort — a discarded query is not worth killing a job over, and the measured loop has its own circuit breaker. |
| `warmup.total_s` | Excluded from `total_elapsed_s`. |

`warmup: null` in a report means warm-up was **switched off**, not that there was no
cold start — the cold start is then inside every latency above. `summary()` says so on
its own line.

**Caveat to state in the write-up:** with one pipeline per SLURM job, warm-up removes
the cold start *within* a run but not the difference *between* runs — two configs
measured on two nodes carry that node difference in their latencies. `hostname` and
`gpu` in the trace `_meta.json` are what makes it checkable.

---

## Offline build — `results/offline_FAISSDB_*.json`

Whole run:

| key | meaning |
|---|---|
| `n_passages` / `n_chunks` | Rows read from the TSV / chunks indexed. Equal, since passages are pre-chunked. |
| `chunk_size_avg/min/max` | Chunk length in **characters**. |
| `latency_ms.embed/index/total` | Time in the embedder, in FAISS `add`, and overall. Embedding dominates. |
| `embed_throughput.chunks_per_sec` | Build speed. |

Per batch (`batches[]`), for spotting drift mid-build:

| key | meaning |
|---|---|
| `is_training_batch` | `true` on the one batch where IVF training fired. **The vectors seen up to this point are the entire training sample** — if the corpus is ordered, those centroids are biased. |
| `n_skipped` / `skip_rate` | Chunks the embedder rejected. Should be ~0; a rise means malformed input. |
| `chunk_length_*` | Length distribution. Drives prompt size downstream — 10 chunks × mean chars ÷ ~4 = context tokens. |
| `embed_norm_*` | L2 norms **before** normalisation. Should be stable; drift signals an embedder or input problem. Under `metric="cosine"` these are normalised away, under `metric="dot"` magnitude carries meaning. |

---

## Known limits

1. **Relevance is string matching.** A correct passage worded differently counts as
   irrelevant. All retrieval metrics are lower bounds.
2. **`pool_recall_at_k` is not corpus recall** and can reward a worse retriever.
3. **13.6% of NQ "short" answers exceed 50 characters** — some are full sentences.
   These almost never match verbatim, so they depress retrieval metrics and force EM
   to ~0. Consider reporting short-answer and long-answer strata separately.
4. **No ANN recall.** Nothing here separates "IVF-PQ missed it" from "the embedder
   ranked it low". That needs a brute-force exact top-k on the same corpus.
5. **Point estimates only.** No confidence intervals, so small differences between
   configs are not yet distinguishable from noise.
