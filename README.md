# hnsw-from-scratch

An implementation of approximate nearest-neighbour search in Python, progressing
from exact brute-force search to HNSW, with benchmarks on SIFT1M against exact
search and FAISS.

**Status: work in progress.** Brute-force kNN is partly done — `l2_distance` is
implemented but `knn_bruteforce` is not, so `pytest` is expected to be partially
red until that milestone is finished.

## Why

Retrieval is the layer underneath every RAG system, and it is usually a single
call to a vector database. This project implements that layer directly — the
graph structure, the search, and the trade-off between recall and latency — so
the behaviour and cost of an approximate index are measured rather than assumed.

Exact search is the reference implementation throughout: an approximate index
is only correct to the extent that it agrees with brute force.

## Roadmap

- [ ] **Brute-force kNN.** Exact nearest neighbours over a vector set. Serves
      as the correctness oracle for everything that follows.
- [ ] **Single-layer graph search.** A navigable small-world graph with greedy
      search, introducing measurable recall below 100%.
- [ ] **Hierarchy.** Multiple layers with exponential level assignment — the
      full HNSW algorithm.
- [ ] **Benchmark.** Recall@10 against queries-per-second on SIFT1M, compared
      with exact search and FAISS.

## Layout

```
src/hnsw/    the library; `bruteforce.py` is the exact-search reference that
             the approximate index is measured against
tests/       one `test_<module>.py` per library module
```

## Usage

```bash
pip install -e ".[dev]"
pytest -q
```

## Reference

Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs* — [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
