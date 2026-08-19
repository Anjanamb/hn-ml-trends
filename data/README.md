# Data

This directory is empty in the git repository. The pipeline populates
it locally from the Hacker News public API (notebook 01) and from
downstream notebooks (embeddings, cluster labels, agent-generated
reports).

**Expected files after running the notebooks end to end:**

| File                     | Written by | Contents                                          |
| ------------------------ | ---------- | ------------------------------------------------- |
| `stories_raw.csv`        | nb01       | Raw fetched stories + metadata                    |
| `comments_top.csv`       | nb01       | Top comments per story                            |
| `gold_labels.csv`        | nb02       | Hand-labelled category ground truth               |
| `predictions_tfidf.csv`  | nb03       | TF-IDF baseline predictions                       |
| `predictions_llm.csv`    | nb04       | Ollama LLM predictions                            |
| `embeddings.npy`         | nb05       | Story embeddings from sentence-transformers       |
| `clusters.csv`           | nb05       | Cluster assignments per story                     |
| `trend_report.md`        | nb07       | Final human-readable trend summary                |

All files here are gitignored (see `.gitignore`); only this README is
committed.
