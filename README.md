# Hacker News ML trends: a LangGraph agent for the HN AI/ML slice

Learning-first walkthrough of building a **stateful LangGraph agent**
that ingests Hacker News stories, classifies them into subtopics,
embeds and clusters them, detects trending topics over time, and
produces a human-readable summary.

The point is not another HN scraper. It is to build a real multi-node
agent workflow end to end using open-source tooling: the public HN
Firebase API for ingestion, `sentence-transformers` for embeddings,
`scikit-learn` for the classical baselines, **Ollama** for local LLM
inference, **LangChain** for prompt orchestration, and **LangGraph**
for the state machine and conditional drill-down.

Hacker News was chosen over subreddit APIs because it has no auth
walls, no rate-limit gates, and no policy applications: the entire
API is a public read-only Firebase JSON endpoint documented at
<https://github.com/HackerNews/API>.

> **Status: scaffold.** Repository layout and dependencies are in place.
> Implementations land per notebook.

## What this project does

- **Fetches** stories from the HN public API (`topstories`, `newstories`,
  `item/{id}` endpoints), no authentication required.
- **Labels** a small gold set by hand to build ground truth for
  classification evaluation.
- **Baselines** with a classical TF-IDF + logistic regression classifier
  so we know what a serious non-LLM baseline looks like.
- **Classifies** with an Ollama-hosted LLM (`llama3.1:8b`) via
  LangChain, and compares against the classical baseline.
- **Embeds** stories with a sentence-transformer and clusters them
  (KMeans / HDBSCAN) to find themes the labels miss.
- **Detects trends** by comparing subtopic frequency and total scores
  across time windows.
- **Orchestrates** the whole pipeline as a LangGraph state machine
  with a conditional *drill-down* node: when a subtopic spikes, the
  agent pulls more stories on that topic and produces a deeper
  summary.

## Tech stack

| Layer          | Library                                                                |
| -------------- | ---------------------------------------------------------------------- |
| Ingestion      | `requests` against the HN Firebase API (public, no auth)               |
| Storage        | `pandas` + local CSV                                                   |
| Classical NLP  | `scikit-learn` (TF-IDF, logistic regression)                           |
| Embeddings     | `sentence-transformers` (all-MiniLM-L6-v2, 384-dim, runs locally)      |
| Clustering     | `scikit-learn` (KMeans), `hdbscan` (density)                           |
| LLM            | **Ollama** running `llama3.1:8b` locally (no API costs)                |
| Orchestration  | **LangChain** for LLM calls + prompts, **LangGraph** for the agent    |
| Evaluation     | classification metrics + cluster purity + manual trend verification    |

## Quick start

```bash
git clone https://github.com/Anjanamb/hn-ml-trends.git
cd hn-ml-trends
python -m venv venv
source venv/Scripts/activate     # Windows Git Bash; on macOS/Linux use bin/activate
pip install -r requirements.txt

# Copy the config template (only Ollama settings are needed).
cp .env.example .env

# Ollama must be running locally with the model pulled:
ollama pull llama3.1:8b

jupyter lab
```

Then open `notebooks/00_intro.ipynb` and work through the numbered
notebooks in order.

## Notebooks

| #  | File                                                                              | What it does                                                                                                              |
| -- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 00 | [`00_intro.ipynb`](notebooks/00_intro.ipynb)                                      | The problem, dataset, framing, roadmap                                                                                    |
| 01 | [`01_fetch_and_eda.ipynb`](notebooks/01_fetch_and_eda.ipynb)                      | Pull `topstories` + `newstories`, describe volume, score curves, type distribution                                        |
| 02 | [`02_manual_gold_labels.ipynb`](notebooks/02_manual_gold_labels.ipynb)            | Define 7 subtopic categories, hand-label 200-300 stories as ground truth                                                  |
| 03 | [`03_tfidf_baseline.ipynb`](notebooks/03_tfidf_baseline.ipynb)                    | Classical baseline: TF-IDF + logistic regression, evaluated on the gold set                                               |
| 04 | [`04_ollama_llm_classifier.ipynb`](notebooks/04_ollama_llm_classifier.ipynb)      | Few-shot prompt via LangChain against a local Ollama model, evaluated on the same gold set                                |
| 05 | [`05_embeddings_and_clusters.ipynb`](notebooks/05_embeddings_and_clusters.ipynb)  | Sentence embeddings, KMeans + HDBSCAN clustering, cluster-vs-label agreement                                              |
| 06 | [`06_langgraph_agent.ipynb`](notebooks/06_langgraph_agent.ipynb)                  | The full state machine: multi-node graph, conditional drill-down when a subtopic spikes                                   |
| 07 | [`07_trend_report.ipynb`](notebooks/07_trend_report.ipynb)                        | Trend detection + LLM-generated summary; manual verification of the top-N trending subtopics                              |

## Repository layout

```text
hn-ml-trends/
├── data/                              Fetched stories, embeddings, labels (gitignored)
├── notebooks/                         The eight notebooks
├── src/                               Thin reusable modules
│   ├── hn_client.py       HN Firebase-API client (top/new stories, item, comments)
│   ├── labels.py          Category definitions + gold-label I/O
│   ├── llm.py             Ollama + LangChain wrapper
│   ├── embed.py           Sentence-transformer wrapper + similarity helpers
│   ├── classify.py        TF-IDF baseline + LLM classifier heads
│   ├── trend.py           Trend detection (frequency + score deltas)
│   ├── agent.py           LangGraph state machine
│   └── evaluate.py        Classification metrics, cluster purity
├── tests/                             Sanity checks for src/
├── requirements.txt                   requests, langchain, langgraph, ollama, sklearn, sentence-transformers
├── Dockerfile                         Reproducible env for anyone who wants a container
├── .env.example                       Template for local config (Ollama host + model)
└── LICENSE                            MIT
```

## Running the tests

```bash
python -m pytest tests/ -q
```

## Running in Docker

```bash
docker build -t hn-ml-trends .
docker run --rm -p 8888:8888 -v "$(pwd)":/app --env-file .env hn-ml-trends
```

The container does **not** include Ollama; keep it running on the host
and the notebooks will connect to `http://host.docker.internal:11434`
(Docker Desktop) or the host's IP.

## License

MIT. See [`LICENSE`](LICENSE).
