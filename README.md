# Multi-Objective Recommender System

A professional-grade recommendation engine that balances **Relevance**, **Novelty**, and **Diversity** to overcome the "Filter Bubble" and "Popularity Bias" common in standard ML models.

## 🚀 The Innovation
Unlike standard recommenders that only optimize for accuracy (RMSE), this system implements a **Multi-Objective Re-ranking** layer. It uses a hybrid approach:
1.  **Predictive Layer:** A Matrix Factorization model (Truncated SVD) predicts user preferences based on latent factors.
2.  **Re-ranking Layer:** A custom algorithm that adjusts results based on:
    - **Relevance:** Predicted rating.
    - **Novelty:** Inverse popularity (surfacing "Hidden Gems").
    - **Diversity:** Category distribution (preventing redundant recommendations).

## 🛠️ Tech Stack
- **Language:** Python 3.12+
- **ML/Stats:** Scikit-Learn, Pandas, Numpy, SciPy
- **API:** FastAPI, Uvicorn
- **Database:** SQLite
- **Dashboard:** Streamlit
- **Env Management:** UV
- **Testing:** Pytest

## 📈 Project Roadmap & Milestones
- [x] **Data Engineering:** Raw data $\rightarrow$ Processed $\rightarrow$ Analytical Summaries.
- [x] **Deep Analytics:** Statistical proof of "Long Tail" distribution and Gini Coefficient.
- [x] **Feature Store:** Numerical encoding and Min-Max scaling of user/item features.
- [x] **Model Ladder:** Baseline (User-Average) $\rightarrow$ ML Model (SVD).
- [x] **Core Innovation:** Multi-Objective Re-ranker implementation.
- [x] **Productization:** Full-stack API and Interactive Dashboard.
- [x] **Quality Assurance:** Automated test suite for model and reranker logic.

## 🏃 Quick Start

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd multi-objective-recommender
uv sync
```

### 2. Setup Database
```bash
$env:PYTHONPATH = "."; uv run python scripts/migrate_to_db.py
```

### 3. Launch the Product
**Terminal 1 (Backend):**
```bash
$env:PYTHONPATH = "."; uv run python api/main.py
```

**Terminal 2 (Frontend):**
```bash
uv run streamlit run dashboard/app.py
```

## 🎓 Key Engineering Decisions
- **SVD with Mean Centering:** Implemented to handle dataset sparsity and remove zero-bias.
- **Absolute Pathing:** Ensured robust file handling across different OS environments.
- **Singleton Service Pattern:** Optimized API performance by loading the model into memory once.
- **Pydantic Validation:** Ensured type-safety for API requests.
