# Signature Verification Streamlit App

This folder is self-contained and uses `best_model_first.pt`. It preprocesses two
uploaded images with the repository's binarizing preprocessing pipeline, generates normalized
embeddings, and reports cosine similarity and a configurable match threshold.

## Run locally

```bash
cd streamlit
python -m pip install -r requirements.txt
streamlit run app.py
```

The app listens on `http://localhost:8501`.

## Deploy

Deploy this folder as the application root on Streamlit Community Cloud:

- Main file: `app.py`
- Python dependencies: `requirements.txt`

The checkpoint is included in this folder, so no additional model download is
required.
