from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from preprocessing.signature_preprocessing import preprocess_signature, to_model_input
from training.stage3_metric_learning import SiameseSignatureModel


APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "best_model_first.pt"
DEFAULT_THRESHOLD = 0.70


@st.cache_resource
def load_model() -> SiameseSignatureModel:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False,
        embedding_dim=int(checkpoint["embedding_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def decode_upload(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {uploaded_file.name}")
    return image


def prepare_image(image: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    processed = preprocess_signature(image, binarize=True)
    tensor = torch.from_numpy(to_model_input(processed)).unsqueeze(0)
    return processed, tensor


@torch.inference_mode()
def similarity(model: SiameseSignatureModel, first: torch.Tensor, second: torch.Tensor) -> float:
    embeddings = model(torch.cat((first, second), dim=0))
    return float(torch.sum(embeddings[0] * embeddings[1]).item())


def similarity_percentage(score: float) -> float:
    return max(0.0, score) * 100.0


st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="centered")
st.title("Signature Verification")
st.write("Compare two signature images using a Siamese ConvNeXt-Tiny model.")

st.sidebar.header("Decision settings")
threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_THRESHOLD,
    step=0.01,
    help="Scores at or above this threshold are classified as a likely match.",
)
st.sidebar.caption("The threshold is adjustable because it depends on your operating point.")

first_file = st.file_uploader(
    "Reference signature",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    key="reference",
)
second_file = st.file_uploader(
    "Query signature",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    key="query",
)

if first_file and second_file:
    try:
        first_image = decode_upload(first_file)
        second_image = decode_upload(second_file)
        first_processed, first_tensor = prepare_image(first_image)
        second_processed, second_tensor = prepare_image(second_image)
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()

    left, right = st.columns(2)
    with left:
        st.image(cv2.cvtColor(first_processed, cv2.COLOR_GRAY2RGB), caption="Processed reference")
    with right:
        st.image(cv2.cvtColor(second_processed, cv2.COLOR_GRAY2RGB), caption="Processed query")

    try:
        model = load_model()
        score = similarity(model, first_tensor, second_tensor)
    except (FileNotFoundError, RuntimeError, KeyError) as error:
        st.error(f"Model could not be loaded: {error}")
        st.stop()

    matched = score >= threshold
    percentage = similarity_percentage(score)
    metric_left, metric_right = st.columns(2)
    with metric_left:
        st.metric("Cosine similarity", f"{score:.4f}")
    with metric_right:
        st.metric("Similarity percentage", f"{percentage:.2f}%")
    if matched:
        st.success(f"Likely genuine match (score ≥ {threshold:.2f})")
    else:
        st.error(f"Likely mismatch (score < {threshold:.2f})")
else:
    st.info("Upload one reference signature and one query signature to begin.")
