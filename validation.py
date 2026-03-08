import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from schema import EMOTION_LABELS

load_dotenv()

# --- CONFIGURATION ---
# We use a robust, pre-trained Twitter Sentiment model as our Baseline
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


class Validator:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None

    def _load_model(self):
        if self.model is not None:
            return

        print("Loading Baseline AI (RoBERTa)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL_NAME
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print("Baseline AI Loaded.")
        except Exception as e:
            print(f"Failed to load Baseline AI: {e}")
            raise e

    def get_baseline_prob(self, text: str):
        self._load_model()

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()[0]

    def map_plutchik_to_sentiment(self, emotion_probs_8dim):
        e = dict(zip(EMOTION_LABELS, emotion_probs_8dim))
        neg = (
            e.get("Fear", 0)
            + e.get("Sadness", 0)
            + e.get("Disgust", 0)
            + e.get("Anger", 0)
        )
        neu = (
            e.get("Surprise", 0)
            + (0.5 * e.get("Anticipation", 0))
            + (0.2 * e.get("Trust", 0))
        )
        pos = (
            e.get("Joy", 0)
            + (0.8 * e.get("Trust", 0))
            + (0.5 * e.get("Anticipation", 0))
        )
        total = neg + neu + pos + 1e-9
        return np.array([neg / total, neu / total, pos / total])

    def calculate_divergence(self, system_probs_8dim, baseline_probs):
        q_baseline = np.array(baseline_probs)
        p_system = self.map_plutchik_to_sentiment(system_probs_8dim)

        from scipy.stats import entropy, wasserstein_distance

        indices = np.arange(len(p_system))

        # Normalize just in case and clip for entropy
        q = np.clip(q_baseline, 1e-9, 1.0)
        q /= q.sum()
        p = np.clip(p_system, 1e-9, 1.0)
        p /= p.sum()

        w_dist = wasserstein_distance(indices, indices, u_weights=p, v_weights=q)
        kl_div = entropy(q, p)

        interpretation = "Consensus"

        if w_dist > 0.2:
            interpretation = "Significant Divergence"

        return {
            "baseline_probs": q_baseline.tolist(),
            "wasserstein_distance": round(float(w_dist), 4),
            "kl_divergence": round(float(kl_div), 4),
            "interpretation": interpretation,
        }
