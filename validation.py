import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from schema import emotions_to_sentiment_distribution

load_dotenv()

# --- CONFIGURATION ---
# We use a robust, pre-trained Twitter Sentiment model as our Baseline
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


class Validator:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        import threading

        self.lock = threading.Lock()

    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return

        with self.lock:
            # Double checked locking
            if self.model is not None and self.tokenizer is not None:
                return

            print("Loading Baseline AI (RoBERTa)...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(
                    HF_MODEL_NAME,
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                # Assign atomically
                self.tokenizer = tokenizer
                self.model = model
                self.device = device
                print("Baseline AI Loaded.")
            except Exception as e:
                print(f"Failed to load Baseline AI: {e}")
                raise e

    def get_baseline_prob(self, text: str):
        self._load_model()

        if self.tokenizer is None or self.model is None or self.device is None:
            raise RuntimeError("Baseline model or tokenizer failed to load.")

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()[0]

    def map_plutchik_to_sentiment(self, emotion_probs_8dim):
        return emotions_to_sentiment_distribution(emotion_probs_8dim).cpu().numpy()
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

    def validate_stewing(self, negative_integral: float, ticks: int):
        """Interprets the long-term impact based on the sustained negative emotion
        over multiple time ticks.
        """
        avg_negativity = negative_integral / max(1, ticks)

        if avg_negativity > 0.8:
            return "Deep Structural Consequence (Severe, sustained outrage)"
        if avg_negativity > 0.4:
            return "Lingering Resentment (Slow-burn polarization)"
        return "Flash in the Pan (Rapid decay of negative arousal)"
