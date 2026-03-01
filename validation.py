import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from schema import EMOTION_LABELS

# --- CONFIGURATION ---
# We use a robust, pre-trained Twitter Sentiment model as our Baseline
# It outputs 3 classes: [Negative, Neutral, Positive]
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


class Validator:
    def __init__(self):
        print("Loading Baseline AI (RoBERTa)...")
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Baseline AI Loaded.")

    def get_baseline_prob(self, text: str):
        """
        Get standard sentiment probabilities [Neg, Neu, Pos].
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()[0]  # Shape (3,)

    def map_plutchik_to_sentiment(self, emotion_probs_8dim):
        """
        Maps our 8-dim Plutchik vector to the 3-dim [Neg, Neu, Pos] space
        to allow comparison with the Baseline.

        Our Order: Based on EMOTION_LABELS (e.g., [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation])
        Target Order: [Negative, Neutral, Positive]
        """
        # Create a dictionary mapping emotion names to their values
        e = dict(zip(EMOTION_LABELS, emotion_probs_8dim))

        # Mapping Logic (Weights):
        # Negative = Fear + Sadness + Disgust + Anger
        neg = e.get("Fear", 0) + e.get("Sadness", 0) + e.get("Disgust", 0) + e.get("Anger", 0)

        # Neutral = Surprise + (0.5 * Anticipation) + (0.2 * Trust - skepticism buffer)
        neu = e.get("Surprise", 0) + (0.5 * e.get("Anticipation", 0)) + (0.2 * e.get("Trust", 0))

        # Positive = Joy + (0.8 * Trust) + (0.5 * Anticipation)
        pos = e.get("Joy", 0) + (0.8 * e.get("Trust", 0)) + (0.5 * e.get("Anticipation", 0))

        # Normalize to sum to 1.0
        total = neg + neu + pos + 1e-9
        return np.array([neg / total, neu / total, pos / total])

    def calculate_kl_divergence(self, system_probs_8dim, baseline_probs):
        """
        Calculates how much our system diverges from the standard baseline.

        Args:
            system_probs_8dim: List/Array of 8 float probabilities from PhysicsEngine.
            baseline_probs: Pre-calculated [Neg, Neu, Pos] probabilities from the baseline AI.

        Returns:
            dict: {
                "baseline_probs": [Neg, Neu, Pos],
                "mapped_system_probs": [Neg, Neu, Pos],
                "kl_divergence": float,
                "interpretation": str
            }
        """
        # 1. Get Q (Baseline)
        q_baseline = np.array(baseline_probs)

        # 2. Get P (Our System, Mapped)
        p_system = self.map_plutchik_to_sentiment(system_probs_8dim)

        # 3. Calculate KL Divergence (Entropy)
        # D_kl(Q || P) - How much information is lost if we approximate Q with P?
        q = np.clip(q_baseline, 1e-9, 1.0)
        q /= q.sum()
        p = np.clip(p_system, 1e-9, 1.0)
        p /= p.sum()
        kl_div = entropy(q, p)

        # 4. Interpret
        interpretation = "Consensus"
        if kl_div > 0.5:
            interpretation = "Significant Divergence (Nuance Detected)"
        if kl_div > 1.0:
            interpretation = "High Anomaly (Model Disagreement)"

        return {
            "baseline_probs": q_baseline.round(3).tolist(),
            "mapped_system_probs": p_system.round(3).tolist(),
            "kl_divergence": round(float(kl_div), 4),
            "interpretation": interpretation,
        }


if __name__ == "__main__":
    # Test
    val = Validator()

    # Mock Data
    text = "The stock market crashed today."
    baseline_probs = val.get_baseline_prob(text)
    # Our system says: High Fear, High Anger (Negative)
    # [Joy, Tru, Fea, Sur, Sad, Dis, Ang, Ant]
    system_output = [0.0, 0.0, 0.6, 0.1, 0.1, 0.0, 0.2, 0.0]

    result = val.calculate_kl_divergence(system_output, baseline_probs)

    print(f"Input: {text}")
    print(f"Baseline (Standard AI (Neg, Neu, Pos)): {result['baseline_probs']}")
    print(f"Our System (Mapped (Neg, Neu, Pos)):    {result['mapped_system_probs']}")
    print(f"Divergence Score:       {result['kl_divergence']}")
    print(f"Status:                 {result['interpretation']}")
