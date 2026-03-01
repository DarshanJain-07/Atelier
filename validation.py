import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

from schema import EMOTION_LABELS

load_dotenv()

# --- CONFIGURATION ---
# We use a robust, pre-trained Twitter Sentiment model as our Baseline
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# High-Resolution Baseline (28 classes) - Industry standard for GoEmotions
GO_EMOTIONS_MODEL = "sam-lowe/roberta-base-go_emotions"

# The 28 labels from the GoEmotions dataset
GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]


class Validator:
    def __init__(self):
        print("Loading Baseline AI (RoBERTa)...")
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Baseline AI Loaded.")

    def get_baseline_prob(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()[0]

    def map_plutchik_to_sentiment(self, emotion_probs_8dim):
        e = dict(zip(EMOTION_LABELS, emotion_probs_8dim))
        neg = e.get("Fear", 0) + e.get("Sadness", 0) + e.get("Disgust", 0) + e.get("Anger", 0)
        neu = e.get("Surprise", 0) + (0.5 * e.get("Anticipation", 0)) + (0.2 * e.get("Trust", 0))
        pos = e.get("Joy", 0) + (0.8 * e.get("Trust", 0)) + (0.5 * e.get("Anticipation", 0))
        total = neg + neu + pos + 1e-9
        return np.array([neg / total, neu / total, pos / total])

    def calculate_kl_divergence(self, system_probs_8dim, baseline_probs):
        q_baseline = np.array(baseline_probs)
        p_system = self.map_plutchik_to_sentiment(system_probs_8dim)
        q = np.clip(q_baseline, 1e-9, 1.0)
        q /= q.sum()
        p = np.clip(p_system, 1e-9, 1.0)
        p /= p.sum()
        kl_div = entropy(q, p)
        interpretation = "Consensus"
        if kl_div > 0.5: interpretation = "Significant Divergence"
        return {
            "baseline_probs": q_baseline.round(3).tolist(),
            "mapped_system_probs": p_system.round(3).tolist(),
            "kl_divergence": round(float(kl_div), 4),
            "interpretation": interpretation,
        }


class GoEmotionsValidator:
    def __init__(self):
        # Use HF_TOKEN from .env if available to bypass 401 errors
        token = os.getenv("HF_TOKEN", False)
        
        print(f"Loading High-Resolution Baseline AI ({GO_EMOTIONS_MODEL})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(GO_EMOTIONS_MODEL, token=token)
            self.model = AutoModelForSequenceClassification.from_pretrained(GO_EMOTIONS_MODEL, token=token)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print("GoEmotions Baseline AI Loaded.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load GoEmotions model. Check your HF_TOKEN in .env. Details: {e}")
            # Fallback values to prevent simulation crash
            self.model = None

    def get_baseline_prob(self, text: str):
        if self.model is None: return np.zeros(len(GO_EMOTIONS_LABELS))
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # RobertaGoEmotions often uses Sigmoid for multi-label, we normalize for KL
            probs = torch.sigmoid(outputs.logits) 
            probs = probs / probs.sum()
        return probs.cpu().numpy()[0]

    def map_goemotions_to_plutchik(self, go_probs_28dim):
        g = dict(zip(GO_EMOTIONS_LABELS, go_probs_28dim))
        p = {label: 0.0 for label in EMOTION_LABELS}

        # Mapping Logic (Same as before, adapted for GoEmotions labels)
        p["Joy"] = g.get("joy", 0) + g.get("amusement", 0) + g.get("excitement", 0) + g.get("love", 0) + g.get("pride", 0) + g.get("relief", 0) + g.get("gratitude", 0)*0.5
        p["Trust"] = g.get("admiration", 0) + g.get("approval", 0) + g.get("caring", 0) + g.get("gratitude", 0)*0.5 + g.get("optimism", 0)*0.5
        p["Fear"] = g.get("fear", 0) + g.get("nervousness", 0)
        p["Surprise"] = g.get("surprise", 0) + g.get("realization", 0) + g.get("confusion", 0) + g.get("curiosity", 0)
        p["Sadness"] = g.get("sadness", 0) + g.get("disappointment", 0) + g.get("disapproval", 0) + g.get("grief", 0) + g.get("remorse", 0) + g.get("embarrassment", 0)
        p["Disgust"] = g.get("disgust", 0)
        p["Anger"] = g.get("anger", 0) + g.get("annoyance", 0)
        p["Anticipation"] = g.get("desire", 0) + g.get("optimism", 0)*0.5

        plutchik_vec = np.array([p[label] for label in EMOTION_LABELS])
        return plutchik_vec / (plutchik_vec.sum() + 1e-9)

    def calculate_kl_divergence(self, system_probs_8dim, baseline_probs_28dim):
        if self.model is None:
            return {"baseline_plutchik": [], "kl_divergence": 0.0, "interpretation": "Baseline Offline"}

        q_baseline_8 = self.map_goemotions_to_plutchik(baseline_probs_28dim)
        p_system_8 = np.array(system_probs_8dim)

        q = np.clip(q_baseline_8, 1e-9, 1.0)
        q /= q.sum()
        p = np.clip(p_system_8, 1e-9, 1.0)
        p /= p.sum()
        
        kl_div = entropy(q, p)
        
        interpretation = "Consensus"
        if kl_div > 0.4: interpretation = "Nuance Detected"
        if kl_div > 0.8: interpretation = "Significant Divergence"

        return {
            "baseline_plutchik": q_baseline_8.round(3).tolist(),
            "kl_divergence": round(float(kl_div), 4),
            "interpretation": interpretation,
        }
