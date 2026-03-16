import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.metrics import davies_bouldin_score, silhouette_score
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
                    HF_MODEL_NAME
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

    def calculate_cluster_metrics(self, final_emotions: torch.Tensor):
        try:
            features = final_emotions.cpu().numpy()
            labels = np.argmax(features, axis=1)

            unique_labels = np.unique(labels)
            
            per_cluster_sil = {}
            per_cluster_db = {}
            
            if len(unique_labels) > 1 and len(unique_labels) < len(labels):
                sil_score = silhouette_score(features, labels)
                db_score = davies_bouldin_score(features, labels)
                
                from sklearn.metrics import silhouette_samples
                from sklearn.metrics.pairwise import pairwise_distances
                
                sil_samples = silhouette_samples(features, labels)
                for label in unique_labels:
                    emotion_name = EMOTION_LABELS[label]
                    per_cluster_sil[emotion_name] = float(np.mean(sil_samples[labels == label]))
                    
                centroids = []
                s_i = []
                for label in unique_labels:
                    cluster_points = features[labels == label]
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)
                    s_i.append(np.mean(pairwise_distances(cluster_points, [centroid])))
                    
                centroids_dist = pairwise_distances(centroids)
                
                for i, label in enumerate(unique_labels):
                    max_R_ij = 0.0
                    for j in range(len(unique_labels)):
                        if i != j and centroids_dist[i, j] > 0:
                            R_ij = (s_i[i] + s_i[j]) / centroids_dist[i, j]
                            if R_ij > max_R_ij:
                                max_R_ij = R_ij
                    emotion_name = EMOTION_LABELS[label]
                    per_cluster_db[emotion_name] = float(max_R_ij)
            else:
                sil_score = 0.0
                db_score = 0.0
                per_cluster_sil = {EMOTION_LABELS[label]: 0.0 for label in unique_labels}
                per_cluster_db = {EMOTION_LABELS[label]: 0.0 for label in unique_labels}

            return {
                "silhouette_score": round(float(sil_score), 4),
                "davies_bouldin_index": round(float(db_score), 4),
                "per_cluster_silhouette": {k: round(v, 4) for k, v in per_cluster_sil.items()},
                "per_cluster_davies_bouldin": {k: round(v, 4) for k, v in per_cluster_db.items()}
            }
        except Exception as e:
            print(f"Failed to calculate cluster metrics: {e}")
            return {
                "silhouette_score": 0.0, 
                "davies_bouldin_index": 0.0,
                "per_cluster_silhouette": {},
                "per_cluster_davies_bouldin": {}
            }

    def validate_stewing(self, negative_integral: float, ticks: int):
        """
        Interprets the long-term impact based on the sustained negative emotion
        over multiple time ticks.
        """
        avg_negativity = negative_integral / max(1, ticks)
        
        if avg_negativity > 0.8:
            return "Deep Structural Consequence (Severe, sustained outrage)"
        elif avg_negativity > 0.4:
            return "Lingering Resentment (Slow-burn polarization)"
        else:
            return "Flash in the Pan (Rapid decay of negative arousal)"
