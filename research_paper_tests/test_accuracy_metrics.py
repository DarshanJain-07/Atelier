import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import pipeline

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from input_layer import get_world_state
from physics_engine import SocialPhysicsEngine
from schema import EMOTION_LABELS, PSYCH_PROJECTION, SimConfig

def map_plutchik_to_sentiment(emotion_probs_8dim):
    """Maps ATELIER's 8D Plutchik output to (Negative, Neutral, Positive) probabilities."""
    e = dict(zip(EMOTION_LABELS, emotion_probs_8dim))
    neg = e.get("Fear", 0) + e.get("Sadness", 0) + e.get("Disgust", 0) + e.get("Anger", 0)
    neu = e.get("Surprise", 0) + (0.5 * e.get("Anticipation", 0)) + (0.2 * e.get("Trust", 0))
    pos = e.get("Joy", 0) + (0.8 * e.get("Trust", 0)) + (0.5 * e.get("Anticipation", 0))
    
    total = neg + neu + pos + 1e-9
    return np.array([neg / total, neu / total, pos / total])

def get_atelier_sentiment(prompt, config, df_meta, exposures, personalities, affinities, adjacency_matrix):
    """Runs the full ATELIER pipeline and returns the sentiment probabilities."""
    # 1. LLM
    try:
        world_tensor, urgency, is_personal, detected_biases, reasoning = get_world_state(prompt)
    except Exception as e:
        print(f"ATELIER LLM Failed for prompt '{prompt}': {e}")
        return None, []
        
    # 2. Cognitive Engine
    agent_memory = torch.zeros_like(exposures)
    cog_engine = CognitiveEngine(config)
    ctx, att, eng, agent_memory = cog_engine.run(
        world_tensor_raw=world_tensor,
        urgency=urgency,
        is_personal=is_personal,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=agent_memory,
    )
    
    # 3. Social Physics Engine
    phys_engine = SocialPhysicsEngine(config)
    device = ctx.device
    projection_matrix = PSYCH_PROJECTION.to(device)
    final_emotions = torch.matmul(ctx, projection_matrix)
    final_emotions = F.softmax(final_emotions / max(0.01, config.emotion_temperature), dim=1)
    
    influence = df_meta["Influence"].to_numpy()
    social_state = phys_engine.aggregate_society(
        final_emotions, influence, eng, adjacency_matrix
    )
    
    # Map to sentiment
    obj_center = social_state["objective_center"]
    sentiment_probs = map_plutchik_to_sentiment(obj_center)
    return sentiment_probs, detected_biases

def test_accuracy_comparisons():
    print("--- Testing Accuracy & Spin Detection vs Baseline Models ---")

    # The goal is to show ATELIER detects underlying negative material realities 
    # even when the text uses positive spin, whereas standard sentiment models fail.
    
    prompts = [
        {
            "text": "The company is optimizing its workforce by letting go of 10,000 employees.",
            "ground_truth_label": "Negative",
            "type": "Corporate Spin"
        },
        {
            "text": "We are thrilled to announce a strategic realignment that will allow our previous executive board to pursue new opportunities outside the firm.",
            "ground_truth_label": "Negative",
            "type": "Corporate Euphemism"
        },
        {
            "text": "The controversial politician gave a speech claiming a massive economic victory despite rising inflation.",
            "ground_truth_label": "Negative",
            "type": "Political Spin"
        },
        {
            "text": "The government is implementing a new patriotic surveillance program to ensure the ultimate safety and freedom of all loyal citizens.",
            "ground_truth_label": "Negative",
            "type": "Authoritarian Framing"
        },
        {
            "text": "Our community garden initiative received record funding and successfully planted 500 new trees.",
            "ground_truth_label": "Positive",
            "type": "Objective Positive"
        }
    ]

    print("\nLoading Baseline Models...")
    # 1. DistilBERT SST-2 (Standard Positive/Negative)
    pipe_sst2 = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # 2. Twitter RoBERTa (Negative, Neutral, Positive)
    pipe_roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    
    # 3. J-Hartmann Emotion (Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise)
    pipe_emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    # Generate ATELIER Society
    print("\nGenerating ATELIER Society...")
    config = SimConfig(num_agents=1000)
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)

    results = []

    for i, p in enumerate(prompts):
        text = p["text"]
        print(f"\n[{i+1}/{len(prompts)}] Analyzing: '{text}'")
        print(f"  Ground Truth: {p['ground_truth_label']} ({p['type']})")
        
        # --- Run Baseline 1: SST-2 ---
        res_sst2 = pipe_sst2(text)[0]
        sst2_label = res_sst2['label'].capitalize()
        
        # --- Run Baseline 2: RoBERTa ---
        res_roberta = pipe_roberta(text)[0]
        # LABEL_0: negative, LABEL_1: neutral, LABEL_2: positive
        rob_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        roberta_label = rob_map.get(res_roberta['label'], "Unknown")
        
        # --- Run Baseline 3: J-Hartmann Emotion ---
        res_emo = pipe_emotion(text)[0]
        emo_label = res_emo['label'].capitalize()
        if emo_label in ["Anger", "Disgust", "Fear", "Sadness"]:
            hartmann_sentiment = "Negative"
        elif emo_label == "Joy":
            hartmann_sentiment = "Positive"
        else:
            hartmann_sentiment = "Neutral"

        # --- Run ATELIER ---
        time.sleep(2) # Avoid Gemini Rate Limits
        atelier_probs, biases = get_atelier_sentiment(text, config, df_meta, exposures, personalities, affinities, adjacency_matrix)
        
        if atelier_probs is not None:
            # Argmax [Neg, Neu, Pos]
            idx = np.argmax(atelier_probs)
            atelier_label = ["Negative", "Neutral", "Positive"][idx]
        else:
            atelier_label = "Error"
            biases = []

        print(f"  > SST-2:       {sst2_label}")
        print(f"  > RoBERTa:     {roberta_label}")
        print(f"  > Hartmann:    {hartmann_sentiment} (Raw: {emo_label})")
        print(f"  > ATELIER:     {atelier_label}")
        print(f"    Biases Found: {biases}")

        results.append({
            "Prompt Type": p["type"],
            "Ground Truth": p["ground_truth_label"],
            "SST-2 (Baseline)": sst2_label,
            "RoBERTa (Baseline)": roberta_label,
            "Hartmann (Baseline)": hartmann_sentiment,
            "ATELIER": atelier_label
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Accuracies
    def calc_acc(model_col):
        correct = sum(df[model_col] == df["Ground Truth"])
        return (correct / len(df)) * 100

    print("\n--- Accuracy Report ---")
    acc_sst2 = calc_acc("SST-2 (Baseline)")
    acc_roberta = calc_acc("RoBERTa (Baseline)")
    acc_hartmann = calc_acc("Hartmann (Baseline)")
    acc_atelier = calc_acc("ATELIER")

    print(f"SST-2 Accuracy:       {acc_sst2:.1f}%")
    print(f"RoBERTa Accuracy:     {acc_roberta:.1f}%")
    print(f"Hartmann Accuracy:    {acc_hartmann:.1f}%")
    print(f"ATELIER Accuracy:     {acc_atelier:.1f}%")

    # Plotting
    models = ["SST-2", "RoBERTa", "Hartmann", "ATELIER"]
    accuracies = [acc_sst2, acc_roberta, acc_hartmann, acc_atelier]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#ff9999', '#ff9999', '#ff9999', '#66b3ff'])
    
    plt.title("Model Accuracy on Deceptive/Spin Prompts", fontsize=15, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 110)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = os.path.join(os.path.dirname(__file__), "accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "accuracy_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw data to: {csv_path}")

if __name__ == "__main__":
    test_accuracy_comparisons()
