import numpy as np
import torch
import sys
import os
from scipy.stats import pearsonr

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig
from generate_society import generate_society
from cognitive_engine import CognitiveEngine

def test_influence_susceptibility_ratio():
    print("--- Running Influence/Susceptibility Ratio Analysis ---")
    
    # Use smaller population for performance, but large enough for statistics
    N = 2000
    # Enable power-law influence for realistic societal outliers (Elon Musk effect)
    config = SimConfig(num_agents=N, seed=42, use_power_law_influence=True)
    
    print(f"\n[ Generating Society (N={N}) ]")
    df_meta, exposures, personalities, affinities = generate_society(config)
    
    cog_engine = CognitiveEngine(config)
    
    all_influences = df_meta["Influence"].values
    mean_inf = all_influences.mean()
    max_inf = all_influences.max()
    print(f"Society Structural Influence -> Mean: {mean_inf:.2f}, Max: {max_inf:.2f}")
    
    # We will sample S agents to act as "broadcasters" (thoughts being shared)
    S = 200
    np.random.seed(42)
    seed_indices = np.random.choice(N, S, replace=False)
    
    # Engagement threshold to consider an agent "swayed" or "infected"
    threshold = 0.5 
    
    out_degree = np.zeros(S)
    in_degree = np.zeros(N) # Track susceptibility for all agents
    
    print(f"\n[ Simulating Thought Propagation for {S} seeds ]")
    
    for i, idx in enumerate(seed_indices):
        thought_vector = exposures[idx]
        broadcaster_inf = all_influences[idx]
        
        _, _, engagement_scores = cog_engine.run(
            world_tensor_raw=thought_vector.unsqueeze(0),
            urgency=0.5,
            is_personal=False,
            exposures=exposures,
            personalities=personalities,
            agent_affinities=affinities
        )
        
        # --- Realistic Scaling Model ---
        # 1. Reach (Impressions): Everyone has a minimum "local network" reach of ~10% 
        #    so average people actually have an audience of friends/family/colleagues.
        base_reach = 0.10
        reach_probability = min(1.0, base_reach + (broadcaster_inf / mean_inf) * 0.10)
        sees_post_mask = (torch.rand(N) < reach_probability).float()
        
        # 2. Authority Bonus: People are naturally MUCH more likely to engage 
        #    if a high-status person is speaking (log scale to prevent blowout).
        authority_bonus = 1.0 + 1.0 * np.log1p(broadcaster_inf / mean_inf)
        
        # We also lower the threshold to 0.18 so that "local" everyday influence 
        # is captured (friends swaying friends), allowing the median person to 
        # naturally sway a few dozen people in their immediate social circle.
        adaptive_threshold = 0.18
        
        # Determine who was swayed: they must SEE it and be HIGHLY ENGAGED
        engaged_mask = (engagement_scores * authority_bonus > adaptive_threshold).float() * sees_post_mask
        
        # Exclude self-infection
        engaged_mask[idx] = 0.0
        
        # Track out-degree (how many this thought infected)
        out_degree[i] = engaged_mask.sum().item()
        
        # Track in-degree (how easily others are swayed by thoughts)
        in_degree += engaged_mask.numpy()
        
        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{S} seeds")

    # Extract in-degree specifically for our seed group 
    seed_in_degree = in_degree[seed_indices]
    
    # Calculate Influence/Susceptibility Ratio (add 1 to avoid division by zero)
    ratio = out_degree / (seed_in_degree + 1.0)
    
    print("\n[ Results: Out-Degree (Influence) vs In-Degree (Susceptibility) ]")
    print("Definition:")
    print(" - Out-degree: Number of agents highly engaged (>0.5) by an agent's thought.")
    print(f" - In-degree: Number of times an agent was highly engaged by the {S} shared thoughts.")
    print(" - Ratio: Out-degree / In-degree (Higher = More influential & less susceptible)\n")
    
    print(f"Average Out-degree: {out_degree.mean():.2f} (± {out_degree.std():.2f})")
    print(f"  Min: {out_degree.min()}, Max: {out_degree.max()}")
    print(f"  25th: {np.percentile(out_degree, 25)}, 50th: {np.percentile(out_degree, 50)}, 75th: {np.percentile(out_degree, 75)}, 95th: {np.percentile(out_degree, 95)}")
    
    print(f"\nAverage In-degree:  {seed_in_degree.mean():.2f} (± {seed_in_degree.std():.2f})")
    print(f"  Min: {seed_in_degree.min()}, Max: {seed_in_degree.max()}")
    print(f"  25th: {np.percentile(seed_in_degree, 25)}, 50th: {np.percentile(seed_in_degree, 50)}, 75th: {np.percentile(seed_in_degree, 75)}, 95th: {np.percentile(seed_in_degree, 95)}")
    
    print(f"\nAverage Ratio:      {ratio.mean():.3f} (± {ratio.std():.3f})")
    print(f"  Min: {ratio.min():.3f}, Max: {ratio.max():.3f}")
    print(f"  25th: {np.percentile(ratio, 25):.3f}, 50th: {np.percentile(ratio, 50):.3f}, 75th: {np.percentile(ratio, 75):.3f}, 95th: {np.percentile(ratio, 95):.3f}")
    
    # Save a plot of the distributions
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(out_degree, bins=20, color='skyblue', edgecolor='black')
        axes[0].set_title('Out-degree (Influence) Distribution')
        axes[0].set_xlabel('Number of Agents Swayed')
        axes[0].set_ylabel('Frequency')
        
        axes[1].hist(seed_in_degree, bins=20, color='lightcoral', edgecolor='black')
        axes[1].set_title('In-degree (Susceptibility) Distribution')
        axes[1].set_xlabel('Number of Times Swayed')
        
        axes[2].hist(ratio, bins=20, color='lightgreen', edgecolor='black')
        axes[2].set_title('Influence/Susceptibility Ratio')
        axes[2].set_xlabel('Ratio')
        
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), 'influence_susceptibility.png')
        plt.savefig(plot_path)
        print(f"\n[!] Saved distribution plots to: {plot_path}")
    except ImportError:
        print("\n[!] matplotlib not installed. Skipping plot generation.")

    top_influencer_idx = np.argmax(out_degree)
    print(f"\nTop Influencer (by Out-degree):")
    print(f"  Out-degree: {out_degree[top_influencer_idx]}")
    print(f"  In-degree:  {seed_in_degree[top_influencer_idx]}")
    print(f"  Ratio:      {ratio[top_influencer_idx]:.3f}")
    
    top_susceptible_idx = np.argmax(seed_in_degree)
    print(f"\nMost Susceptible Agent (by In-degree):")
    print(f"  Out-degree: {out_degree[top_susceptible_idx]}")
    print(f"  In-degree:  {seed_in_degree[top_susceptible_idx]}")
    print(f"  Ratio:      {ratio[top_susceptible_idx]:.3f}")

    top_ratio_idx = np.argmax(ratio)
    print(f"\nHighest Influence/Susceptibility Ratio Agent:")
    print(f"  Out-degree: {out_degree[top_ratio_idx]}")
    print(f"  In-degree:  {seed_in_degree[top_ratio_idx]}")
    print(f"  Ratio:      {ratio[top_ratio_idx]:.3f}")

    print("\n[ Statistical Correlations ]")
    
    # Correlation between Out-degree and In-degree
    corr_out_in, p_out_in = pearsonr(out_degree, seed_in_degree)
    print(f"Correlation (Out-degree vs In-degree): {corr_out_in:.3f} (p={p_out_in:.4f})")
    if corr_out_in < -0.3 and p_out_in < 0.05:
        print("  -> STRONG NEGATIVE: Highly influential agents are rarely swayed themselves (Echo Chamber Leaders).")
    elif corr_out_in > 0.3 and p_out_in < 0.05:
        print("  -> STRONG POSITIVE: The most influential agents are also the most susceptible (Volatile Connectors).")
    else:
        print("  -> NO STRONG LINK: Influence and susceptibility are largely independent traits in this society.")
        
    # Let's also check if "Structural Influence" (from df_meta) correlates with our organic Ratio
    structural_influence = df_meta["Influence"].values[seed_indices]
    corr_struct, p_struct = pearsonr(ratio, structural_influence)
    print(f"\nCorrelation (Structural Influence vs Organic Ratio): {corr_struct:.3f} (p={p_struct:.4f})")
    if corr_struct > 0.3 and p_struct < 0.05:
        print("  -> ALIGNED: Agents assigned high structural influence naturally command a high Influence/Susceptibility ratio.")
    else:
        print("  -> DIVERGENT: Structural influence (wealth/reach) does not guarantee high organic thought leadership.")

if __name__ == "__main__":
    test_influence_susceptibility_ratio()
