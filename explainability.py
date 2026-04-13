from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from schema import DIMENSIONS, EMOTION_LABELS


class ExplainabilityEngine:
    """
    Translates complex tensor mathematics and simulation states into
    human-readable, qualitative explanations, drawing from physics,
    cognitive, and evolutionary layers.
    """

    def __init__(self):
        # Thresholds for qualitative descriptions
        self.high_polarization_threshold = 0.5
        self.high_entropy_threshold = 1.8
        self.high_bimodality_threshold = 0.3
        self.high_elite_divergence_threshold = 0.4
        self.high_virality_threshold = 3.0
        self.high_gini_threshold = 0.6

    def _get_dominant_emotion_from_vector(self, vector: List[float]) -> str:
        """Returns the dominant emotion label from an 8D emotion vector."""
        if not vector:
            return "Neutral"
        idx = np.argmax(vector)
        return EMOTION_LABELS[idx]

    def _generate_shift_story(self, social_state: Dict[str, Any]) -> str:
        """Explains the difference between objective, viral, and elite consensus."""
        objective_emotion = self._get_dominant_emotion_from_vector(
            social_state["objective_center"]
        )
        viral_emotion = self._get_dominant_emotion_from_vector(
            social_state["viral_center"]
        )
        elite_emotion = self._get_dominant_emotion_from_vector(
            social_state["elite_center"]
        )

        elite_div = social_state.get("elite_divergence") or 0.0

        story_parts = []

        # 1. Compare Objective vs Viral
        if objective_emotion == viral_emotion:
            story_parts.append(
                f"The society reached a broad consensus. The final emotion of **{viral_emotion}** "
                f"was felt evenly across the population and accurately reflects the objective center."
            )
        else:
            story_parts.append(
                f"While the average person leaned towards **{objective_emotion}**, a highly engaged "
                f"and vocal group amplified **{viral_emotion}**, causing it to go viral and dominate the discourse."
            )

        # 2. Add Elite Context
        if elite_div > self.high_elite_divergence_threshold:
            if elite_emotion != viral_emotion:
                story_parts.append(
                    f"There was a sharp disconnect across class lines. Influential elites felt **{elite_emotion}**, "
                    f"while the broader public narrative was driven by **{viral_emotion}**."
                )
            else:
                story_parts.append(
                    f"Despite significant differences in underlying emotional intensity between classes, "
                    f"both elites and the general public ultimately converged on **{elite_emotion}**."
                )

        return " ".join(story_parts)

    def _generate_narrative_competition(
        self,
        narrative_frame: str | None = None,
        backlash_potential: float | None = None,
        backlash_diagnostics: Dict[str, Any] | None = None,
        official_world_tensor: torch.Tensor | None = None,
        skeptical_world_tensor: torch.Tensor | None = None,
    ) -> str:
        """Explains whether a backlash A/B test selected the official or skeptical narrative."""
        if backlash_diagnostics is None:
            return ""

        sample_size = int(backlash_diagnostics.get("sample_size") or 0)
        skeptical_count = int(backlash_diagnostics.get("skeptical_count") or 0)
        official_count = int(backlash_diagnostics.get("official_count") or 0)
        skeptical_energy = float(backlash_diagnostics.get("skeptical_energy") or 0.0)
        official_energy = float(backlash_diagnostics.get("official_energy") or 0.0)
        selected_frame = narrative_frame or backlash_diagnostics.get("chosen_frame") or "official"
        potential = float(
            backlash_potential
            if backlash_potential is not None
            else backlash_diagnostics.get("backlash_potential") or 0.0
        )

        if selected_frame == "skeptical":
            base = (
                f"A vanguard sample of **{sample_size}** agents internally A/B tested the event, and the "
                f"**skeptical backlash frame won**. Skeptical agents generated **{skeptical_energy:.2f}** "
                f"engagement energy versus **{official_energy:.2f}** for the official frame, so the broader "
                f"population was exposed to the cynical narrative."
            )
        else:
            base = (
                f"A vanguard sample of **{sample_size}** agents internally A/B tested the event, but the "
                f"**official frame held**. Official-facing agents generated **{official_energy:.2f}** "
                f"engagement energy versus **{skeptical_energy:.2f}** for the backlash frame, so the wider "
                f"population stayed on the intended narrative."
            )

        routing = (
            f" The sample split into **{skeptical_count}** skeptical-prone agents and "
            f"**{official_count}** conformist/trusting agents, with an overall backlash potential of "
            f"**{potential:.2f}**."
        )

        frame_gap = ""
        if official_world_tensor is not None and skeptical_world_tensor is not None:
            off = official_world_tensor.squeeze().detach().cpu()
            skp = skeptical_world_tensor.squeeze().detach().cpu()
            gap = torch.abs(off - skp)
            top_gap = torch.topk(gap, k=min(2, gap.numel())).indices.tolist()
            if top_gap:
                labels = [DIMENSIONS[idx] for idx in top_gap if gap[idx].item() > 0.05]
                if labels:
                    frame_gap = (
                        " The sharpest disagreement between frames centered on **"
                        + "** and **".join(labels)
                        + "**."
                    )

        return f"{base}{routing}{frame_gap}"

    def _generate_tug_of_war(self, social_state: Dict[str, Any]) -> str:
        """Explains the polarization, fragmentation, and sentiment of the society."""
        polarization = social_state.get("polarization") or 0.0
        entropy = social_state.get("entropy") or 0.0
        bimodality = social_state.get("bimodality") or 0.0
        sentiment = social_state.get("sentiment_valence") or 0.0

        if (
            polarization > self.high_polarization_threshold
            and bimodality > self.high_bimodality_threshold
        ):
            base = "The society is **deeply divided**, showing strong evidence of isolated **echo chambers**."
        elif entropy > self.high_entropy_threshold:
            base = "The society is **fragmented**."
        elif polarization < (self.high_polarization_threshold / 2):
            base = "The society is **highly unified**."
        else:
            base = "The society shows a **moderate mix** of opinions."

        if "divided" in base:
            desc = "There is no single consensus; instead, the population is fiercely split between opposing viewpoints."
        elif "fragmented" in base:
            desc = "Reactions are scattered across the emotional spectrum with no clear, unified response."
        elif "unified" in base:
            desc = (
                "Most individuals share a very similar emotional reaction to the event."
            )
        else:
            desc = "While there is a dominant narrative, a noticeable variety of secondary emotions exists."

        sentiment_desc = ""
        if sentiment > 0.3:
            sentiment_desc = "The overall mood leans decidedly **positive**."
        elif sentiment < -0.3:
            sentiment_desc = "The overall mood is distinctly **negative**."
        else:
            sentiment_desc = "The overall mood is relatively **neutral** or mixed."

        return f"{base} {desc} {sentiment_desc}"

    def _generate_cognitive_drivers(self, attention_weights: torch.Tensor) -> str:
        """Analyzes the attention weights to determine what dimensions agents cared about most."""
        if attention_weights is None or attention_weights.numel() == 0:
            return "Cognitive attention data is unavailable."

        # Average attention across all agents
        mean_attention = attention_weights.mean(dim=0).cpu().numpy()
        top_idx = np.argmax(mean_attention)
        top_dimension = DIMENSIONS[top_idx]
        top_percentage = (mean_attention[top_idx] / np.sum(mean_attention)) * 100

        desc = (
            f"The population's attention was primarily captured by the **{top_dimension}** aspect of the event, "
            f"which drove **{top_percentage:.1f}%** of their overall cognitive focus."
        )

        # Find secondary driver if prominent
        sorted_indices = np.argsort(mean_attention)[::-1]
        second_idx = sorted_indices[1]
        second_percentage = (mean_attention[second_idx] / np.sum(mean_attention)) * 100
        if second_percentage > 15.0:
            desc += f" A secondary focus emerged around **{DIMENSIONS[second_idx]}** ({second_percentage:.1f}%)."

        return desc

    def _generate_viral_dynamics(self, social_state: Dict[str, Any]) -> str:
        """Translates the nonlinear outrage multiplier into a virality score proxy (R0 equivalent)."""
        mean_multiplier = social_state.get("mean_outrage_multiplier") or 1.0
        max_multiplier = social_state.get("max_outrage_multiplier") or 1.0

        desc = ""
        if mean_multiplier > self.high_virality_threshold:
            desc = (
                f"This event was **highly viral**. The algorithm heavily boosted outlier emotional responses "
                f"(Average Outrage Multiplier: **{mean_multiplier:.1f}x**), simulating rapid social contagion akin to an R0 > 2.5."
            )
        elif mean_multiplier > 1.5:
            desc = f"This event experienced **moderate algorithmic amplification** (Average Outrage Multiplier: **{mean_multiplier:.1f}x**)."
        else:
            desc = (
                f"The emotional response propagated naturally without significant algorithmic viral boosting "
                f"(Average Outrage Multiplier: **{mean_multiplier:.1f}x**)."
            )

        if max_multiplier > mean_multiplier * 3 and max_multiplier > 5.0:
            desc += f" However, extreme outlier agents generated localized viral storms with up to **{max_multiplier:.1f}x** amplification."

        return desc

    def _compute_gini(self, array: np.ndarray) -> float:
        """Calculates the Gini coefficient of a numpy array."""
        array = np.sort(array)
        n = array.shape[0]
        if n == 0 or np.sum(array) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    def _generate_societal_structure(self, metadata: pd.DataFrame) -> str:
        """Explains baseline structural inequality and its role."""
        if "Influence" not in metadata.columns:
            return "Societal influence metrics unavailable."

        influence = metadata["Influence"].to_numpy()
        gini = self._compute_gini(influence)

        # Calculate what % of influence the top 10% holds
        sorted_influence = np.sort(influence)
        top_10_percent_idx = int(len(sorted_influence) * 0.9)
        top_10_sum = np.sum(sorted_influence[top_10_percent_idx:])
        total_sum = np.sum(sorted_influence)
        top_10_share = (top_10_sum / total_sum) * 100 if total_sum > 0 else 0

        desc = f"Influence inequality Gini coefficient is **{gini:.2f}**. "

        if gini > self.high_gini_threshold:
            desc += f"Society is structurally hierarchical. The top 10% of agents control **{top_10_share:.1f}%** of the societal narrative power, heavily skewing the objective consensus."
        elif gini > 0.4:
            desc += f"Society exhibits moderate influence distribution. The top 10% control **{top_10_share:.1f}%** of the narrative."
        else:
            desc += "Society is relatively egalitarian. Influence is widely distributed among the population."

        return desc

    def _generate_demographic_summary(
        self,
        metadata: pd.DataFrame,
        personalities: torch.Tensor,
        final_emotions: torch.Tensor,
    ) -> List[Dict[str, str]]:
        """
        Groups agents into archetypes (e.g., Secure Elites, Vulnerable Population, Apathetic)
        and explains their dominant reactions.
        """
        N = len(metadata)
        if N == 0:
            return []

        # Extract features
        influence = metadata["Influence"].to_numpy()
        neuroticism = personalities[:, 4].cpu().numpy()

        # Find medians to define high/low
        med_influence = np.median(influence)
        med_neuro = np.median(neuroticism)

        # Get emotion indices
        emotion_indices = torch.argmax(final_emotions, dim=1).cpu().numpy()

        archetypes = []

        # 1. Secure Elites (High Influence, Low/Med Neuroticism)
        secure_mask = (influence >= med_influence) & (neuroticism < med_neuro)
        if np.any(secure_mask):
            secure_emotions = emotion_indices[secure_mask]
            dom_idx = np.bincount(secure_emotions).argmax()
            dom_emotion = EMOTION_LABELS[int(dom_idx)]
            archetypes.append(
                {
                    "name": "Secure Elites",
                    "description": f"Highly influential but emotionally stable individuals predominantly felt **{dom_emotion}**.",
                }
            )

        # 2. Vulnerable Population (Low Influence, High Neuroticism)
        vuln_mask = (influence < med_influence) & (neuroticism >= med_neuro)
        if np.any(vuln_mask):
            vuln_emotions = emotion_indices[vuln_mask]
            dom_idx = np.bincount(vuln_emotions).argmax()
            dom_emotion = EMOTION_LABELS[int(dom_idx)]
            archetypes.append(
                {
                    "name": "Vulnerable Population",
                    "description": f"Individuals with less influence and higher anxiety were driven by **{dom_emotion}**.",
                }
            )

        # 3. The Anxious Elites (High Influence, High Neuroticism)
        anx_elite_mask = (influence >= med_influence) & (neuroticism >= med_neuro)
        if np.any(anx_elite_mask):
            anx_emotions = emotion_indices[anx_elite_mask]
            dom_idx = np.bincount(anx_emotions).argmax()
            dom_emotion = EMOTION_LABELS[int(dom_idx)]
            archetypes.append(
                {
                    "name": "Anxious Elites",
                    "description": f"Influential but highly reactive individuals predominantly felt **{dom_emotion}**.",
                }
            )

        # 4. The Stoic Majority (Low Influence, Low Neuroticism)
        stoic_mask = (influence < med_influence) & (neuroticism < med_neuro)
        if np.any(stoic_mask):
            stoic_emotions = emotion_indices[stoic_mask]
            dom_idx = np.bincount(stoic_emotions).argmax()
            dom_emotion = EMOTION_LABELS[int(dom_idx)]
            archetypes.append(
                {
                    "name": "Stoic Public",
                    "description": f"Everyday citizens with lower emotional volatility largely reacted with **{dom_emotion}**.",
                }
            )

        return archetypes

    def _generate_endogenous_event_explanation(self, social_state: Dict[str, Any]) -> str:
        """Explains if the societal tension triggered an autopoietic macro-event."""
        action_name = social_state.get("action_name") or ""
        if not action_name:
            return "Society remained stable enough that no macro-level endogenous events were triggered."
        
        polarization = social_state.get("polarization") or 0.0
        elite_divergence = social_state.get("elite_divergence") or 0.0
        
        reasons = []
        if elite_divergence > self.high_elite_divergence_threshold:
            reasons.append("extreme elite divergence")
        if polarization > self.high_polarization_threshold:
            reasons.append("high structural polarization")
            
        reason_str = " and ".join(reasons) if reasons else "high societal tension"
        
        return (
            f"Autopoietic Trigger: The simulation generated an endogenous event "
            f"{action_name}, due to {reason_str}, combined with **sufficient individual Action Potential** across the populace, the system reached a breaking point, "
            f"automatically feeding this new macro-action back into the society."
        )

    def generate_explanation(
        self,
        social_state: Dict[str, Any],
        metadata: pd.DataFrame,
        personalities: torch.Tensor,
        final_emotions: torch.Tensor,
        attention_weights: torch.Tensor,
        narrative_frame: str | None = None,
        backlash_potential: float | None = None,
        backlash_diagnostics: Dict[str, Any] | None = None,
        official_world_tensor: torch.Tensor | None = None,
        skeptical_world_tensor: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        """
        Main entry point to generate the full multi-layered explainability package.
        """
        narrative_competition = self._generate_narrative_competition(
            narrative_frame=narrative_frame,
            backlash_potential=backlash_potential,
            backlash_diagnostics=backlash_diagnostics,
            official_world_tensor=official_world_tensor,
            skeptical_world_tensor=skeptical_world_tensor,
        )
        shift_story = self._generate_shift_story(social_state)
        if narrative_competition:
            shift_story = f"{narrative_competition} {shift_story}"

        return {
            "shift_story": shift_story,
            "narrative_competition": narrative_competition,
            "tug_of_war": self._generate_tug_of_war(social_state),
            "cognitive_drivers": self._generate_cognitive_drivers(attention_weights),
            "viral_dynamics": self._generate_viral_dynamics(social_state),
            "societal_structure": self._generate_societal_structure(metadata),
            "demographics": self._generate_demographic_summary(
                metadata, personalities, final_emotions
            ),
            "endogenous_events": self._generate_endogenous_event_explanation(social_state),
        }
