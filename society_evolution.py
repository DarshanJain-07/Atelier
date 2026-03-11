import numpy as np
import torch


class SocietyEvolution:
    def __init__(
        self, config, initial_metadata, initial_exposures, initial_personalities
    ):
        self.config = config
        self.metadata = initial_metadata.copy()
        self.exposures = initial_exposures.clone()
        self.personalities = initial_personalities.clone()

        self.num_agents = len(self.metadata)
        self.wealth_idx = (
            self.config.wealth_dim_idx
            if hasattr(self.config, "wealth_dim_idx")
            else None
        )
        if self.wealth_idx is None:
            raise ValueError(
                "Config must provide 'wealth_dim_idx' for wealth column index"
            )

        # Initialize influence from metadata (or calculate if needed)
        self.influence = torch.tensor(
            self.metadata["Influence"].values, dtype=torch.float32
        )

        # Store history if requested
        self.history = {
            "wealth": [self.exposures[:, self.wealth_idx].clone()],
            "influence": [self.influence.clone()],
        }

    def apply_inheritance(self):
        """
        Simulate intergenerational inheritance with some decay and noise.

        Refactored Logic:
        - 'inheritance_fraction' acts as the amount PASSED DOWN (e.g., 0.7).
        - The remaining (0.3) is NOT destroyed. It is taxed/lost and then
          REDISTRIBUTED back to the population (simulating public services/infrastructure).
        - This prevents the economy from shrinking to zero over time.
        """
        inherit_frac = self.config.inheritance_fraction  # e.g., 0.7
        noise_std = self.config.inheritance_noise_std  # e.g., 0.05

        parent_wealth = self.exposures[:, self.wealth_idx]

        # 1. Tax / Loss Phase
        inherited_wealth = parent_wealth * inherit_frac

        # Calculate total 'tax' collected
        total_tax = (parent_wealth - inherited_wealth).sum()

        # 2. Redistribution Phase (Uniform Basic Income / Public Goods)
        # Everyone gets an equal share of the taxed wealth
        redistribution_per_capita = total_tax / self.num_agents

        # 3. Apply changes with noise
        noise = torch.randn(self.num_agents) * noise_std * parent_wealth.mean()

        new_wealth = inherited_wealth + redistribution_per_capita + noise
        new_wealth = torch.clamp(new_wealth, min=0.0)  # No negative wealth

        self.exposures[:, self.wealth_idx] = new_wealth

    def apply_reinvestment(self):
        """
        Capital reinvestment cycles: wealth grows by a factor influenced by influence score
        and stochastic returns.
        """
        base_return = self.config.base_return_rate  # e.g., 0.03
        influence_factor = self.config.influence_reinvestment_factor  # e.g., 0.1
        noise_std = self.config.reinvestment_noise_std  # e.g., 0.02

        returns = base_return + influence_factor * (
            self.influence / self.influence.mean()
        )
        noise = torch.randn(self.num_agents) * noise_std
        returns = returns + noise
        returns = torch.clamp(returns, min=-0.2, max=0.5)  # realistic return bounds

        self.exposures[:, self.wealth_idx] *= 1 + returns

    def apply_economic_shocks(self, generation):
        """
        Simulate economic shocks as rare but impactful negative or positive multipliers.
        Frequency and magnitude can be configured.
        """
        shock_freq = self.config.shock_frequency  # e.g., 0.1 per generation
        shock_magnitude = self.config.shock_magnitude  # e.g., 0.2 (20%)

        # Randomly determine if shock occurs this generation
        if np.random.rand() < shock_freq:
            shock_impact = 1 - shock_magnitude * np.random.uniform(
                0.5, 1.0
            )  # Reduce wealth
            print(
                f"Applying economic shock at generation {generation}, multiplier {shock_impact:.2f}"
            )
            self.exposures[:, self.wealth_idx] *= shock_impact

    def apply_mobility(self):
        """
        Stochastic social mobility: randomly reshuffle a small % of agents' influence and wealth,
        simulating social moves up or down.
        """
        mobility_rate = self.config.mobility_rate  # e.g., 0.05

        n_movers = int(self.num_agents * mobility_rate)
        indices = np.random.choice(self.num_agents, n_movers, replace=False)

        # Shuffle influence and wealth among movers
        shuffled_indices = np.random.permutation(indices)

        new_influence = self.influence.clone()
        new_wealth = self.exposures[:, self.wealth_idx].clone()

        new_influence[indices] = self.influence[shuffled_indices]
        new_wealth[indices] = self.exposures[shuffled_indices, self.wealth_idx]

        self.influence = new_influence
        self.exposures[:, self.wealth_idx] = new_wealth

    def apply_ideological_drift(self):
        """
        Simulate generational ideological drift. Agents' traits drift slightly towards
        the societal mean, with some random generational noise, mimicking cultural shifts.
        Occasionally (e.g. 5% chance), society drifts toward the Elite's mean instead of the global mean (Hegemony).
        Also includes "Repulsion" where alienated agents actively move away from the mainstream.
        """
        if not getattr(self.config, "use_ideological_drift", False):
            return

        drift_rate = getattr(self.config, "ideological_drift_rate", 0.05)
        noise_std = getattr(self.config, "ideological_drift_noise", 0.02)
        elite_chance = getattr(self.config, "elite_influence_drift_chance", 0.05)

        use_repulsion = getattr(self.config, "use_ideological_repulsion", False)
        repulsion_threshold = getattr(self.config, "repulsion_threshold", 0.0)
        repulsion_rate = getattr(self.config, "repulsion_rate", 0.02)

        # We only drift non-wealth traits
        non_wealth_mask = torch.ones(self.exposures.shape[1], dtype=torch.bool)
        if self.wealth_idx is not None:
            non_wealth_mask[self.wealth_idx] = False

        # Decide what the "cultural attractor" is for this generation
        if np.random.rand() < elite_chance and "Class" in self.metadata:
            # Cultural Hegemony: Society drifts toward the Elite class
            elite_mask = (self.metadata["Class"] == "Elite").values
            if elite_mask.sum() > 1:
                target_exposures = self.exposures[torch.from_numpy(elite_mask)]
                target_mean = target_exposures.mean(dim=0)
                target_var = target_exposures.var(dim=0, unbiased=False)
            elif elite_mask.sum() == 1:
                target_exposures = self.exposures[torch.from_numpy(elite_mask)]
                target_mean = target_exposures.mean(dim=0)
                target_var = torch.zeros_like(target_mean)
            else:
                target_mean = self.exposures.mean(dim=0)
                target_var = self.exposures.var(dim=0, unbiased=False)
        else:
            # Normal cultural consensus
            target_mean = self.exposures.mean(dim=0)
            target_var = self.exposures.var(dim=0, unbiased=False)

        # Prevent zero variance issues
        target_var = torch.clamp(target_var, min=1e-6)

        # Bayesian Belief Updating:
        # If the target (evidence) is highly polarized (high variance), agents trust it less.
        # If there is strong consensus (low variance), agents are pulled more strongly.
        prior_var = getattr(self.config, "bayesian_prior_variance", 0.1)
        
        # Calculate dynamic Bayesian update rate (Kalman Gain style)
        # Shape will be (D,) allowing per-dimension belief updating based on consensus.
        bayesian_update_rate = prior_var / (prior_var + target_var)
        dynamic_drift_rate = drift_rate * bayesian_update_rate

        # Calculate difference from target
        diff_from_target = target_mean - self.exposures

        # Calculate 'distance' or 'alienation' of each agent from the target mean
        # Using cosine similarity to measure if they are conceptually aligned
        if use_repulsion:
            # Calculate cosine similarity manually for tensors
            # (N, D) and (D,)
            target_mean_norm = target_mean[non_wealth_mask] / (
                torch.norm(target_mean[non_wealth_mask]) + 1e-8
            )
            agent_norms = self.exposures[:, non_wealth_mask] / (
                torch.norm(self.exposures[:, non_wealth_mask], dim=1, keepdim=True)
                + 1e-8
            )
            alignment = torch.matmul(agent_norms, target_mean_norm)

            # Agents with alignment < threshold are "alienated" and experience repulsion instead of attraction
            is_alienated = (alignment < repulsion_threshold).unsqueeze(1)

            # Attracted agents move toward target (+), Alienated move away (-)
            # Repulsion rate usually slightly smaller than drift to avoid explosions
            drift_direction = torch.where(
                is_alienated,
                -diff_from_target * repulsion_rate,
                diff_from_target * dynamic_drift_rate,
            )
        else:
            drift_direction = diff_from_target * dynamic_drift_rate

        # Add random noise
        noise = torch.randn_like(self.exposures) * noise_std

        # 1. Apply raw drift + noise
        new_exposures = self.exposures[:, non_wealth_mask].clone()
        new_exposures += drift_direction[:, non_wealth_mask] + noise[:, non_wealth_mask]

        # 2. Standardize to prevent variance collapse
        # We only do this if repulsion is OFF, because repulsion is SUPPOSED to increase variance (fracturing)
        if not use_repulsion:
            current_std = self.exposures[:, non_wealth_mask].std(dim=0, keepdim=True)
            new_mean = new_exposures.mean(dim=0, keepdim=True)
            new_std = new_exposures.std(dim=0, keepdim=True)
            new_std = torch.clamp(new_std, min=1e-6)
            new_exposures = (
                (new_exposures - new_mean) / new_std
            ) * current_std + new_mean

        self.exposures[:, non_wealth_mask] = new_exposures

        # Clamp back to [-1, 1]
        self.exposures[:, non_wealth_mask] = torch.clamp(
            self.exposures[:, non_wealth_mask], -1.0, 1.0
        )

    def reassign_classes(self):

        wealth = self.exposures[:, self.wealth_idx]

        # Normalize to percentiles (0–1)
        wealth_rank = torch.argsort(torch.argsort(wealth))
        wealth_norm = wealth_rank.float() / (self.num_agents - 1)

        influence_rank = torch.argsort(torch.argsort(self.influence))
        influence_norm = influence_rank.float() / (self.num_agents - 1)

        # Simple power score (one scalar)
        power_score = (
            0.5 * wealth_norm
            + 0.4 * influence_norm
            + 0.1 * self.personalities.mean(dim=1)
        )

        # Define 5 structural classes
        class_centers = torch.tensor([0.1, 0.3, 0.5, 0.75, 0.95])

        # Compute fitness = closeness to each class center
        fitness = -((power_score.unsqueeze(1) - class_centers) ** 2)

        # Softmax
        probs = torch.softmax(fitness / self.config.class_temperature, dim=1)

        elite_mask = wealth_norm < self.config.elite_wealth_threshold
        probs[elite_mask, 4] = 0
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Sample new class
        new_classes_indices = torch.multinomial(probs, 1).squeeze().numpy()

        # Map indices to generic class names
        class_map = {
            0: "Underclass",
            1: "Working Class",
            2: "Middle Class",
            3: "Upper Middle",
            4: "Elite",
        }
        mapped_classes = [class_map.get(idx, "Unknown") for idx in new_classes_indices]

        self.metadata["Class"] = mapped_classes

    def evolve(self):
        """
        Run the full evolution cycle for the configured number of generations
        """
        generations = self.config.evolution_generations

        for gen in range(1, generations + 1):
            self.apply_inheritance()
            self.apply_reinvestment()
            self.apply_economic_shocks(gen)
            self.apply_mobility()
            self.apply_ideological_drift()

            if self.config.use_dynamic_classes:
                self.reassign_classes()

            # Clamp wealth to avoid negative values or extreme outliers
            self.exposures[:, self.wealth_idx] = torch.clamp(
                self.exposures[:, self.wealth_idx], min=0.0, max=1e6
            )

            # Optionally record history
            if self.config.record_history:
                self.history["wealth"].append(
                    self.exposures[:, self.wealth_idx].clone()
                )
                self.history["influence"].append(self.influence.clone())

        # Update metadata influence for output consistency
        self.metadata["Influence"] = self.influence.numpy()

        return self.metadata, self.exposures, self.personalities
