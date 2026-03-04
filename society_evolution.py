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
        A fraction of wealth is passed on, rest lost or consumed.
        """
        inherit_frac = self.config.inheritance_fraction  # e.g., 0.7
        noise_std = self.config.inheritance_noise_std  # e.g., 0.05

        parent_wealth = self.exposures[:, self.wealth_idx]

        inherited_wealth = parent_wealth * inherit_frac
        noise = torch.randn(self.num_agents) * noise_std * parent_wealth.mean()

        new_wealth = inherited_wealth + noise
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

    def reassign_roles(self):

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

        # Define 5 structural roles
        role_centers = torch.tensor([0.1, 0.3, 0.5, 0.75, 0.95])

        # Compute fitness = closeness to each role center
        fitness = -((power_score.unsqueeze(1) - role_centers) ** 2)

        # Softmax
        probs = torch.softmax(fitness / self.config.role_temperature, dim=1)

        elite_mask = wealth_norm < self.config.elite_wealth_threshold
        probs[elite_mask, 4] = 0
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Sample new role
        new_roles_indices = torch.multinomial(probs, 1).squeeze().numpy()

        # Map indices to generic class names since original roles are removed
        role_map = {0: "Underclass", 1: "Working Class", 2: "Middle Class", 3: "Upper Middle", 4: "Elite"}
        mapped_roles = [role_map.get(idx, "Unknown") for idx in new_roles_indices]

        self.metadata["Role"] = mapped_roles

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

            if self.config.use_dynamic_roles:
                self.reassign_roles()

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
