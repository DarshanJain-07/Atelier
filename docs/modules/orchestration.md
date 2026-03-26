# Orchestration Layer (`main.py`)

The **Orchestration Layer** is a FastAPI-based backend that coordinates the various modules of the ATELIER framework. It handles concurrent execution, state management, and the advanced feedback loops that make the simulation autopoietic.

## 1. Concurrent Pipeline Architecture

To minimize latency, `main.py` utilizes Python's `asyncio` to run independent tasks in parallel during a simulation request:
1.  **Perception Task:** Calls the Gemini API to generate the 12D World Tensor.
2.  **Baseline Task:** Generates a statistical baseline for validation.
3.  **Society Task:** Loads or generates the agent population, including wealth and topology.

## 2. Advanced Caching (`SOCIETY_CACHE`)

Generating a population of 10,000+ agents with complex network topologies is computationally expensive. 
- **Mechanism:** An LRU (Least Recently Used) cache stores the generated PyTorch tensors and metadata.
- **Key:** The cache is keyed by `(seed, agent_count, mutation_temperature, use_power_law)`.
- **Benefit:** Consecutive runs on the same population (e.g., testing different events on the "same society") are near-instant.

## 3. Social Filtering & Slicing

ATELIER allows researchers to analyze reactions within specific sub-populations (e.g., "Elite" vs. "Working Class").
- **Metadata Filtering:** Agents are filtered based on their `Class` attribute.
- **Sparse Adjacency Reslicing:** When a sub-population is selected, the framework performs a complex "subset adjacency" operation. This remaps the sparse indices to preserve the local network structure *only* among the selected agents, allowing for accurate sub-group physics.

## 4. The 2-Pass Filter Bubble (Algorithmic Amplification)

This feature simulates how digital platforms amplify polarizing content to drive engagement.
- **Pass 1 (A/B Test):** The simulation is run on a small sample (e.g., 10%) of the population.
- **Identification:** The engine identifies which dimensions caused the highest "Engagement Weighted Attention".
- **Mutation:** The global World Tensor is mutated to exaggerate these dimensions (e.g., amplifying a minor threat into a crisis).
- **Pass 2 (Broadcast):** The entire society is exposed to this "algorithmically curated" version of the event.

## 5. Autopoietic Feedback Loop

The simulation is **autopoietic**—it can generate its own subsequent events without further user input.
1.  **Trigger:** If the Physics Engine detects that enough agents have crossed the action threshold (e.g., a "Populist Uprising"), it generates a new 12D `action_vector`.
2.  **Feedback:** `main.py` immediately feeds this vector back into the **Cognitive Engine**.
3.  **Secondary Cascade:** The society reacts to the *reaction* (e.g., how the rest of the population feels about a sudden protest), creating a multi-stage longitudinal simulation.

## 6. Memory Consolidation

Once the simulation ticks are complete, the framework performs **2-Stage Memory Consolidation**:
- **Immediate Imprint:** The `Context Vector` is added to the agent's memory.
- **Social Rehearsal:** The decay rate is adjusted based on the event's global salience (a combination of viral intensity and acting ratio). This updated memory persists for the next simulation request on the same society.
