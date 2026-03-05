const canvas = document.getElementById("agent-canvas");
const ctx = canvas.getContext("2d");
const container = document.querySelector(".canvas-wrapper");
const tooltip = document.getElementById("tooltip");
const sidebar = document.getElementById("config-sidebar");
const dossier = document.getElementById("dossier-card");

// --- CONSTANTS ---
const PALETTE = {
  Neutral: "#333333",
  Joy: "#FFD93D", // Yellow
  Trust: "#1F75FE", // Blue
  Surprise: "#F4A261", // Orange
  Anticipation: "#9B5DE5", // Purple
  Fear: "#F9F8F6", // White-grey
  Anger: "#E63946", // Red
  Disgust: "#2A9D8F", // Green
  Sadness: "#457B9D", // Dark Blue
};

// Map 8 Emotions to specific angles on the circle
// Adjusted for better visual separation
const EMOTION_ANGLES = {
  Fear: -Math.PI / 2, // Top
  Joy: -Math.PI / 4, // Top-Right
  Trust: 0, // Right
  Surprise: Math.PI / 4, // Bottom-Right
  Anticipation: Math.PI / 2, // Bottom
  Sadness: (3 * Math.PI) / 4, // Bottom-Left
  Disgust: Math.PI, // Left
  Anger: (-3 * Math.PI) / 4, // Top-Left
};

let agents = [];
let agentMetadata = [];
let clusterAggregates = {};
let simulationHistory = [];
let currentSessionIndex = -1;
let currentRunInSessionIndex = -1;
let selectedAgentIndex = null;
let width, height, centerX, centerY;
let clusterSpread = 1.0;

// --- BATCH STATE ---
let batchRuns = [
  {
    id: Date.now(),
    seed: 42,
    temperature: 0.7,
    emotion_temperature: 0.2,
    panic_threshold: -1.2,
    region: "All",
    role: "All",
    agent_count: 5000,
    use_distortion: true,
    use_pressure: true,
    use_maslow: true,
    use_power_law: false,
    cross_dim_interaction_strength: 0.3,
    threat_sensitivity_gain: 1.5,
    k_processing_tanh_gain: 1.5,
    relevance_importance_weight: 0.7,
    relevance_base_weight: 0.3,
    threat_amplifier_gain: 1.5,
    stress_neurotic_amplification: 1.5,
    stress_openness_reduction: 0.5,
    stress_extraversion_boost: 0.7,
    outrage_gain: 2.5,
    max_viral_multiplier: 10.0,
    saturation_midpoint: 1.5,
    distortion_max_noise: 0.4,
    distortion_neurotic_gain: 0.6,
    evolution_generations: 10,
    inheritance_fraction: 0.7,
    shock_frequency: 0.1,
    shock_magnitude: 0.2,
  },
];
let currentBatchResults = [];

const historyPanel = document.getElementById("history-panel");
const filmstrip = document.getElementById("run-filmstrip");
const filmstripToggleBtn = document.getElementById("filmstrip-toggle-btn");
let isFilmstripManuallyHidden = false;

// --- BATCH UI RENDERER ---
function renderBatchUI() {
  const container = document.getElementById("batch-container");
  container.innerHTML = "";

  batchRuns.forEach((run, index) => {
    const item = document.createElement("div");
    item.className = "batch-item";
    item.innerHTML = `
            <div class="batch-header">
                <span class="batch-title">EXPERIMENT ${index + 1}</span>
                ${batchRuns.length > 1 ? `<button class="batch-remove" onclick="removeRun(${run.id})">&times;</button>` : ""}
            </div>
            <div class="sidebar-grid">
                <div class="sidebar-field">
                    <label data-tooltip="Random seed for deterministic outcomes.">Seed</label>
                    <input type="number" class="sidebar-input" value="${run.seed}" onchange="updateRun(${run.id}, 'seed', this.value)">
                </div>
                <div class="sidebar-field">
                    <label data-tooltip="Controls the mutation rate. Higher values increase randomness.">Temp</label>
                    <input type="number" class="sidebar-input" step="0.1" min="0" max="1" value="${run.temperature}" onchange="updateRun(${run.id}, 'temperature', this.value)">
                </div>
            </div>
            <div class="sidebar-grid">
                <div class="sidebar-field sidebar-field-span-2">
                    <label data-tooltip="Filter agents by socioeconomic role.">Role</label>
                    <select class="sidebar-input select-ui-font" onchange="updateRun(${run.id}, 'role', this.value)">
                        <option value="All" ${run.role === "All" ? "selected" : ""}>All</option>
                        <option value="Underclass" ${run.role === "Underclass" ? "selected" : ""}>Underclass</option>
                        <option value="Working Class" ${run.role === "Working Class" ? "selected" : ""}>Working Class</option>
                        <option value="Middle Class" ${run.role === "Middle Class" ? "selected" : ""}>Middle Class</option>
                        <option value="Upper Middle" ${run.role === "Upper Middle" ? "selected" : ""}>Upper Middle</option>
                        <option value="Elite" ${run.role === "Elite" ? "selected" : ""}>Elite</option>
                    </select>
                </div>
            </div>
            <div class="sidebar-grid sidebar-grid-margin">
                <div class="sidebar-field sidebar-field-span-2">
                    <label data-tooltip="Total number of agents to simulate.">Population: <span id="batch-pop-val-${run.id}">${(run.agent_count / 1000).toFixed(1)}k</span></label>
                    <input type="range" min="1000" max="15000" step="1000" value="${run.agent_count}"
                        oninput="document.getElementById('batch-pop-val-${run.id}').textContent = (this.value/1000).toFixed(1) + 'k'; updateRun(${run.id}, 'agent_count', this.value)">
                </div>
            </div>
            <div class="batch-toggle-grid">
                <button class="batch-tog ${run.use_distortion ? "active" : ""}" onclick="this.classList.toggle('active'); updateRun(${run.id}, 'use_distortion', this.classList.contains('active'))">DIST</button>
                <button class="batch-tog ${run.use_pressure ? "active" : ""}" onclick="this.classList.toggle('active'); updateRun(${run.id}, 'use_pressure', this.classList.contains('active'))">TIME</button>
                <button class="batch-tog ${run.use_maslow ? "active" : ""}" onclick="this.classList.toggle('active'); updateRun(${run.id}, 'use_maslow', this.classList.contains('active'))">MSLW</button>
                <button class="batch-tog ${run.use_power_law ? "active" : ""}" onclick="this.classList.toggle('active'); updateRun(${run.id}, 'use_power_law', this.classList.contains('active'))">PWR</button>
            </div>
        `;
    container.appendChild(item);
  });
}

window.removeRun = (id) => {
  batchRuns = batchRuns.filter((r) => r.id !== id);
  renderBatchUI();
};

window.updateRun = (id, field, value) => {
  const run = batchRuns.find((r) => r.id === id);
  if (run) {
    if (typeof value === "boolean") {
      run[field] = value;
    } else {
      run[field] =
        field === "seed" ||
        field === "temperature" ||
        field === "emotion_temperature" ||
        field === "panic_threshold" ||
        field === "agent_count"
          ? parseFloat(value)
          : value;
    }
  }
};

document.getElementById("btn-add-run").addEventListener("click", () => {
  if (batchRuns.length >= 6) return;
  const last = batchRuns[batchRuns.length - 1];
  batchRuns.push({
    id: Date.now(),
    seed: Math.floor(Math.random() * 10000),
    temperature: last.temperature,
    emotion_temperature: last.emotion_temperature,
    panic_threshold: last.panic_threshold,
    region: last.region,
    role: last.role,
    agent_count: last.agent_count,
    use_distortion: last.use_distortion,
    use_pressure: last.use_pressure,
    use_maslow: last.use_maslow,
    use_power_law: last.use_power_law,
    cross_dim_interaction_strength: last.cross_dim_interaction_strength,
    threat_sensitivity_gain: last.threat_sensitivity_gain,
    k_processing_tanh_gain: last.k_processing_tanh_gain,
    relevance_importance_weight: last.relevance_importance_weight,
    relevance_base_weight: last.relevance_base_weight,
    threat_amplifier_gain: last.threat_amplifier_gain,
    stress_neurotic_amplification: last.stress_neurotic_amplification,
    stress_openness_reduction: last.stress_openness_reduction,
    stress_extraversion_boost: last.stress_extraversion_boost,
    outrage_gain: last.outrage_gain,
    max_viral_multiplier: last.max_viral_multiplier,
    saturation_midpoint: last.saturation_midpoint,
    distortion_max_noise: last.distortion_max_noise,
    distortion_neurotic_gain: last.distortion_neurotic_gain,
    evolution_generations: last.evolution_generations,
    inheritance_fraction: last.inheritance_fraction,
    shock_frequency: last.shock_frequency,
    shock_magnitude: last.shock_magnitude,
  });
  renderBatchUI();
});

function renderFilmstrip(results) {
  filmstrip.innerHTML = "";

  // Hide everything if single result
  if (!results || results.length <= 1) {
    filmstrip.classList.add("hidden");
    filmstripToggleBtn.classList.add("hidden");
    return;
  }

  // Multiple results available, show the toggle button
  filmstripToggleBtn.classList.remove("hidden");

  results.forEach((res, idx) => {
    const card = document.createElement("div");
    card.className = `filmstrip-card ${idx === 0 ? "active" : ""}`;
    card.onclick = () => selectRun(idx);

    card.innerHTML = `
            <div class="filmstrip-header">
                <span class="filmstrip-tag">RUN ${idx + 1}</span>
                <div class="emotion-indicator" style="background: ${PALETTE[res.dominant_emotion] || "#333"}"></div>
            </div>
            <div class="filmstrip-emotion" style="color: ${PALETTE[res.dominant_emotion]}">${res.dominant_emotion.toUpperCase()}</div>
            <div class="filmstrip-stats">
                <span>POL: ${res.polarization}</span>
                <span>W-DIST: ${res.divergence.toFixed(2)} | KL: ${res.kl_divergence.toFixed(2)}</span>
            </div>
        `;
    filmstrip.appendChild(card);
  });

  // Apply manual toggle state
  if (isFilmstripManuallyHidden) {
    filmstrip.classList.add("hidden");
    filmstripToggleBtn.style.color = "var(--text-secondary)";
    filmstripToggleBtn.style.borderColor = "var(--border)";
  } else {
    filmstrip.classList.remove("hidden");
    filmstripToggleBtn.style.color = "var(--accent)";
    filmstripToggleBtn.style.borderColor = "var(--accent)";
  }
}

function toggleFilmstrip() {
  isFilmstripManuallyHidden = !isFilmstripManuallyHidden;

  if (isFilmstripManuallyHidden) {
    filmstrip.classList.add("hidden");
    filmstripToggleBtn.style.color = "var(--text-secondary)";
    filmstripToggleBtn.style.borderColor = "var(--border)";
  } else {
    filmstrip.classList.remove("hidden");
    filmstripToggleBtn.style.color = "var(--accent)";
    filmstripToggleBtn.style.borderColor = "var(--accent)";
  }
}

function selectRun(index) {
  const result = currentBatchResults[index];
  if (!result) return;

  // Update filmstrip UI
  document.querySelectorAll(".filmstrip-card").forEach((c, i) => {
    c.classList.toggle("active", i === index);
  });

  displayResult(result);
}

function displayResult(data) {
  // 3. Update UI Metrics
  const sentimentEl = document.getElementById("val-sentiment");
  sentimentEl.textContent = data.dominant_emotion.toUpperCase();
  sentimentEl.style.color = PALETTE[data.dominant_emotion] || "#ffffff";

  const polVal = parseFloat(data.polarization);
  const polEl = document.getElementById("val-polarization");
  polEl.textContent = isNaN(polVal) ? "--" : polVal.toFixed(3);

  if (!isNaN(polVal)) {
    if (polVal < 0.15) polEl.style.color = "#10b981";
    else if (polVal < 0.35) polEl.style.color = "#fbbf24";
    else polEl.style.color = "#ef4444";
  }

  const divVal = parseFloat(data.divergence);
  document.getElementById("val-divergence").textContent = isNaN(divVal)
    ? data.divergence
    : divVal.toFixed(3);

  const goDivVal = data.go_validation_details
    ? parseFloat(data.go_validation_details.wasserstein_distance)
    : NaN;
  const goDivEl = document.getElementById("val-go-divergence");
  if (goDivEl) {
    goDivEl.textContent = isNaN(goDivVal) ? "--" : goDivVal.toFixed(3);
  }

  // 4. Update Agents
  const states = data.agent_states;
  const influences = data.agent_influence || [];
  agentMetadata = data.agent_metadata || [];

  selectedAgentIndex = null;
  dossier.classList.add("hidden");

  const emotionCounts = {};
  for (let i = 0; i < agents.length; i++) {
    if (i < states.length) {
      const emote = states[i];
      emotionCounts[emote] = (emotionCounts[emote] || 0) + 1;
      agents[i].updateState(emote, emotionCounts[emote]);
      agents[i].updateInfluence(influences[i]);
    } else {
      agents[i].updateState("Neutral");
      agents[i].updateInfluence(1.0);
    }
  }

  let majorityEmotion = "Neutral";
  let maxCount = 0;
  for (const emote in emotionCounts) {
    if (emotionCounts[emote] > maxCount) {
      maxCount = emotionCounts[emote];
      majorityEmotion = emote;
    }
  }

  const majorityEl = document.getElementById("val-majority");
  if (majorityEl) {
    majorityEl.textContent = majorityEmotion.toUpperCase();
    majorityEl.style.color = PALETTE[majorityEmotion] || "#ffffff";
  }

  calculateClusterAggregates();

  // 5. Update Explainability UI
  const explainBtn = document.getElementById("btn-explain");
  if (data.explainability) {
    explainBtn.classList.remove("hidden");
    // Format bold text as HTML strong
    const formatBold = (str) => str.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    document.getElementById("ex-cognitive").innerHTML = formatBold(data.explainability.cognitive_drivers || "--");
    document.getElementById("ex-shift-story").innerHTML = formatBold(data.explainability.shift_story || "--");
    document.getElementById("ex-viral").innerHTML = formatBold(data.explainability.viral_dynamics || "--");
    document.getElementById("ex-tug-of-war").innerHTML = formatBold(data.explainability.tug_of_war || "--");
    document.getElementById("ex-structure").innerHTML = formatBold(data.explainability.societal_structure || "--");
    
    const demoList = document.getElementById("ex-demographics");
    demoList.innerHTML = "";
    if (data.explainability.demographics && data.explainability.demographics.length > 0) {
      data.explainability.demographics.forEach(demo => {
        const item = document.createElement("div");
        item.className = "demo-item";
        item.innerHTML = `<div class="demo-item-name">${demo.name}</div><div class="demo-item-desc">${formatBold(demo.description)}</div>`;
        demoList.appendChild(item);
      });
    } else {
      demoList.innerHTML = "<div>No specific demographics found.</div>";
    }
  } else {
    explainBtn.classList.add("hidden");
  }
}

// --- CLUSTER AGGREGATE CALCULATION ---
let totalCounts = { regions: {}, roles: {} };

function calculateClusterAggregates() {
  clusterAggregates = {};
  totalCounts = { regions: {}, roles: {} };
  const emotions = [
    "Joy",
    "Trust",
    "Fear",
    "Surprise",
    "Sadness",
    "Disgust",
    "Anger",
    "Anticipation",
    "Neutral",
  ];

  emotions.forEach((e) => {
    clusterAggregates[e] = {
      count: 0,
      big5: [0, 0, 0, 0, 0],
      regions: {},
      roles: {},
    };
  });

  for (let i = 0; i < agents.length; i++) {
    const agent = agents[i];
    const meta = agentMetadata[i];
    if (!meta) continue;

    const emotion = agent.emotion;
    const agg = clusterAggregates[emotion];

    // Global Totals
    totalCounts.regions[meta.region] =
      (totalCounts.regions[meta.region] || 0) + 1;
    totalCounts.roles[meta.role] = (totalCounts.roles[meta.role] || 0) + 1;

    // Cluster Specific
    agg.count++;
    for (let j = 0; j < 5; j++) {
      agg.big5[j] += meta.big5[j];
    }

    agg.regions[meta.region] = (agg.regions[meta.region] || 0) + 1;
    agg.roles[meta.role] = (agg.roles[meta.role] || 0) + 1;
  }

  // Calculate means
  emotions.forEach((e) => {
    const agg = clusterAggregates[e];
    if (agg.count > 0) {
      for (let j = 0; j < 5; j++) {
        agg.big5[j] /= agg.count;
      }
    }
  });
}

class Agent {
  constructor(id) {
    this.id = id;
    this.x = Math.random() * width;
    this.y = Math.random() * height;
    this.tx = this.x;
    this.ty = this.y;
    this.speed = 0.02 + Math.random() * 0.04;
    this.emotion = "Neutral";
    this.active = false;
    this.influence = 1.0;
    this.rank = 0;

    this.baseSize = 1.0;
    this.currentSize = 1.0;
  }

  updateState(newEmotion, rank = 0) {
    this.emotion = newEmotion;
    this.active = newEmotion !== "Neutral";
    this.rank = rank;
    this.setTarget();
  }

  updateInfluence(inf) {
    this.influence = inf || 1.0;
  }

  setTarget() {
    // SCALABLE AGENT SIZING
    // Smaller agents for higher counts to reduce visual noise
    const countFactor = Math.max(0, 1 - agents.length / 15000);
    this.baseSize = 0.8 + countFactor * 1.2 + Math.random() * 0.5;

    if (this.emotion === "Neutral") {
      // Constrain Neutral agents to a central circle to avoid overlap with emotion ring
      const innerRadius = Math.min(width, height) * 0.12;
      const angle = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.random()) * innerRadius;
      this.tx = centerX + Math.cos(angle) * r;
      this.ty = centerY + Math.sin(angle) * r;
    } else {
      const angle =
        EMOTION_ANGLES[this.emotion] !== undefined
          ? EMOTION_ANGLES[this.emotion]
          : -Math.PI / 2;

      // Push radius out further for more space
      const radius = Math.min(width, height) * 0.36;

      const cx = centerX + Math.cos(angle) * radius;
      const cy = centerY + Math.sin(angle) * radius;

      // ENGINEERED SPACING: Phyllotaxis
      const phi = 137.508 * (Math.PI / 180);
      const r = clusterSpread * Math.sqrt(this.rank);
      const theta = this.rank * phi;

      this.tx = cx + Math.cos(theta) * r;
      this.ty = cy + Math.sin(theta) * r;
    }

    const margin = 20;
    this.tx = Math.max(margin, Math.min(width - margin, this.tx));
    this.ty = Math.max(margin, Math.min(height - margin, this.ty));
  }

  update() {
    this.x += (this.tx - this.x) * this.speed;
    this.y += (this.ty - this.y) * this.speed;
  }

  draw() {
    ctx.fillStyle = PALETTE[this.emotion] || PALETTE["Neutral"];

    const influenceScale = Math.log(this.influence + 1) * 0.8;
    const size = this.baseSize + influenceScale;
    this.currentSize = size;

    if (selectedAgentIndex !== null && agents[selectedAgentIndex] === this) {
      ctx.strokeStyle = "white";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        this.x - size / 2 - 2,
        this.y - size / 2 - 2,
        size + 4,
        size + 4,
      );
    }

    if (this.active) {
      ctx.globalAlpha = 0.9;
      ctx.fillRect(this.x - size / 2, this.y - size / 2, size, size);
    } else {
      ctx.globalAlpha = 0.15;
      ctx.fillRect(
        this.x - this.baseSize / 2,
        this.y - this.baseSize / 2,
        this.baseSize,
        this.baseSize,
      );
    }
    ctx.globalAlpha = 1.0;
  }

  isPointInside(px, py) {
    const size = this.active ? this.currentSize : this.baseSize;
    const padding = agents.length > 5000 ? 4 : 8;
    return (
      px >= this.x - size / 2 - padding &&
      px <= this.x + size / 2 + padding &&
      py >= this.y - size / 2 - padding &&
      py <= this.y + size / 2 + padding
    );
  }
}

// --- SETUP FUNCTIONS ---

function updateVisualScaling() {
  if (!width || !height) return;

  const count = agents.length;
  const emotionRadius = Math.min(width, height) * 0.36;

  // Distance between adjacent emotion cluster centers
  const distBetweenClusters = 2 * emotionRadius * Math.sin(Math.PI / 8);

  // Calculate spread 'c' such that even if 50% of agents are in one emotion,
  // they don't overlap with the neighbor cluster.
  // D/2 is the max radius allowed for a cluster.
  const maxClusterRadius = distBetweenClusters * 0.48;
  const maxExpectedAgentsPerCluster = Math.max(count / 2, 500);

  clusterSpread = maxClusterRadius / Math.sqrt(maxExpectedAgentsPerCluster);

  // Scale-based clamping
  clusterSpread = Math.max(0.4, Math.min(2.5, clusterSpread));

  agents.forEach((a) => a.setTarget());
}

function resize() {
  if (!container) return;
  width = container.clientWidth;
  height = container.clientHeight;
  canvas.width = width;
  canvas.height = height;
  centerX = width / 2;
  centerY = height / 2;

  updateVisualScaling();
}

function initAgents() {
  const countInput = document.getElementById("param-count");
  if (!countInput) return;

  const count = parseInt(countInput.value);
  document.getElementById("disp-count").textContent =
    (count / 1000).toFixed(1) + "k";

  if (agents.length > count) {
    agents = agents.slice(0, count);
  } else {
    for (let i = agents.length; i < count; i++) {
      agents.push(new Agent(i));
    }
  }

  updateVisualScaling();
}

// --- SIMULATION LOGIC ---

function updateSeedInputs() {
  const countSlider = document.getElementById("seed-count-slider");
  if (!countSlider) return;

  const count = parseInt(countSlider.value);
  document.getElementById("disp-seed-count").textContent = count;

  const container = document.getElementById("seed-inputs-container");
  container.innerHTML = ""; // Clear

  for (let i = 0; i < count; i++) {
    const wrapper = document.createElement("div");
    wrapper.className = "seed-item";

    const label = document.createElement("span");
    label.className = "seed-label";
    label.textContent = `Seed ${i + 1}`;

    const input = document.createElement("input");
    input.type = "number";
    input.className = "seed-input";
    // Default random seed
    input.value = Math.floor(Math.random() * 10000);

    wrapper.appendChild(label);
    wrapper.appendChild(input);
    container.appendChild(wrapper);
  }
}

async function runSimulation() {
  const inputVal = document.getElementById("news-input").value;
  if (!inputVal) {
    alert("Please enter a news headline.");
    return;
  }

  const statusLabel = document.getElementById("sys-status");
  const runBtn = document.getElementById("btn-run");

  statusLabel.textContent = "COMPUTING...";
  statusLabel.style.color = "#fbbf24";
  runBtn.disabled = true;
  runBtn.style.opacity = "0.5";

  // Clear UI
  filmstrip.classList.add("hidden");
  agents.forEach((a) => a.updateState("Neutral"));

  // 1. Prepare Payload: Include Main UI config + Batch experiments
  const mainRun = {
    seed: Math.floor(42),
    temperature: parseFloat(document.getElementById("param-temp").value),
    emotion_temperature: parseFloat(
      document.getElementById("param-emotion-temp").value || 0.2,
    ),
    panic_threshold: parseFloat(
      document.getElementById("param-panic-thresh").value || -1.2,
    ),
    region: "All",
    role: document.getElementById("filter-role").value,
    agent_count: parseInt(document.getElementById("param-count").value),
    use_distortion: document
      .getElementById("tog-distortion")
      .classList.contains("active"),
    use_pressure: document
      .getElementById("tog-pressure")
      .classList.contains("active"),
    use_maslow: document
      .getElementById("tog-maslow")
      .classList.contains("active"),
    use_power_law: document
      .getElementById("tog-power-law")
      .classList.contains("active"),
    cross_dim_interaction_strength: parseFloat(
      document.getElementById("res-cross-dim")?.value || 0.3,
    ),
    threat_sensitivity_gain: parseFloat(
      document.getElementById("res-threat-sens")?.value || 1.5,
    ),
    k_processing_tanh_gain: parseFloat(
      document.getElementById("res-k-process")?.value || 1.5,
    ),
    relevance_importance_weight: parseFloat(
      document.getElementById("res-rel-imp")?.value || 0.7,
    ),
    relevance_base_weight: parseFloat(
      document.getElementById("res-rel-base")?.value || 0.3,
    ),
    threat_amplifier_gain: parseFloat(
      document.getElementById("res-threat-amp")?.value || 1.5,
    ),
    stress_neurotic_amplification: parseFloat(
      document.getElementById("res-stress-neur")?.value || 1.5,
    ),
    stress_openness_reduction: parseFloat(
      document.getElementById("res-stress-open")?.value || 0.5,
    ),
    stress_extraversion_boost: parseFloat(
      document.getElementById("res-stress-ext")?.value || 0.7,
    ),
    outrage_gain: parseFloat(
      document.getElementById("res-outrage")?.value || 2.5,
    ),
    max_viral_multiplier: parseFloat(
      document.getElementById("res-viral")?.value || 10.0,
    ),
    saturation_midpoint: parseFloat(
      document.getElementById("res-sat")?.value || 1.5,
    ),
    distortion_max_noise: parseFloat(
      document.getElementById("res-dist-max")?.value || 0.4,
    ),
    distortion_neurotic_gain: parseFloat(
      document.getElementById("res-dist-neur")?.value || 0.6,
    ),
    evolution_generations: parseInt(
      document.getElementById("res-evo-gen")?.value || 10,
    ),
    inheritance_fraction: parseFloat(
      document.getElementById("res-evo-inh")?.value || 0.7,
    ),
    shock_frequency: parseFloat(
      document.getElementById("res-evo-shock-freq")?.value || 0.1,
    ),
    shock_magnitude: parseFloat(
      document.getElementById("res-evo-shock-mag")?.value || 0.2,
    ),
  };

  const payload = {
    news_text: inputVal,
    runs: [
      mainRun,
      ...batchRuns.map((run) => ({
        seed: run.seed,
        temperature: run.temperature,
        emotion_temperature: run.emotion_temperature,
        panic_threshold: run.panic_threshold,
        region: run.region,
        role: run.role,
        agent_count: run.agent_count,
        use_distortion: run.use_distortion,
        use_pressure: run.use_pressure,
        use_maslow: run.use_maslow,
        use_power_law: run.use_power_law,
        cross_dim_interaction_strength: run.cross_dim_interaction_strength,
        threat_sensitivity_gain: run.threat_sensitivity_gain,
        k_processing_tanh_gain: run.k_processing_tanh_gain,
        relevance_importance_weight: run.relevance_importance_weight,
        relevance_base_weight: run.relevance_base_weight,
        threat_amplifier_gain: run.threat_amplifier_gain,
        stress_neurotic_amplification: run.stress_neurotic_amplification,
        stress_openness_reduction: run.stress_openness_reduction,
        stress_extraversion_boost: run.stress_extraversion_boost,
        outrage_gain: run.outrage_gain,
        max_viral_multiplier: run.max_viral_multiplier,
        saturation_midpoint: run.saturation_midpoint,
        distortion_max_noise: run.distortion_max_noise,
        distortion_neurotic_gain: run.distortion_neurotic_gain,
        evolution_generations: run.evolution_generations,
        inheritance_fraction: run.inheritance_fraction,
        shock_frequency: run.shock_frequency,
        shock_magnitude: run.shock_magnitude,
      })),
    ],
  };

  try {
    console.log("Sending Batch Request:", payload);
    const response = await fetch("http://localhost:8000/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const data = await response.json();
    console.log("Batch Results:", data);

    currentBatchResults = data.results;

    // 2. Display First Result as default
    displayResult(currentBatchResults[0]);

    // 3. Render Filmstrip if multiple runs
    renderFilmstrip(currentBatchResults);

    // 4. Save to History
    simulationHistory.push({
      prompt: inputVal,
      timestamp: new Date().toLocaleString(),
      runs: currentBatchResults,
    });

    statusLabel.textContent = "CONVERGED";
    statusLabel.style.color = "#10b981";
  } catch (error) {
    console.error("Simulation failed:", error);
    statusLabel.textContent = "ERROR";
    statusLabel.style.color = "#ef4444";
    alert(`Simulation failed: ${error.message}`);
  } finally {
    runBtn.disabled = false;
    runBtn.style.opacity = "1.0";
  }
}

// --- ANIMATION LOOP ---
function animate() {
  ctx.clearRect(0, 0, width, height);
  for (let i = 0; i < agents.length; i++) {
    if (!agents[i].active) {
      agents[i].update();
      agents[i].draw();
    }
  }
  for (let i = 0; i < agents.length; i++) {
    if (agents[i].active) {
      agents[i].update();
      agents[i].draw();
    }
  }
  requestAnimationFrame(animate);
}

// --- EVENT LISTENERS ---
window.addEventListener("resize", resize);
document.getElementById("btn-run").addEventListener("click", runSimulation);
document.getElementById("param-count").addEventListener("input", initAgents);

// Explainability Toggle
const explainPanel = document.getElementById("explainability-panel");
document.getElementById("btn-explain").addEventListener("click", () => {
  explainPanel.classList.toggle("hidden");
});
document.getElementById("close-explainability").addEventListener("click", () => {
  explainPanel.classList.add("hidden");
});

// Settings Panel Toggle
document.getElementById("settings-btn").addEventListener("click", () => {
  sidebar.classList.toggle("retracted");
  setTimeout(resize, 310);
});

// --- HISTORY LOGIC ---
function toggleHistory() {
  historyPanel.classList.toggle("hidden");
  if (!historyPanel.classList.contains("hidden")) renderHistory();
}

function renderHistory() {
  const container = document.getElementById("history-graph");
  if (simulationHistory.length === 0) {
    container.innerHTML =
      '<div class="empty-history-msg">No simulations recorded.</div>';
    return;
  }
  container.innerHTML = "";

  simulationHistory.forEach((session, sIdx) => {
    const entry = document.createElement("div");
    entry.className = "history-entry";

    // Parent Prompt Node
    entry.innerHTML = `
            <div class="parent-node">
                <span class="parent-text">"${session.prompt}"</span>
                <div class="parent-meta">
                    <span>${session.timestamp}</span>
                    <span>${session.runs.length} Runs</span>
                </div>
            </div>
            <div class="child-nodes-wrapper">
                ${session.runs
                  .map((run, rIdx) => {
                    const cfg = run.config;
                    const params =
                      [
                        cfg.use_distortion ? "DIST" : null,
                        cfg.use_pressure ? "TIME" : null,
                        cfg.use_maslow ? "MSLW" : null,
                        cfg.use_power_law ? "PWR" : null,
                      ]
                        .filter(Boolean)
                        .join(", ") || "NONE";

                    const isActive =
                      sIdx === currentSessionIndex &&
                      rIdx === currentRunInSessionIndex;

                    return `
                        <div class="run-node ${isActive ? "active" : ""}" onclick="restoreSimulation(${sIdx}, ${rIdx})">
                            <div class="emotion-indicator" style="background: ${PALETTE[run.dominant_emotion] || "#333"}"></div>
                            <div class="run-info">
                                <div class="run-label">${run.dominant_emotion.toUpperCase()} (Seed: ${cfg.seed}, T: ${cfg.temperature})</div>
                                <div class="run-stats">
                                    POL: ${run.polarization} | POP: ${(cfg.agent_count / 1000).toFixed(1)}k | ${cfg.role}
                                    <br>PARAMS: ${params}
                                </div>
                            </div>
                            <div class="node-action" onclick="event.stopPropagation(); downloadData(${sIdx}, ${rIdx})">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4m4-5l5 5 5-5m-5 5V3"/></svg>
                            </div>
                        </div>
                    `;
                  })
                  .join("")}
            </div>
        `;
    container.appendChild(entry);
  });
}

function restoreSimulation(sIdx, rIdx) {
  const session = simulationHistory[sIdx];
  if (!session) return;
  const run = session.runs[rIdx];
  if (!run) return;

  currentSessionIndex = sIdx;
  currentRunInSessionIndex = rIdx;

  // Update Batch State to match restored run
  currentBatchResults = session.runs;

  displayResult(run);
  renderFilmstrip(session.runs);
  renderHistory();
}

function downloadData(sIdx, rIdx) {
  const run = simulationHistory[sIdx].runs[rIdx];
  const blob = new Blob([JSON.stringify(run, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `run_session${sIdx}_run${rIdx}.json`;
  a.click();
}

function downloadAllData() {
  const blob = new Blob([JSON.stringify(simulationHistory, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `session_history.json`;
  a.click();
}

function handleLoadHistory(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      simulationHistory = Array.isArray(data) ? data : [data];
      renderHistory();
      if (simulationHistory.length > 0) {
        const sIdx = simulationHistory.length - 1;
        const rIdx = simulationHistory[sIdx].runs ? 0 : -1;
        if (rIdx !== -1) restoreSimulation(sIdx, rIdx);
      }
    } catch (err) {
      alert("Invalid JSON");
    }
  };
  reader.readAsText(file);
}

// --- INITIALIZATION & STATUS ---
async function checkBackendStatus() {
  const statusLabel = document.getElementById("sys-status");
  const runBtn = document.getElementById("btn-run");
  try {
    const response = await fetch("http://localhost:8000/health");
    if (response.ok) {
      statusLabel.textContent = "READY";
      statusLabel.style.color = "#10b981";
      runBtn.disabled = false;
      runBtn.style.opacity = "1.0";
    } else {
      console.error(
        "Backend health check failed with status:",
        response.status,
      );
      statusLabel.textContent = "SERVER ERROR";
      statusLabel.style.color = "#fbbf24";
    }
  } catch (error) {
    console.error("Backend connection failed:", error);
    statusLabel.textContent = "OFFLINE";
    statusLabel.style.color = "#ef4444";
  }
}

function showDossier(index) {
  const meta = agentMetadata[index];
  if (!meta) return;
  selectedAgentIndex = index;
  const emotion = agents[index].emotion;

  document.getElementById("ds-id").textContent = meta.id
    .toString()
    .padStart(4, "0");
  document.getElementById("ds-role").textContent = meta.role;

  const traits = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
  ];
  meta.big5.forEach((val, i) => {
    const percent = Math.round(val * 100);
    document.getElementById(`bar-${traits[i]}`).style.width = `${percent}%`;
    document.getElementById(`val-${traits[i]}`).textContent = `${percent}%`;
  });

  const agg = clusterAggregates[emotion];
  if (agg) {
    document.getElementById("ds-cluster-name").textContent =
      emotion.toUpperCase();
    agg.big5.forEach((val, i) => {
      const percent = Math.round(val * 100);
      document.getElementById(`c-bar-${traits[i]}`).style.width = `${percent}%`;
      document.getElementById(`c-val-${traits[i]}`).textContent = `${percent}%`;
    });

    const rolesContainer = document.getElementById("ds-cluster-roles");
    if (rolesContainer) {
      rolesContainer.innerHTML = "";
      const totalRoles = Object.values(agg.roles).reduce((a, b) => a + b, 0);
      for (const [roleName, count] of Object.entries(agg.roles)) {
        const percent = totalRoles > 0 ? Math.round((count / totalRoles) * 100) : 0;
        rolesContainer.innerHTML += `
          <div class="dist-item">
              <div class="dist-label-row">
                  <span>${roleName}</span>
                  <span>${percent}% (${count})</span>
              </div>
              <div class="dist-bar-bg">
                  <div class="dist-bar" style="width: ${percent}%"></div>
              </div>
          </div>
        `;
      }
    }
  }
  dossier.classList.remove("hidden");
}

// --- EVENT LISTENERS ---
document.getElementById("history-btn").addEventListener("click", toggleHistory);
document
  .getElementById("close-history")
  .addEventListener("click", toggleHistory);
document
  .getElementById("download-all-btn")
  .addEventListener("click", downloadAllData);
document
  .getElementById("load-history-btn")
  .addEventListener("click", () =>
    document.getElementById("load-history-input").click(),
  );
document
  .getElementById("load-history-input")
  .addEventListener("change", handleLoadHistory);

document.getElementById("tab-agent").addEventListener("click", () => {
  document.getElementById("tab-agent").classList.add("active");
  document.getElementById("tab-cluster").classList.remove("active");
  document.getElementById("view-agent").classList.remove("hidden");
  document.getElementById("view-cluster").classList.add("hidden");
});
document.getElementById("tab-cluster").addEventListener("click", () => {
  document.getElementById("tab-cluster").classList.add("active");
  document.getElementById("tab-agent").classList.remove("active");
  document.getElementById("view-cluster").classList.remove("hidden");
  document.getElementById("view-agent").classList.add("hidden");
});

canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left,
    y = e.clientY - rect.top;
  let found = -1;
  for (let i = agents.length - 1; i >= 0; i--) {
    if (agents[i].isPointInside(x, y)) {
      found = i;
      break;
    }
  }

  if (found !== -1) {
    const meta = agentMetadata[found];
    const agent = agents[found];
    if (meta) {
      document.getElementById("tt-id").textContent = meta.id.toString().padStart(4, "0");
      document.getElementById("tt-role").textContent = meta.role;
      document.getElementById("tt-influence").textContent = agent.influence.toFixed(2);
      document.getElementById("tt-state").textContent = agent.emotion.toUpperCase();
      document.getElementById("tt-state").style.color = PALETTE[agent.emotion] || "#fff";
      
      tooltip.style.left = (e.clientX + 15) + "px";
      tooltip.style.top = (e.clientY + 15) + "px";
      tooltip.style.opacity = "1";
    }
  } else {
    tooltip.style.opacity = "0";
  }
});

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left,
    y = e.clientY - rect.top;
  let found = -1;
  for (let i = agents.length - 1; i >= 0; i--) {
    if (agents[i].isPointInside(x, y)) {
      found = i;
      break;
    }
  }
  if (found !== -1) showDossier(found);
  else {
    selectedAgentIndex = null;
    dossier.classList.add("hidden");
  }
});

document.getElementById("ds-close").addEventListener("click", () => {
  selectedAgentIndex = null;
  dossier.classList.add("hidden");
});

filmstripToggleBtn.addEventListener("click", toggleFilmstrip);

document.getElementById("param-temp")?.addEventListener("input", (e) => {
  document.getElementById("disp-temp").textContent = parseFloat(
    e.target.value,
  ).toFixed(1);
});
document
  .getElementById("param-emotion-temp")
  ?.addEventListener("input", (e) => {
    document.getElementById("disp-emotion-temp").textContent = parseFloat(
      e.target.value,
    ).toFixed(2);
  });
document
  .getElementById("param-panic-thresh")
  ?.addEventListener("input", (e) => {
    document.getElementById("disp-panic-thresh").textContent = parseFloat(
      e.target.value,
    ).toFixed(1);
  });

document.getElementById("tab-basic")?.addEventListener("click", () => {
  document.getElementById("tab-basic").classList.add("active");
  document.getElementById("tab-researcher").classList.remove("active");
  document.getElementById("sidebar-basic").classList.remove("hidden");
  document.getElementById("sidebar-researcher").classList.add("hidden");
});
document.getElementById("tab-researcher")?.addEventListener("click", () => {
  document.getElementById("tab-researcher").classList.add("active");
  document.getElementById("tab-basic").classList.remove("active");
  document.getElementById("sidebar-researcher").classList.remove("hidden");
  document.getElementById("sidebar-basic").classList.add("hidden");
});

resize();
initAgents();
renderBatchUI();
animate();
checkBackendStatus();
