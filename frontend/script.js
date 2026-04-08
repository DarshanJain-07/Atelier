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
let clusterMetrics = { silhouette_score: 0, davies_bouldin_index: 0 };
let simulationHistory = [];
let currentSessionIndex = -1;
let currentRunInSessionIndex = -1;
let selectedAgentIndex = null;
let width, height, centerX, centerY;
let clusterSpread = 1.0;

// --- BATCH STATE ---
let batchRuns = [];
let currentBatchResults = [];

const historyPanel = document.getElementById("history-panel");
const filmstrip = document.getElementById("run-filmstrip");
const filmstripToggleBtn = document.getElementById("filmstrip-toggle-btn");
let isFilmstripManuallyHidden = false;

// --- BATCH UI RENDERER ---
function renderBatchUI() {
  const container = document.getElementById("batch-container");
  container.innerHTML = "";

  if (batchRuns.length === 0) {
    const emptyMsg = document.createElement("div");
    emptyMsg.className = "empty-history-msg";
    emptyMsg.style.padding = "10px 0";
    emptyMsg.textContent = "No extra experiments added.";
    container.appendChild(emptyMsg);
    return;
  }

  batchRuns.forEach((run, index) => {
    const item = document.createElement("div");
    item.className = "batch-item";
    item.innerHTML = `
            <div class="batch-header">
                <span class="batch-title">EXPERIMENT ${index + 1}</span>
                <button class="batch-remove" onclick="removeRun(${run.id})">&times;</button>
            </div>
            <div class="sidebar-grid">
                <div class="sidebar-field">
                    <label data-tooltip="Random seed for deterministic outcomes.">Seed</label>
                    <input type="number" class="sidebar-input" value="${run.seed}" onchange="debouncedUpdateRun(${run.id}, 'seed', this.value)">
                </div>
                <div class="sidebar-field">
                    <label data-tooltip="Controls the mutation rate. Higher values increase randomness.">Temp</label>
                    <input type="number" class="sidebar-input" step="0.1" min="0" max="1" value="${run.temperature}" onchange="debouncedUpdateRun(${run.id}, 'temperature', this.value)">
                </div>
            </div>
            <div class="sidebar-grid">
                <div class="sidebar-field sidebar-field-span-2">
                    <label data-tooltip="Filter agents by socioeconomic class.">Class</label>
                    <select class="sidebar-input select-ui-font" onchange="debouncedUpdateRun(${run.id}, 'social_class', this.value)">
                        <option value="All" ${run.social_class === "All" ? "selected" : ""}>All</option>
                        <option value="Underclass" ${run.social_class === "Underclass" ? "selected" : ""}>Underclass</option>
                        <option value="Working Class" ${run.social_class === "Working Class" ? "selected" : ""}>Working Class</option>
                        <option value="Middle Class" ${run.social_class === "Middle Class" ? "selected" : ""}>Middle Class</option>
                        <option value="Upper Middle" ${run.social_class === "Upper Middle" ? "selected" : ""}>Upper Middle</option>
                        <option value="Elite" ${run.social_class === "Elite" ? "selected" : ""}>Elite</option>
                    </select>
                </div>
            </div>
            <div class="sidebar-grid sidebar-grid-margin">
                <div class="sidebar-field sidebar-field-span-2">
                    <label data-tooltip="Total number of agents to simulate.">Population: <span id="batch-pop-val-${run.id}">${(run.agent_count / 1000).toFixed(1)}k</span></label>
                    <input type="range" min="1000" max="15000" step="1000" value="${run.agent_count}"
                        oninput="document.getElementById('batch-pop-val-${run.id}').textContent = (this.value/1000).toFixed(1) + 'k'; debouncedUpdateRun(${run.id}, 'agent_count', this.value)">
                </div>
            </div>
            <div class="batch-toggle-grid">
                <button class="batch-tog ${run.use_distortion ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'use_distortion', this.classList.contains('active'))" title="Distortion">DIST</button>
                <button class="batch-tog ${run.use_pressure ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'use_pressure', this.classList.contains('active'))" title="Time Pressure">TIME</button>
                <button class="batch-tog ${run.use_maslow ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'use_maslow', this.classList.contains('active'))" title="Maslow Gate">MSLW</button>
                <button class="batch-tog ${run.use_power_law ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'use_power_law', this.classList.contains('active'))" title="Power Law">PWR</button>
                <button class="batch-tog ${run.use_agent_memory ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'use_agent_memory', this.classList.contains('active'))" title="Agent Memory">MEM</button>
                <button class="batch-tog ${run.use_algorithmic_amplification ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'use_algorithmic_amplification', this.classList.contains('active'))" title="Algo Amplification">ALGO</button>
                <button class="batch-tog ${run.use_network_topology ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'use_network_topology', this.classList.contains('active'))" title="Network Topology">NET</button>
                <button class="batch-tog ${run.enable_evolution ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'enable_evolution', this.classList.contains('active'))" title="Evolution">EVO</button>
                <button class="batch-tog ${run.use_granovetter_thresholds ? "active" : ""}" onclick="this.classList.toggle('active'); debouncedUpdateRun(${run.id}, 'use_granovetter_thresholds', this.classList.contains('active'))" title="Granovetter Model">GRAN</button>
            </div>
        `;
    container.appendChild(item);
  });
}

window.removeRun = (id) => {
  batchRuns = batchRuns.filter((r) => r.id !== id);
  renderBatchUI();
};

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

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
          ? Number(value)
          : value;
    }
  }
};

window.debouncedUpdateRun = debounce(window.updateRun, 50);

function readFloat(id, fallback) {
  return parseFloat(document.getElementById(id)?.value || fallback);
}

function readInt(id, fallback) {
  return parseInt(document.getElementById(id)?.value || fallback, 10);
}

function isToggleActive(id, fallback = false) {
  return document.getElementById(id)?.classList.contains("active") ?? fallback;
}

function getResearcherSettings() {
  return {
    cross_dim_interaction_strength: readFloat("res-cross-dim", 0.3),
    threat_sensitivity_gain: readFloat("res-threat-sens", 1.5),
    k_processing_tanh_gain: readFloat("res-k-process", 1.5),
    relevance_importance_weight: readFloat("res-rel-imp", 0.7),
    relevance_base_weight: readFloat("res-rel-base", 0.3),
    threat_amplifier_gain: readFloat("res-threat-amp", 1.5),
    stress_neurotic_amplification: readFloat("res-stress-neur", 1.5),
    stress_openness_reduction: readFloat("res-stress-open", 0.5),
    stress_extraversion_boost: readFloat("res-stress-ext", 0.7),
    outrage_gain: readFloat("res-outrage", 8.0),
    max_viral_multiplier: readFloat("res-viral", 10.0),
    saturation_midpoint: readFloat("res-sat", 0.5),
    distortion_max_noise: readFloat("res-dist-max", 0.4),
    distortion_neurotic_gain: readFloat("res-dist-neur", 0.6),
    distortion_relative_cap: readFloat("res-dist-rel-cap", 1.1),
    distortion_absolute_cap: readFloat("res-dist-abs-cap", 0.5),
    evolution_generations: readInt("res-evo-gen", 10),
    inheritance_fraction: readFloat("res-evo-inh", 0.7),
    shock_frequency: readFloat("res-evo-shock-freq", 0.1),
    shock_magnitude: readFloat("res-evo-shock-mag", 0.2),
    algo_sample_size: readFloat("res-algo-sample", 0.1),
    algo_top_k: readInt("res-algo-topk", 2),
    algo_min_active_value: readFloat("res-algo-active", 0.05),
    algo_exaggeration_factor: readFloat("res-algo-exagg", 1.5),
    memory_decay_rate: readFloat("res-mem-decay", 0.7),
    memory_desensitization_gain: readFloat("res-mem-desens", 0.5),
    memory_trigger_stacking_gain: readFloat("res-mem-trigger", 1.2),
    stewing_ticks: readInt("res-stew-ticks", 5),
    stewing_self_retention: readFloat("res-stew-self", 0.6),
    stewing_local_influence: readFloat("res-stew-local", 0.3),
    stewing_viral_influence: readFloat("res-stew-viral", 0.1),
    perception_social_consensus_gain: readFloat("res-consensus", 0.25),
    triadic_closure_prob: readFloat("res-triadic-prob", 0.2),
    triadic_closure_iterations: readInt("res-triadic-iter", 1),
    homophily_strength: readFloat("res-homophily", 2.0),
    personality_socialization_gain: readFloat("res-socialize", 0.05),
    use_granovetter_thresholds: isToggleActive("tog-granovetter", true),
    granovetter_threshold_mean: readFloat("res-gran-mean", 0.25),
    granovetter_threshold_std: readFloat("res-gran-std", 0.15),
    memory_social_rehearsal_gain: readFloat("res-mem-rehearsal", 0.4),
    use_selective_exposure: isToggleActive("tog-selective-exposure", true),
    selective_exposure_base_tolerance: readFloat("res-selective-base", -0.3),
    selective_exposure_openness_factor: readFloat("res-selective-open", 0.4),
    selective_exposure_gain: readFloat("res-selective-gain", 8.0),
    selective_exposure_max_suppression: readFloat("res-selective-max", 0.85),
    threshold_gain: readFloat("res-threshold-gain", 18.0),
    engagement_threshold: readFloat("res-engagement-threshold", 0.15),
    engagement_gain: readFloat("res-engagement-gain", 10.0),
  };
}

document.getElementById("btn-add-run").addEventListener("click", () => {
  if (batchRuns.length >= 6) return;
  
  let defaults;
  if (batchRuns.length > 0) {
    defaults = batchRuns[batchRuns.length - 1];
  } else {
    // Use current UI values
    defaults = {
        temperature: parseFloat(document.getElementById("param-temp").value),
        emotion_temperature: parseFloat(document.getElementById("param-emotion-temp").value || 0.2),
        panic_threshold: parseFloat(document.getElementById("param-panic-thresh").value || -1.2),
        region: "All",
        social_class: document.getElementById("filter-class").value,
        agent_count: parseInt(document.getElementById("param-count").value),
        use_distortion: document.getElementById("tog-distortion").classList.contains("active"),
        use_pressure: document.getElementById("tog-pressure").classList.contains("active"),
        use_maslow: document.getElementById("tog-maslow").classList.contains("active"),
        use_power_law: document.getElementById("tog-power-law").classList.contains("active"),
        use_agent_memory: document.getElementById("tog-memory").classList.contains("active"),
        use_algorithmic_amplification: document.getElementById("tog-algo-amp").classList.contains("active"),
        use_network_topology: document.getElementById("tog-network").classList.contains("active"),
        enable_evolution: document.getElementById("tog-evolution").classList.contains("active"),
        ...getResearcherSettings(),
    };
  }

  batchRuns.push({
    id: Date.now(),
    ...defaults,
    seed: defaults.seed || 42,
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

  const polVal = parseFloat(data.bimodality !== undefined ? data.bimodality : data.polarization);
  const polEl = document.getElementById("val-polarization");
  polEl.textContent = isNaN(polVal) ? "--" : polVal.toFixed(3);

  if (!isNaN(polVal)) {
    if (polVal < 0.3) polEl.style.color = "#10b981"; // Consensus
    else if (polVal < 0.555) polEl.style.color = "#fbbf24"; // Fragmenting
    else polEl.style.color = "#ef4444"; // Bimodal (Sarle's Threshold)
  }

  const eliteDivVal = parseFloat(data.elite_divergence);
  const eliteDivEl = document.getElementById("val-elite-divergence");
  if (eliteDivEl) {
    eliteDivEl.textContent = isNaN(eliteDivVal) ? "--" : eliteDivVal.toFixed(3);
    if (!isNaN(eliteDivVal)) {
        if (eliteDivVal < 0.2) eliteDivEl.style.color = "#10b981";
        else if (eliteDivVal < 0.4) eliteDivEl.style.color = "#fbbf24";
        else eliteDivEl.style.color = "#ef4444";
    }
  }

  const negIntVal = data.negative_integral !== undefined ? parseFloat(data.negative_integral) : NaN;
  const negIntEl = document.getElementById("val-negative-integral");
  if (negIntEl) {
    negIntEl.textContent = isNaN(negIntVal) ? "--" : negIntVal.toFixed(3);
    if (!isNaN(negIntVal)) {
      if (negIntVal < 2.0) negIntEl.style.color = "#10b981";
      else if (negIntVal < 5.0) negIntEl.style.color = "#fbbf24";
      else negIntEl.style.color = "#ef4444";
    }
  }

  const activePopVal = data.acting_ratio !== undefined ? parseFloat(data.acting_ratio) * 100 : NaN;
  const activePopEl = document.getElementById("val-active-pop");
  if (activePopEl) {
    activePopEl.textContent = isNaN(activePopVal) ? "--" : `${activePopVal.toFixed(1)}%`;
  }

  // 4. Update Agents
  const states = data.agent_states;
  const influences = data.agent_influence || [];
  agentMetadata = data.agent_metadata || [];
  
  if (data.cluster_metrics) {
    clusterMetrics = data.cluster_metrics;
  } else {
    clusterMetrics = { silhouette_score: 0, davies_bouldin_index: 0 };
  }

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
    document.getElementById("ex-reasoning").innerHTML = formatBold(data.reasoning || "--");
    document.getElementById("ex-shift-story").innerHTML = formatBold(data.explainability.shift_story || "--");
    document.getElementById("ex-viral").innerHTML = formatBold(data.explainability.viral_dynamics || "--");
    document.getElementById("ex-tug-of-war").innerHTML = formatBold(data.explainability.tug_of_war || "--");
    document.getElementById("ex-structure").innerHTML = formatBold(data.explainability.societal_structure || "--");
    document.getElementById("ex-stewing-impact").innerHTML = formatBold(data.validation_details?.stewing_interpretation || "--");
    document.getElementById("ex-endogenous-events").innerHTML = formatBold(data.explainability.endogenous_events || "--");

    const biasContainer = document.getElementById("ex-biases");
    biasContainer.innerHTML = "";
    if (data.detected_biases && data.detected_biases.length > 0) {
        data.detected_biases.forEach(bias => {
            const tag = document.createElement("span");
            tag.className = "bias-tag";
            tag.textContent = bias;
            biasContainer.appendChild(tag);
        });
    } else {
        biasContainer.innerHTML = "<span>None detected</span>";
    }

    const demoContainer = document.getElementById("ex-demographics");
    demoContainer.innerHTML = "";
    if (data.explainability.demographics && data.explainability.demographics.length > 0) {
      data.explainability.demographics.forEach(demo => {
        const item = document.createElement("div");
        item.className = "demo-item";
        item.innerHTML = `<div class="demo-item-name">${demo.name}</div><div class="demo-item-desc">${formatBold(demo.description)}</div>`;
        demoContainer.appendChild(item);
      });
    } else {
      demoContainer.innerHTML = "<div>No specific demographics found.</div>";
    }
  } else {
    explainBtn.classList.add("hidden");
  }

  // 6. Show Endogenous Event Toast if present
  if (data.endogenous_event) {
    showToast(`⚠️ Autopoietic Trigger: ${data.endogenous_event}`, "warning");
  }
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("fade-out");
    setTimeout(() => toast.remove(), 400);
  }, 5000);
}

// --- CLUSTER AGGREGATE CALCULATION ---
let totalCounts = { regions: {}, classes: {} };

function calculateClusterAggregates() {
  clusterAggregates = {};
  totalCounts = { regions: {}, classes: {} };
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
      classes: {},
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
    totalCounts.classes[meta.social_class] = (totalCounts.classes[meta.social_class] || 0) + 1;

    // Cluster Specific
    agg.count++;
    for (let j = 0; j < 5; j++) {
      agg.big5[j] += meta.big5[j];
    }

    agg.regions[meta.region] = (agg.regions[meta.region] || 0) + 1;
    agg.classes[meta.social_class] = (agg.classes[meta.social_class] || 0) + 1;
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
    showToast("Please enter a news headline.", "warning");
    return;
  }

  const statusLabel = document.getElementById("sys-status");
  const runBtn = document.getElementById("btn-run");

  statusLabel.textContent = "COMPUTING...";
  statusLabel.style.color = "#fbbf24";
  runBtn.disabled = true;
  runBtn.style.opacity = "0.5";
  runBtn.style.cursor = "not-allowed";
  runBtn.textContent = "PROCESSING...";

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
    social_class: document.getElementById("filter-class").value,
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
    use_agent_memory: document
      .getElementById("tog-memory")
      .classList.contains("active"),
    use_algorithmic_amplification: document
      .getElementById("tog-algo-amp")
      .classList.contains("active"),
    use_network_topology: document
      .getElementById("tog-network")
      .classList.contains("active"),
    enable_evolution: document
      .getElementById("tog-evolution")
      .classList.contains("active"),
    ...getResearcherSettings(),
  };

  const payload = {
    news_text: inputVal,
    runs: [
      mainRun,
      ...batchRuns.map((run) => {
        const { id, ...runPayload } = run;
        return runPayload;
      }),
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
    showToast(`Simulation failed: ${error.message}`, "error");
  } finally {
    runBtn.disabled = false;
    runBtn.style.opacity = "1.0";
    runBtn.style.cursor = "pointer";
    runBtn.textContent = "RUN MODEL";
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
                                    POL: ${run.polarization} | POP: ${(cfg.agent_count / 1000).toFixed(1)}k | ${cfg.social_class}
                                </div>
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
  showToast("Download successful.", "success");
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
  showToast("Session history downloaded successfully.", "success");
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
      showToast("History loaded successfully.", "success");
    } catch (err) {
      showToast("Invalid JSON file.", "error");
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
      showToast("Backend server error.", "error");
    }
  } catch (error) {
    console.error("Backend connection failed:", error);
    statusLabel.textContent = "OFFLINE";
    statusLabel.style.color = "#ef4444";
    showToast("Backend is offline.", "error");
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
  document.getElementById("ds-class").textContent = meta.social_class;

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

    const classesContainer = document.getElementById("ds-cluster-classes");
    if (classesContainer) {
      classesContainer.innerHTML = "";
      for (const [className, count] of Object.entries(agg.classes)) {
        const globalCount = totalCounts.classes[className] || 1;
        const percent = Math.round((count / globalCount) * 100);
        classesContainer.innerHTML += `
          <div class="dist-item">
              <div class="dist-label-row">
                  <span>${className}</span>
                  <span>${percent}% (${count}/${globalCount})</span>
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
      document.getElementById("tt-class").textContent = meta.social_class;
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

function resetSection(btn) {
    const section = btn.closest('.config-section');
    if (!section) return;

    // Reset range and number inputs
    const inputs = section.querySelectorAll('input[type="range"], input[type="number"], input[type="text"]');
    inputs.forEach(input => {
        if (input.defaultValue !== undefined) {
            input.value = input.defaultValue;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }
    });

    // Reset selects
    const selects = section.querySelectorAll('select');
    selects.forEach(select => {
        const defaultOption = Array.from(select.options).find(opt => opt.defaultSelected);
        if (defaultOption) {
            select.value = defaultOption.value;
            select.dispatchEvent(new Event('change', { bubbles: true }));
        }
    });

    // Reset toggle buttons
    const toggles = section.querySelectorAll('.toggle-btn');
    toggles.forEach(toggle => {
        if (toggle.hasAttribute('data-default-active')) {
            const shouldBeActive = toggle.getAttribute('data-default-active') === 'true';
            if (shouldBeActive) {
                toggle.classList.add('active');
            } else {
                toggle.classList.remove('active');
            }
            // Trigger any onclick logic if it's not inline, but here they are mostly inline `onclick="this.classList.toggle('active')"`
            // The simulation just reads classList.contains('active') later.
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.setAttribute('data-default-active', btn.classList.contains('active'));
    });
});

resize();
initAgents();
renderBatchUI();
animate();
checkBackendStatus();
