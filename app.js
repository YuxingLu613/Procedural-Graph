const results = {
  "Claude 4.6": {
    HotpotQA: [74.6, 75.4, 74.5],
    MultiChallenge: [83.73, 89.76, 89.76],
    GDPval: [46.23, 50.13, 51.49],
    ALFWorld: [86.57, 91.79, 93.28],
    "τ-bench": [66.09, 71.3, 73.91],
    "BFCL v3": [61, 65, 67],
  },
  "Gemini 3.1 Pro": {
    HotpotQA: [85.9, 86, 87.3],
    MultiChallenge: [87.95, 95.18, 95.78],
    GDPval: [56.39, 71.37, 78.78],
    ALFWorld: [94.78, 99.25, 100],
    "τ-bench": [72.17, 73.04, 80],
    "BFCL v3": [59, 64, 66],
  },
  "Gemini 3.5 Flash": {
    HotpotQA: [83.1, 83.4, 84.5],
    MultiChallenge: [81.33, 91.57, 91.57],
    GDPval: [59.33, 62.45, 64.42],
    ALFWorld: [82.84, 90.3, 94.03],
    "τ-bench": [31.3, 38.26, 44.35],
    "BFCL v3": [56, 58, 67],
  },
  "Grok 4.1 Fast": {
    HotpotQA: [72.7, 74.3, 74.4],
    MultiChallenge: [68.07, 84.94, 86.75],
    GDPval: [57.23, 68.15, 71.19],
    ALFWorld: [26.12, 42.54, 39.55],
    "τ-bench": [64.35, 68.7, 67.83],
    "BFCL v3": [52, 56, 60],
  },
};

const benchmarkMeta = {
  HotpotQA: ["Open-domain QA", "Accuracy ↑"],
  MultiChallenge: ["Multi-turn dialogue", "Overall success ↑"],
  GDPval: ["Professional work", "Rubric score ↑"],
  ALFWorld: ["Embodied household", "Success rate ↑"],
  "τ-bench": ["Customer service", "Pass¹ ↑"],
  "BFCL v3": ["Function calling", "Accuracy ↑"],
};

let selectedModel = "Gemini 3.1 Pro";
let selectedBenchmark = "HotpotQA";

const modelSwitcher = document.querySelector("#model-switcher");
const benchmarkList = document.querySelector("#benchmark-list");
const resultBars = document.querySelector("#result-bars");

Object.keys(results).forEach((model) => {
  const button = document.createElement("button");
  button.className = `model-button${model === selectedModel ? " active" : ""}`;
  button.type = "button";
  button.textContent = model;
  button.setAttribute("role", "tab");
  button.setAttribute("aria-selected", model === selectedModel ? "true" : "false");
  button.addEventListener("click", () => {
    selectedModel = model;
    document.querySelectorAll(".model-button").forEach((item) => {
      const active = item.textContent === model;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", active ? "true" : "false");
    });
    renderResult();
  });
  modelSwitcher.appendChild(button);
});

Object.keys(benchmarkMeta).forEach((benchmark) => {
  const button = document.createElement("button");
  button.className = `benchmark-button${benchmark === selectedBenchmark ? " active" : ""}`;
  button.type = "button";
  button.textContent = benchmark;
  button.setAttribute("role", "tab");
  button.setAttribute("aria-selected", benchmark === selectedBenchmark ? "true" : "false");
  button.addEventListener("click", () => {
    selectedBenchmark = benchmark;
    document.querySelectorAll(".benchmark-button").forEach((item) => {
      const active = item.textContent === benchmark;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", active ? "true" : "false");
    });
    renderResult();
  });
  benchmarkList.appendChild(button);
});

function renderResult() {
  const [react, bestMemory, pg] = results[selectedModel][selectedBenchmark];
  const delta = pg - bestMemory;
  document.querySelector("#result-domain").textContent = benchmarkMeta[selectedBenchmark][0];
  document.querySelector("#result-metric").textContent = benchmarkMeta[selectedBenchmark][1];
  document.querySelector("#result-value").textContent = `${pg.toFixed(pg % 1 ? 2 : 0)}%`;
  document.querySelector("#result-delta").textContent =
    `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts vs. strongest memory baseline`;
  resultBars.innerHTML = "";
  [
    ["Vanilla ReAct", react, false],
    ["Best memory baseline", bestMemory, false],
    ["Procedural Graph", pg, true],
  ].forEach(([label, value, pgRow]) => {
    const row = document.createElement("div");
    row.className = `bar-row${pgRow ? " pg" : ""}`;
    row.innerHTML = `<span>${label}</span><div class="bar-track"><div class="bar-fill"></div></div><strong>${value.toFixed(value % 1 ? 2 : 0)}</strong>`;
    resultBars.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = `${value}%`;
    });
  });
}
renderResult();

const methodStages = {
  read: ["focus-read", "Localize the agent and translate its connected neighborhood into guidance."],
  log: ["focus-log", "Record the unfolding trajectory and its outcome as evidence for later refinement."],
  update: ["focus-update", "Propose topology edits; commit improvement and remember rejected mutations."],
};
document.querySelectorAll(".method-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".method-tab").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", active ? "true" : "false");
    });
    const [focusClass, guidance] = methodStages[button.dataset.stage];
    document.querySelector("#visual-focus").className = `visual-focus ${focusClass}`;
    document.querySelector("#guidance-text").textContent = guidance;
  });
});

const triplets = [
  ["search", "requires evidence", "analyze"],
  ["analyze", "enables", "compose"],
  ["cash forecast", "precedes", "fundraising"],
  ["test", "on failure returns to", "patch"],
];
let tripletIndex = 0;
document.querySelector("#next-triplet").addEventListener("click", () => {
  tripletIndex = (tripletIndex + 1) % triplets.length;
  const [source, relation, target] = triplets[tripletIndex];
  document.querySelector("#triplet-source").textContent = source;
  document.querySelector(".triplet i").textContent = relation;
  document.querySelector("#triplet-target").textContent = target;
});

const heroCanvas = document.querySelector("#hero-graph");
const heroContext = heroCanvas.getContext("2d");
const heroNodes = [
  [.12, .34, "Start"], [.35, .22, "Search"], [.63, .28, "Read"],
  [.48, .48, "Analyze"], [.77, .51, "Compose"], [.58, .68, "Verify"], [.82, .76, "End"],
];
const heroEdges = [[0,1],[1,2],[1,3],[2,3],[3,4],[3,5],[4,5],[5,6]];
let heroStep = 0;
let heroTimer;

function sizeCanvas(canvas, context) {
  const ratio = window.devicePixelRatio || 1;
  const box = canvas.getBoundingClientRect();
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return box;
}

function drawHero() {
  const box = sizeCanvas(heroCanvas, heroContext);
  heroContext.clearRect(0, 0, box.width, box.height);
  const points = heroNodes.map(([x, y]) => [x * box.width, y * box.height]);
  heroContext.lineWidth = 1.4;
  heroEdges.forEach(([from, to], index) => {
    const active = index < heroStep;
    heroContext.strokeStyle = active ? "#b9e769" : "rgba(255,255,255,.22)";
    heroContext.beginPath();
    heroContext.moveTo(...points[from]);
    heroContext.lineTo(...points[to]);
    heroContext.stroke();
  });
  points.forEach(([x, y], index) => {
    const active = index <= Math.min(heroStep, points.length - 1);
    heroContext.fillStyle = active ? "#b9e769" : "#173f2c";
    heroContext.strokeStyle = active ? "#b9e769" : "rgba(255,255,255,.55)";
    heroContext.lineWidth = active ? 3 : 1.5;
    heroContext.beginPath();
    heroContext.arc(x, y, active ? 9 : 7, 0, Math.PI * 2);
    heroContext.fill();
    heroContext.stroke();
    heroContext.fillStyle = active ? "#b9e769" : "rgba(255,255,255,.64)";
    heroContext.font = "11px monospace";
    heroContext.fillText(heroNodes[index][2], x + 14, y + 4);
  });
}

function replayHero() {
  clearInterval(heroTimer);
  heroStep = 0;
  drawHero();
  const guidance = [
    "Begin with the task objective.",
    "Search only for decision-relevant evidence.",
    "Read the local neighborhood around the active step.",
    "Analyze before composing.",
    "Compose a candidate response.",
    "Verify the result; avoid repeating search.",
    "Finish when the task is satisfied.",
  ];
  heroTimer = setInterval(() => {
    heroStep += 1;
    document.querySelector("#guidance-text").textContent = guidance[Math.min(heroStep, guidance.length - 1)];
    drawHero();
    if (heroStep >= 7) clearInterval(heroTimer);
  }, 650);
}
document.querySelector("#replay-graph").addEventListener("click", replayHero);

const horizonData = {
  "Claude 4.6": { base: 44, pg: 58, baseMonths: 89.8, pgMonths: 98.58 },
  "Gemini 3.1": { base: 6, pg: 34, baseMonths: 50.28, pgMonths: 79.22 },
  "Gemini 3.5": { base: 0, pg: 0, baseMonths: 33.58, pgMonths: 40.62 },
  "Grok 4.1": { base: 26, pg: 40, baseMonths: 63.76, pgMonths: 75.14 },
};
let horizonModel = "Gemini 3.1";
const horizonModels = document.querySelector("#horizon-models");
Object.keys(horizonData).forEach((model) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = model;
  button.className = model === horizonModel ? "active" : "";
  button.addEventListener("click", () => {
    horizonModel = model;
    [...horizonModels.children].forEach((item) => item.classList.toggle("active", item === button));
    renderSurvival();
  });
  horizonModels.appendChild(button);
});

const survivalCanvas = document.querySelector("#survival-canvas");
const survivalContext = survivalCanvas.getContext("2d");
const monthSlider = document.querySelector("#month-slider");

function survivalAt(month, finalValue, lifespan, pg = false) {
  const progress = month / 132;
  const firstCrisis = month >= 32 ? (pg ? .07 : .11) : 0;
  const secondCrisis = month >= 59 ? (pg ? .18 : .27) : 0;
  const thirdCrisis = month >= 112 ? (pg ? .12 : .16) : 0;
  const rawDecline = progress * (pg ? .24 : .21) + firstCrisis + secondCrisis + thirdCrisis;
  const fullDecline = (pg ? .24 : .21) + (pg ? .07 : .11) + (pg ? .18 : .27) + (pg ? .12 : .16);
  const capacityFactor = Math.min(1.12, Math.max(.82, 72 / lifespan));
  const normalized = Math.min(1, (rawDecline / fullDecline) * capacityFactor);
  if (month === 132) return finalValue;
  return 100 - (100 - finalValue) * normalized;
}

function drawSurvival() {
  const box = sizeCanvas(survivalCanvas, survivalContext);
  survivalContext.clearRect(0, 0, box.width, box.height);
  const data = horizonData[horizonModel];
  const selectedMonth = Number(monthSlider.value);
  const lines = [
    [false, data.base, data.baseMonths, "#ed8c3a"],
    [true, data.pg, data.pgMonths, "#b9e769"],
  ];
  lines.forEach(([pg, finalValue, lifespan, color]) => {
    survivalContext.strokeStyle = color;
    survivalContext.lineWidth = pg ? 3 : 2;
    survivalContext.setLineDash(pg ? [] : [6, 5]);
    survivalContext.beginPath();
    for (let month = 1; month <= selectedMonth; month += 1) {
      const x = (month / 132) * box.width;
      const y = box.height - (survivalAt(month, finalValue, lifespan, pg) / 100) * box.height;
      if (month === 1) survivalContext.moveTo(x, y); else survivalContext.lineTo(x, y);
    }
    survivalContext.stroke();
  });
  survivalContext.setLineDash([]);
}

function renderSurvival() {
  const data = horizonData[horizonModel];
  const month = Number(monthSlider.value);
  const base = Math.round(survivalAt(month, data.base, data.baseMonths, false));
  const pg = Math.round(survivalAt(month, data.pg, data.pgMonths, true));
  document.querySelector("#month-label").textContent = month;
  document.querySelector("#baseline-survival").textContent = `${base}%`;
  document.querySelector("#pg-survival").textContent = `${pg}%`;
  document.querySelector("#lifespan-gain").textContent = `+${(data.pgMonths - data.baseMonths).toFixed(1)} mo`;
  drawSurvival();
}
monthSlider.addEventListener("input", renderSurvival);

const evoModes = {
  baseline: {
    score: 87.5,
    delta: "starting point",
    copy: "Without a graph, the solver relies on its full flat trajectory at every step.",
    caption: "Flat trajectory · no procedural structure",
    graph: { nodes: [], edges: [] },
  },
  expert: {
    score: 58.93,
    delta: "−28.57 pts",
    copy: "A flawed expert topology can impose bottlenecks and reduce success.",
    caption: "Rigid prior · one failure-prone bottleneck",
    graph: {
      nodes: [
        { id: "start", label: "START", x: .07, y: .50, status: "endpoint" },
        { id: "read", label: "READ", x: .25, y: .50 },
        { id: "infer", label: "INFER", x: .43, y: .50 },
        { id: "edit", label: "EDIT", x: .61, y: .50, status: "bottleneck" },
        { id: "respond", label: "RESPOND", x: .79, y: .50 },
        { id: "end", label: "END", x: .95, y: .50, status: "endpoint" },
      ],
      edges: [
        ["start", "read"], ["read", "infer"], ["infer", "edit"],
        ["edit", "respond"], ["respond", "end"],
      ],
    },
  },
  repair: {
    score: 92.86,
    delta: "+33.93 pts over expert initialization",
    copy: "Online evolution locates and prunes the human-imposed bottlenecks.",
    caption: "Repaired prior · new checks bypass the bottleneck",
    graph: {
      nodes: [
        { id: "start", label: "START", x: .06, y: .50, status: "endpoint" },
        { id: "read", label: "READ", x: .23, y: .50 },
        { id: "infer", label: "INFER", x: .41, y: .28 },
        { id: "edit", label: "EDIT", x: .41, y: .72, status: "bottleneck" },
        { id: "verify", label: "VERIFY", x: .61, y: .28, status: "added" },
        { id: "repair", label: "REPAIR", x: .61, y: .72, status: "added" },
        { id: "respond", label: "RESPOND", x: .81, y: .50 },
        { id: "end", label: "END", x: .96, y: .50, status: "endpoint" },
      ],
      edges: [
        ["start", "read"], ["read", "infer"], ["read", "edit"],
        ["infer", "verify", "added"], ["edit", "repair", "added"],
        ["verify", "respond", "added"], ["repair", "respond", "added"],
        ["respond", "end"], ["infer", "respond", "pruned"],
      ],
    },
  },
  scratch: {
    score: 91.07,
    delta: "+3.57 pts over unguided",
    copy: "A graph bootstrapped without human prior becomes highly competitive.",
    caption: "Evolved from scratch · compact parallel procedure",
    graph: {
      nodes: [
        { id: "start", label: "START", x: .06, y: .50, status: "endpoint" },
        { id: "read", label: "READ", x: .24, y: .50, status: "added" },
        { id: "plan", label: "PLAN", x: .43, y: .28, status: "added" },
        { id: "act", label: "ACT", x: .43, y: .72, status: "added" },
        { id: "verify", label: "VERIFY", x: .63, y: .28, status: "added" },
        { id: "revise", label: "REVISE", x: .63, y: .72, status: "added" },
        { id: "respond", label: "RESPOND", x: .82, y: .50, status: "added" },
        { id: "end", label: "END", x: .97, y: .50, status: "endpoint" },
      ],
      edges: [
        ["start", "read", "added"], ["read", "plan", "added"],
        ["read", "act", "added"], ["plan", "verify", "added"],
        ["act", "revise", "added"], ["verify", "respond", "added"],
        ["revise", "respond", "added"], ["respond", "end", "added"],
      ],
    },
  },
};

const topologyCanvas = document.querySelector("#topology-canvas");
const topologyContext = topologyCanvas.getContext("2d");
let currentEvoMode = "baseline";
let topologyFrame;

function roundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.lineTo(x + width - r, y);
  context.quadraticCurveTo(x + width, y, x + width, y + r);
  context.lineTo(x + width, y + height - r);
  context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  context.lineTo(x + r, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - r);
  context.lineTo(x, y + r);
  context.quadraticCurveTo(x, y, x + r, y);
  context.closePath();
}

function drawArrow(context, from, to, status, opacity) {
  const color = status === "pruned" ? "#ed8c3a" : status === "added" ? "#1f7a4d" : "#8b948e";
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  const unitX = dx / distance;
  const unitY = dy / distance;
  const startX = from.x + unitX * from.radiusX;
  const startY = from.y + unitY * from.radiusY;
  const endX = to.x - unitX * (to.radiusX + 7);
  const endY = to.y - unitY * (to.radiusY + 7);
  const bend = Math.abs(dy) > 24 ? (dy > 0 ? 9 : -9) : 0;
  const controlX = (startX + endX) / 2;
  const controlY = (startY + endY) / 2 + bend;

  context.save();
  context.globalAlpha = opacity;
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = status === "added" ? 2.2 : 1.45;
  context.setLineDash(status === "pruned" ? [5, 5] : []);
  context.beginPath();
  context.moveTo(startX, startY);
  context.quadraticCurveTo(controlX, controlY, endX, endY);
  context.stroke();

  const angle = Math.atan2(endY - controlY, endX - controlX);
  context.setLineDash([]);
  context.beginPath();
  context.moveTo(endX, endY);
  context.lineTo(endX - 7 * Math.cos(angle - Math.PI / 6), endY - 7 * Math.sin(angle - Math.PI / 6));
  context.lineTo(endX - 7 * Math.cos(angle + Math.PI / 6), endY - 7 * Math.sin(angle + Math.PI / 6));
  context.closePath();
  context.fill();
  context.restore();
}

function drawTopology(modeName, progress = 1) {
  const mode = evoModes[modeName];
  const box = sizeCanvas(topologyCanvas, topologyContext);
  topologyContext.clearRect(0, 0, box.width, box.height);

  if (!mode.graph.nodes.length) {
    const centerX = box.width / 2;
    const centerY = box.height / 2 - 4;
    topologyContext.save();
    topologyContext.strokeStyle = "rgba(102,112,105,.36)";
    topologyContext.lineWidth = 1.5;
    topologyContext.setLineDash([6, 7]);
    topologyContext.beginPath();
    topologyContext.arc(centerX, centerY, Math.min(68, box.height * .29), 0, Math.PI * 2);
    topologyContext.stroke();
    topologyContext.setLineDash([]);
    topologyContext.fillStyle = "#667069";
    topologyContext.textAlign = "center";
    topologyContext.font = "500 11px monospace";
    topologyContext.fillText("NO GRAPH", centerX, centerY + 4);
    topologyContext.font = "9px sans-serif";
    topologyContext.fillStyle = "rgba(102,112,105,.72)";
    topologyContext.fillText("unstructured trajectory history", centerX, centerY + 25);
    topologyContext.restore();
    return;
  }

  const compact = box.width < 560;
  const nodeWidth = compact ? 50 : 62;
  const nodeHeight = compact ? 24 : 28;
  const paddingX = compact ? 28 : 36;
  const paddingY = 24;
  const nodeMap = new Map(mode.graph.nodes.map((node) => [
    node.id,
    {
      ...node,
      x: paddingX + node.x * (box.width - paddingX * 2),
      y: paddingY + node.y * (box.height - paddingY * 2),
      radiusX: nodeWidth / 2,
      radiusY: nodeHeight / 2,
    },
  ]));

  mode.graph.edges.forEach(([source, target, status = "retained"], index) => {
    const edgeProgress = Math.max(0, Math.min(1, progress * mode.graph.edges.length - index));
    if (edgeProgress > 0) drawArrow(topologyContext, nodeMap.get(source), nodeMap.get(target), status, edgeProgress);
  });

  mode.graph.nodes.forEach((node, index) => {
    const point = nodeMap.get(node.id);
    const nodeProgress = Math.max(0, Math.min(1, progress * mode.graph.nodes.length - index));
    if (!nodeProgress) return;
    const scale = .82 + nodeProgress * .18;
    const width = nodeWidth * scale;
    const height = nodeHeight * scale;
    const x = point.x - width / 2;
    const y = point.y - height / 2;
    const fill = node.status === "added" ? "#dff2b7"
      : node.status === "bottleneck" ? "#fff0e2"
      : node.status === "endpoint" ? "#173f2c" : "#fffdf8";
    const stroke = node.status === "bottleneck" ? "#ed8c3a" : "#1f7a4d";
    topologyContext.save();
    topologyContext.globalAlpha = nodeProgress;
    roundedRect(topologyContext, x, y, width, height, 7);
    topologyContext.fillStyle = fill;
    topologyContext.fill();
    topologyContext.strokeStyle = stroke;
    topologyContext.lineWidth = node.status === "added" ? 2 : 1.35;
    topologyContext.stroke();
    topologyContext.fillStyle = node.status === "endpoint" ? "#fffdf8" : "#173f2c";
    topologyContext.textAlign = "center";
    topologyContext.textBaseline = "middle";
    topologyContext.font = `${compact ? 7 : 8}px monospace`;
    topologyContext.fillText(node.label, point.x, point.y + .5);
    if (node.status === "bottleneck") {
      topologyContext.fillStyle = "#ed8c3a";
      topologyContext.beginPath();
      topologyContext.arc(x + width - 3, y + 3, 4, 0, Math.PI * 2);
      topologyContext.fill();
    }
    topologyContext.restore();
  });
}

function animateTopology(modeName) {
  cancelAnimationFrame(topologyFrame);
  const started = performance.now();
  const duration = 520;
  function frame(now) {
    const progress = Math.min(1, (now - started) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    drawTopology(modeName, eased);
    if (progress < 1) topologyFrame = requestAnimationFrame(frame);
  }
  topologyFrame = requestAnimationFrame(frame);
}

document.querySelectorAll(".evo-mode").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".evo-mode").forEach((item) => item.classList.toggle("active", item === button));
    const mode = evoModes[button.dataset.evo];
    currentEvoMode = button.dataset.evo;
    document.querySelector("#evo-score").textContent = `${mode.score.toFixed(2)}%`;
    document.querySelector("#evo-delta").textContent = mode.delta;
    document.querySelector("#evo-copy").textContent = mode.copy;
    document.querySelector("#topology-caption").textContent = mode.caption;
    animateTopology(currentEvoMode);
  });
});

function handleResize() {
  drawHero();
  drawSurvival();
  drawTopology(currentEvoMode);
}
window.addEventListener("resize", handleResize);
window.addEventListener("scroll", () => {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  document.querySelector("#scroll-progress").style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
}, { passive: true });

drawTopology(currentEvoMode);
renderSurvival();
replayHero();
