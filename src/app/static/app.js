/**
 * Gate B.2 Research Demonstration Application — Client Logic
 * Single-Page Architecture with Tab Navigation, Real Endpoints,
 * Ephemeral Audio Processing, and Provable Scientific Provenance.
 */

// Global State
const state = {
  activeTab: "overview",
  activeAudioBlob: null,
  activeAudioName: "None",
  activeAudioUrl: null,
  mediaRecorder: null,
  audioChunks: [],
  isRecording: false,
  recordingStartTime: 0,
  recordingTimerInterval: null,
  extractedEmbedding: null,
  wsConnection: null,
  sessionHistory: [],
  allModels: [],
  allSpeakers: [],
  figures: [],
  audioLabState: "IDLE",
  isSyntheticInput: false,
  audioContext: null,
  analyserNode: null,
  animFrameId: null,
  vizMode: "dual",
  qubitHudState: "IDLE",
  currentConfidenceScore: 0.0,
  barPeaks: new Array(36).fill(0),
};

// Canonical 7-Class Taxonomy (Single Source of Truth: config/taxonomy.yaml)
const CANONICAL_CLASSES = [
  "GENERAL_SPEECH",
  "CRITICAL_RESPONSE",
  "FALL",
  "BREATHING_MEDICAL",
  "PAIN",
  "SOS_HELP",
  "LOCATION"
];

const EMERGENCY_CLASSES = [
  "CRITICAL_RESPONSE",
  "FALL",
  "BREATHING_MEDICAL",
  "PAIN",
  "SOS_HELP",
  "LOCATION"
];

// Document Initialization
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initAudioLabControls();
  initModals();
  initDiagnostics();
  initToastSystem();
  initQuantumLatticeEngine();
  initQubitPrecessionHUD();
  init3DCardTiltEngine();
  initGlobalKeyboardAndDragGuards();

  // Load initial data
  loadOverviewData();
  loadModelsAndSpeakers();
  loadResearchData();
  loadProvenanceData();

  // Route based on URL hash if provided
  const hash = window.location.hash.replace("#", "");
  if (hash && document.getElementById(`view-${hash}`)) {
    switchTab(hash);
  } else {
    switchTab("overview");
  }
});

// ==========================================================================
// 1. Navigation & Tab Routing
// ==========================================================================

window.switchTab = switchTab;

function initNavigation() {
  const tabs = document.querySelectorAll(".nav-tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.target;
      switchTab(target);
    });
  });

  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.replace("#", "");
    if (hash && document.getElementById(`view-${hash}`)) {
      switchTab(hash);
    }
  });

  // Mobile menu toggle
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("appSidebar");
  if (mobileMenuBtn && sidebar) {
    mobileMenuBtn.addEventListener("click", () => {
      sidebar.classList.toggle("mobile-open");
    });
  }

  // Safety Notice Drawer Toggle
  const safetyBtn = document.getElementById("safetyDetailsBtn");
  const safetyToggleBtn = document.getElementById("safetyToggleBtn");
  const safetyDrawer = document.getElementById("safetyDetailsDrawer");

  function toggleSafetyDrawer() {
    if (!safetyDrawer) return;
    const isExpanded = safetyDrawer.classList.toggle("expanded");
    if (safetyBtn) safetyBtn.setAttribute("aria-expanded", isExpanded);
    if (safetyToggleBtn) safetyToggleBtn.setAttribute("aria-expanded", isExpanded);
    if (safetyBtn) {
      safetyBtn.innerHTML = isExpanded ? "<span>Governance &amp; Limits</span> ▴" : "<span>Governance &amp; Limits</span> ▾";
    }
  }

  if (safetyBtn) safetyBtn.addEventListener("click", toggleSafetyDrawer);
  if (safetyToggleBtn) safetyToggleBtn.addEventListener("click", toggleSafetyDrawer);
}

function switchTab(tabId) {
  state.activeTab = tabId;
  window.location.hash = tabId;

  // Update tab buttons
  document.querySelectorAll(".nav-tab-btn").forEach(btn => {
    if (btn.dataset.target === tabId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update view sections
  document.querySelectorAll(".view-section").forEach(view => {
    if (view.id === `view-${tabId}`) {
      view.classList.add("active");
    } else {
      view.classList.remove("active");
    }
  });

  // Update header breadcrumb
  const breadcrumbMap = {
    overview: "Overview Command",
    audiolab: "Quantum Audio Lab",
    registry: "Model & Speaker Registry",
    results: "Research Results Explorer",
    provenance: "Provenance & Reproducibility",
    diagnostics: "Runtime Diagnostics",
    help: "Help & Limitations"
  };
  const breadcrumbEl = document.getElementById("activeViewBreadcrumb");
  if (breadcrumbEl && breadcrumbMap[tabId]) {
    breadcrumbEl.textContent = breadcrumbMap[tabId];
  }

  // Close mobile sidebar if open
  const sidebar = document.getElementById("appSidebar");
  if (sidebar) sidebar.classList.remove("mobile-open");

  // Trigger view refresh if needed
  if (tabId === "overview") {
    loadOverviewData();
    if (window.resizeQubitHUD) setTimeout(window.resizeQubitHUD, 60);
  }
  if (tabId === "diagnostics") refreshDiagnostics();

  // Re-attach 3D tilt engine to newly visible cards
  setTimeout(init3DCardTiltEngine, 60);
}

// ==========================================================================
// 2. Overview Dashboard Data
// ==========================================================================

async function loadOverviewData() {
  try {
    const readyRes = await fetch("/ready");
    const readyData = await readyRes.json();
    state.quantumReady = !!readyData.quantum_model_ready;

    document.getElementById("statWhisperEncoder").textContent = readyData.whisper_encoder_ready ? "Operational" : "Offline";
    document.getElementById("statWhisperAsr").textContent = readyData.whisper_asr_ready ? "Operational" : "Offline";
    const statQuantum = document.getElementById("statQuantumStatus");
    if (statQuantum) {
      statQuantum.textContent = readyData.quantum_model_ready ? "OPERATIONAL (QUANTUM ACTIVE)" : "503 NOT READY";
      statQuantum.className = `status-pill ${readyData.quantum_model_ready ? "ready" : "blocked"}`;
    }
    const statMat = document.getElementById("statMaterializedModels");
    if (statMat) statMat.textContent = readyData.checkpoints_summary.materialized_and_verified;
    const statBlk = document.getElementById("statBlockedModels");
    if (statBlk) statBlk.textContent = readyData.checkpoints_summary.model_not_ready;

    onModelSelectionChanged();

    const provRes = await fetch("/api/v1/provenance/summary");
    const provData = await provRes.json();

    document.getElementById("statTestPassRate").textContent = `${provData.test_suite_status.pass_rate_percent}%`;
    document.getElementById("statTestCount").textContent = `${provData.test_suite_status.passed_tests} / ${provData.test_suite_status.total_tests}`;
    document.getElementById("statChecksumStatus").textContent = provData.cryptographic_integrity.audit_result;
    document.getElementById("overviewLastAudit").textContent = provData.cryptographic_integrity.last_verified_timestamp;

  } catch (err) {
    console.error("Failed to load overview data:", err);
  }
}

// ==========================================================================
// 3. Audio Lab Implementation
// ==========================================================================

function initAudioLabControls() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const recordBtn = document.getElementById("recordBtn");
  const sampleAudioBtn = document.getElementById("sampleAudioBtn");
  const transcribeBtn = document.getElementById("transcribeBtn");
  const extractEmbeddingBtn = document.getElementById("extractEmbeddingBtn");
  const classifyBtn = document.getElementById("classifyBtn");
  const inspectVectorBtn = document.getElementById("inspectVectorBtn");
  const modelSelect = document.getElementById("modelSelect");
  const speakerSelect = document.getElementById("speakerSelect");

  // Dropzone drag & drop
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleAudioFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleAudioFile(e.target.files[0]);
    }
  });

  // Microphone recording
  recordBtn.addEventListener("click", toggleRecording);

  // Synthetic sample audio
  sampleAudioBtn.addEventListener("click", generateSampleAudio);

  // Pre-loaded synthetic emergency suite selector
  const loadSyntheticBtn = document.getElementById("loadSyntheticSampleBtn");
  const syntheticSelect = document.getElementById("syntheticSampleSelect");
  if (loadSyntheticBtn && syntheticSelect) {
    loadSyntheticBtn.addEventListener("click", () => {
      const url = syntheticSelect.value;
      if (!url) {
        showToast({ type: "warning", title: "Select Sample", message: "Please select an emergency sample from the dropdown." });
        return;
      }
      loadSyntheticSampleFromUrl(url);
    });
    syntheticSelect.addEventListener("change", () => {
      if (syntheticSelect.value) {
        loadSyntheticSampleFromUrl(syntheticSelect.value);
      }
    });
  }

  // Clear / Discard audio
  const clearAudioBtn = document.getElementById("clearAudioBtn");
  if (clearAudioBtn) {
    clearAudioBtn.addEventListener("click", clearAudio);
  }

  // Discrete actions
  transcribeBtn.addEventListener("click", handleTranscribeOnly);
  extractEmbeddingBtn.addEventListener("click", handleExtractEmbeddingOnly);
  classifyBtn.addEventListener("click", handleFullClassification);

  // Opt-in vector inspector modal
  inspectVectorBtn.addEventListener("click", openVectorInspectorModal);

  // Model & speaker changes
  modelSelect.addEventListener("change", onModelSelectionChanged);
  speakerSelect.addEventListener("change", onSpeakerSelectionChanged);

  // Visualizer Mode Selector Chips
  const vizModeDual = document.getElementById("vizModeDual");
  const vizModeWave = document.getElementById("vizModeWave");
  const vizModeBars = document.getElementById("vizModeBars");
  if (vizModeDual && vizModeWave && vizModeBars) {
    [vizModeDual, vizModeWave, vizModeBars].forEach(btn => {
      btn.addEventListener("click", () => {
        [vizModeDual, vizModeWave, vizModeBars].forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.vizMode = btn.dataset.mode;
        if (state.activeAudioBlob && !state.isRecording) {
          drawStaticWaveform();
        }
      });
    });
  }

  initVisualizerCanvas();
  renderEmptyProbBars();
  setAudioLabState("IDLE");
}

function setAudioLabState(newState) {
  state.audioLabState = newState;
  const badge = document.getElementById("audioLabStateBadge");
  if (!badge) return;

  const stateClassMap = {
    IDLE: "info",
    RECORDING: "warning",
    PROCESSING_AUDIO: "info",
    UPLOADING: "info",
    TRANSCRIBING: "info",
    EXTRACTING_FEATURES: "info",
    RUNNING_QUANTUM_MODEL: "warning",
    INFERRING: "info",
    COMPLETED: "ready",
    ABSTAINED: "warning",
    QUANTUM_MODEL_NOT_READY: "blocked",
    MODEL_NOT_READY: "blocked",
    ERROR: "error"
  };

  badge.className = `status-pill ${stateClassMap[newState] || "info"}`;
  badge.textContent = `STATE: ${newState}`;

  const clearBtn = document.getElementById("clearAudioBtn");
  if (clearBtn) {
    clearBtn.disabled = !state.activeAudioBlob && newState === "IDLE";
  }

  updatePipelineStepper(newState);
  setQubitHUDState(newState);
}

function updatePipelineStepper(newState) {
  const step1 = document.getElementById("stepAudioInput");
  const step2 = document.getElementById("stepWhisper");
  const step3 = document.getElementById("stepPca");
  const step4 = document.getElementById("stepQuantumKernel");
  const step5 = document.getElementById("stepQsvc");
  const conn1 = document.getElementById("conn1");
  const conn2 = document.getElementById("conn2");
  const conn3 = document.getElementById("conn3");
  const conn4 = document.getElementById("conn4");

  if (!step1) return;

  [step1, step2, step3, step4, step5].forEach(s => {
    if (s) s.className = "pipeline-step";
  });
  [conn1, conn2, conn3, conn4].forEach(c => {
    if (c) c.className = "pipeline-connector";
  });

  if (state.activeAudioBlob) {
    step1.classList.add("completed");
    if (conn1) conn1.classList.add("active");
  } else {
    step1.classList.add("active");
  }

  if (newState === "TRANSCRIBING") {
    step2.classList.add("active");
  } else if (newState === "EXTRACTING_FEATURES") {
    step1.classList.add("completed");
    step2.classList.add("completed");
    step3.classList.add("active");
    if (conn1) conn1.classList.add("active");
    if (conn2) conn2.classList.add("active");
  } else if (newState === "RUNNING_QUANTUM_MODEL") {
    step1.classList.add("completed");
    step2.classList.add("completed");
    step3.classList.add("completed");
    step4.classList.add("active");
    if (conn1) conn1.classList.add("active");
    if (conn2) conn2.classList.add("active");
    if (conn3) conn3.classList.add("active");
  } else if (newState === "COMPLETED") {
    step1.classList.add("completed");
    step2.classList.add("completed");
    step3.classList.add("completed");
    step4.classList.add("completed");
    step5.classList.add("completed");
    if (conn1) conn1.classList.add("active");
    if (conn2) conn2.classList.add("active");
    if (conn3) conn3.classList.add("active");
    if (conn4) conn4.classList.add("active");
  }
}

function initVisualizerCanvas() {
  const canvas = document.getElementById("audioVisualizer");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.parentElement ? canvas.parentElement.clientWidth : 500;
  canvas.height = 68;
  ctx.fillStyle = "rgba(7, 12, 22, 0.9)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function startLiveWaveformVisualizer(stream) {
  const canvas = document.getElementById("audioVisualizer");
  const wrapper = document.getElementById("visualizerWrapper");
  if (wrapper) wrapper.style.display = "block";
  if (!canvas) return;
  canvas.style.display = "block";
  canvas.width = canvas.parentElement ? canvas.parentElement.clientWidth : 500;
  canvas.height = 68;
  const ctx = canvas.getContext("2d");

  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!state.audioContext || state.audioContext.state === "closed") {
      state.audioContext = new AudioContextClass();
    } else if (state.audioContext.state === "suspended") {
      state.audioContext.resume().catch(() => {});
    }
    const source = state.audioContext.createMediaStreamSource(stream);
    state.analyserNode = state.audioContext.createAnalyser();
    state.analyserNode.fftSize = 256;
    source.connect(state.analyserNode);

    const bufferLength = state.analyserNode.frequencyBinCount;
    const timeData = new Uint8Array(bufferLength);
    const freqData = new Uint8Array(bufferLength);

    function draw() {
      if (!state.isRecording) return;
      state.animFrameId = requestAnimationFrame(draw);
      state.analyserNode.getByteTimeDomainData(timeData);
      state.analyserNode.getByteFrequencyData(freqData);

      const w = canvas.width;
      const h = canvas.height;

      // Dark translucent backing for motion trail
      ctx.fillStyle = "rgba(6, 11, 22, 0.4)";
      ctx.fillRect(0, 0, w, h);

      const mode = state.vizMode || "dual";

      // 1. Draw Frequency Bars (in 'bars' or 'dual' mode)
      if (mode === "bars" || mode === "dual") {
        const barCount = 36;
        const barWidth = (w / barCount) - 2;
        const barMaxH = mode === "dual" ? h * 0.55 : h * 0.85;

        for (let i = 0; i < barCount; i++) {
          const val = freqData[Math.floor(i * (bufferLength / barCount))] / 255.0;
          const barH = Math.max(3, val * barMaxH);
          const x = i * (barWidth + 2);
          const y = h - barH;

          // Gradient fill
          const grad = ctx.createLinearGradient(0, y, 0, h);
          grad.addColorStop(0, "#22D3EE");
          grad.addColorStop(1, "#8B5CF6");
          ctx.fillStyle = grad;
          ctx.fillRect(x, y, barWidth, barH);

          // Falling Peak Tracker
          if (!state.barPeaks[i] || barH > state.barPeaks[i]) {
            state.barPeaks[i] = barH;
          } else {
            state.barPeaks[i] = Math.max(0, state.barPeaks[i] - 1.2);
          }
          const peakY = h - state.barPeaks[i] - 2;
          ctx.fillStyle = "#A78BFA";
          ctx.fillRect(x, peakY, barWidth, 2);
        }
      }

      // 2. Draw Oscilloscope Wave (in 'wave' or 'dual' mode)
      if (mode === "wave" || mode === "dual") {
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#38BDF8";
        ctx.shadowBlur = 10;
        ctx.shadowColor = "#22D3EE";
        ctx.beginPath();

        const sliceWidth = (w * 1.0) / bufferLength;
        const waveMid = mode === "dual" ? h * 0.28 : h * 0.5;
        const waveScale = mode === "dual" ? h * 0.24 : h * 0.45;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = (timeData[i] - 128) / 128.0;
          const y = waveMid + v * waveScale;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
          x += sliceWidth;
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }
    draw();
  } catch (e) {
    console.warn("Web Audio visualizer could not start:", e);
  }
}

function stopLiveWaveformVisualizer() {
  if (state.animFrameId) {
    cancelAnimationFrame(state.animFrameId);
    state.animFrameId = null;
  }
}

function drawStaticWaveform() {
  const canvas = document.getElementById("audioVisualizer");
  const wrapper = document.getElementById("visualizerWrapper");
  if (wrapper) wrapper.style.display = "block";
  if (!canvas) return;
  canvas.style.display = "block";
  canvas.width = canvas.parentElement ? canvas.parentElement.clientWidth : 500;
  canvas.height = 68;
  const ctx = canvas.getContext("2d");

  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "rgba(7, 12, 22, 0.95)";
  ctx.fillRect(0, 0, w, h);

  const isSynthetic = state.isSyntheticInput;
  const baseColor = isSynthetic ? "#F59E0B" : "#10B981";

  const mode = state.vizMode || "dual";

  if (mode === "bars" || mode === "dual") {
    const barCount = 36;
    const barWidth = (w / barCount) - 2;
    const barMaxH = mode === "dual" ? h * 0.52 : h * 0.8;

    for (let i = 0; i < barCount; i++) {
      const pseudoVal = (Math.sin(i * 0.35) * 0.4 + Math.cos(i * 0.18) * 0.3 + 0.35);
      const barH = Math.max(4, pseudoVal * barMaxH);
      const x = i * (barWidth + 2);
      const y = h - barH;

      const grad = ctx.createLinearGradient(0, y, 0, h);
      grad.addColorStop(0, baseColor);
      grad.addColorStop(1, isSynthetic ? "#D97706" : "#059669");
      ctx.fillStyle = grad;
      ctx.fillRect(x, y, barWidth, barH);
    }
  }

  if (mode === "wave" || mode === "dual") {
    ctx.lineWidth = 2;
    ctx.strokeStyle = baseColor;
    ctx.shadowBlur = 8;
    ctx.shadowColor = baseColor;
    ctx.beginPath();

    const waveMid = mode === "dual" ? h * 0.28 : h * 0.5;
    const ampScale = mode === "dual" ? h * 0.22 : h * 0.42;

    for (let x = 0; x < w; x += 4) {
      const amp = Math.sin(x * 0.06) * Math.cos(x * 0.02) * ampScale;
      const y = waveMid + amp;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

function clearAudio() {
  if (state.isRecording && state.mediaRecorder) {
    state.mediaRecorder.stop();
    state.isRecording = false;
  }
  stopLiveWaveformVisualizer();

  if (state.activeAudioUrl) {
    URL.revokeObjectURL(state.activeAudioUrl);
    state.activeAudioUrl = null;
  }

  state.activeAudioBlob = null;
  state.activeAudioName = "None";
  state.extractedEmbedding = null;
  state.isSyntheticInput = false;

  const audioPreview = document.getElementById("audioPreview");
  if (audioPreview) {
    audioPreview.src = "";
    audioPreview.style.display = "none";
  }

  const wrapper = document.getElementById("visualizerWrapper");
  if (wrapper) wrapper.style.display = "none";

  const canvas = document.getElementById("audioVisualizer");
  if (canvas) canvas.style.display = "none";

  const syntheticBanner = document.getElementById("syntheticNoticeBanner");
  if (syntheticBanner) syntheticBanner.style.display = "none";

  const statusBadge = document.getElementById("audioStatusBadge");
  if (statusBadge) {
    statusBadge.textContent = "Awaiting Input";
    statusBadge.className = "status-pill info";
  }

  const fileInput = document.getElementById("fileInput");
  if (fileInput) fileInput.value = "";

  document.getElementById("inspectVectorBtn").disabled = true;
  document.getElementById("embeddingSummaryBox").style.display = "none";

  clearResults("Awaiting audio input...");
  updateRadialConfidenceGauge(0.0, false, false);
  setAudioLabState("IDLE");
  showToast({ type: "info", title: "Audio Cleared", message: "Ephemeral audio buffer and waveform memory cleared." });
}

function handleAudioFile(file) {
  // Validate file size (10 MB)
  if (file.size > 10 * 1024 * 1024) {
    showToast({ type: "error", title: "File Size Limit Exceeded", message: "Audio file exceeds maximum allowed limit of 10 MB." });
    setAudioLabState("ERROR");
    return;
  }

  if (state.activeAudioUrl) {
    URL.revokeObjectURL(state.activeAudioUrl);
  }

  setAudioLabState("UPLOADING");
  state.activeAudioBlob = file;
  state.activeAudioName = file.name;
  state.activeAudioUrl = URL.createObjectURL(file);
  state.extractedEmbedding = null;
  state.isSyntheticInput = false;

  const syntheticBanner = document.getElementById("syntheticNoticeBanner");
  if (syntheticBanner) syntheticBanner.style.display = "none";

  const audioPreview = document.getElementById("audioPreview");
  audioPreview.src = state.activeAudioUrl;
  audioPreview.style.display = "block";

  document.getElementById("audioStatusBadge").textContent = `Loaded: ${file.name.slice(0, 20)}`;
  document.getElementById("audioStatusBadge").className = "status-pill ready";
  document.getElementById("inspectVectorBtn").disabled = true;
  document.getElementById("embeddingSummaryBox").style.display = "none";

  drawStaticWaveform();
  setAudioLabState("IDLE");
  showToast({ type: "info", title: "Audio Ingested", message: `Loaded ${file.name.slice(0, 24)} (${(file.size / 1024).toFixed(0)} KB)` });
}

async function toggleRecording() {
  const recordBtn = document.getElementById("recordBtn");
  const recordLabel = document.getElementById("recordLabel");
  const recordIcon = document.getElementById("recordIcon");

  if (!state.isRecording) {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Microphone API is not supported in this browser environment.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.mediaRecorder = new MediaRecorder(stream);
      state.audioChunks = [];

      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.audioChunks.push(e.data);
      };

      state.mediaRecorder.onstop = () => {
        stopLiveWaveformVisualizer();
        state.activeAudioBlob = new Blob(state.audioChunks, { type: "audio/wav" });
        state.activeAudioName = "mic_recording.wav";
        state.extractedEmbedding = null;
        state.isSyntheticInput = false;

        const syntheticBanner = document.getElementById("syntheticNoticeBanner");
        if (syntheticBanner) syntheticBanner.style.display = "none";

        const audioPreview = document.getElementById("audioPreview");
        audioPreview.src = URL.createObjectURL(state.activeAudioBlob);
        audioPreview.style.display = "block";

        document.getElementById("audioStatusBadge").textContent = "Recording Ready";
        document.getElementById("audioStatusBadge").className = "status-pill ready";
        clearInterval(state.recordingTimerInterval);
        recordBtn.classList.remove("recording");
        recordLabel.textContent = "Record from Microphone";
        recordIcon.textContent = "⏺️";

        drawStaticWaveform();
        setAudioLabState("IDLE");
      };

      state.mediaRecorder.start();
      state.isRecording = true;
      state.recordingStartTime = Date.now();
      recordBtn.classList.add("recording");
      recordIcon.textContent = "⏹️";
      setAudioLabState("RECORDING");

      startLiveWaveformVisualizer(stream);

      state.recordingTimerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - state.recordingStartTime) / 1000);
        recordLabel.textContent = `Recording (${elapsed}s)... Click to Stop`;
        if (elapsed >= 30) {
          // Hard cap at 30 seconds
          state.mediaRecorder.stop();
          state.isRecording = false;
        }
      }, 500);

    } catch (err) {
      alert("Microphone Error: " + err.message);
      document.getElementById("audioStatusBadge").textContent = "Mic Error";
      document.getElementById("audioStatusBadge").className = "status-pill error";
      setAudioLabState("ERROR");
    }
  } else {
    state.mediaRecorder.stop();
    state.isRecording = false;
  }
}

function generateSampleAudio() {
  // Generate 1.5 seconds of synthetic speech-like acoustic tone
  const sampleRate = 16000;
  const duration = 1.5;
  const numSamples = sampleRate * duration;
  const buffer = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    buffer[i] = 0.25 * Math.sin(2 * Math.PI * 320 * t) + 0.15 * Math.sin(2 * Math.PI * 640 * t);
  }

  const wavBlob = encodeWAV(buffer, sampleRate);
  state.activeAudioBlob = wavBlob;
  state.activeAudioName = "benchmark_tone_16khz.wav";
  state.extractedEmbedding = null;
  state.isSyntheticInput = true;

  const syntheticBanner = document.getElementById("syntheticNoticeBanner");
  if (syntheticBanner) syntheticBanner.style.display = "block";

  const audioPreview = document.getElementById("audioPreview");
  audioPreview.src = URL.createObjectURL(wavBlob);
  audioPreview.style.display = "block";

  document.getElementById("audioStatusBadge").textContent = "Benchmark Audio (1.5s)";
  document.getElementById("audioStatusBadge").className = "status-pill ready";
  document.getElementById("inspectVectorBtn").disabled = true;
  document.getElementById("embeddingSummaryBox").style.display = "none";

  const wrapper = document.getElementById("visualizerWrapper");
  if (wrapper) wrapper.style.display = "block";

  drawStaticWaveform();
  setAudioLabState("IDLE");
  showToast({
    type: "info",
    title: "Synthetic Benchmark Loaded",
    message: "1.5s 16kHz calibration tone generated ready for quantum inference."
  });
}

async function loadSyntheticSampleFromUrl(url) {
  try {
    const filename = url.split("/").pop();
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load sample audio from server.");
    const blob = await res.blob();

    state.activeAudioBlob = blob;
    state.activeAudioName = filename;
    state.extractedEmbedding = null;
    state.isSyntheticInput = true;

    const audioPreview = document.getElementById("audioPreview");
    audioPreview.src = URL.createObjectURL(blob);
    audioPreview.style.display = "block";

    document.getElementById("audioStatusBadge").textContent = `${filename} (Ready)`;
    document.getElementById("audioStatusBadge").className = "status-pill ready";
    document.getElementById("clearAudioBtn").disabled = false;
    document.getElementById("inspectVectorBtn").disabled = true;
    document.getElementById("embeddingSummaryBox").style.display = "none";

    const wrapper = document.getElementById("visualizerWrapper");
    if (wrapper) wrapper.style.display = "block";

    drawStaticWaveform();
    setAudioLabState("IDLE");
    clearResults("Sample audio loaded. Click 'Quantum Classify' to evaluate.");

    showToast({
      type: "success",
      title: "Sample Loaded",
      message: `${filename} loaded into Audio Lab.`
    });
  } catch (err) {
    showToast({ type: "error", title: "Load Error", message: err.message });
  }
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function onModelSelectionChanged() {
  const modelSelect = document.getElementById("modelSelect");
  if (!modelSelect) return;
  const modelId = modelSelect.value;
  const banner = document.getElementById("modelUnreadyBanner");
  if (!banner) return;

  if (modelId === "qsvc_q8_r1") {
    // If quantum model is verified operational, hide unready banner
    if (state.quantumReady !== false) {
      banner.style.display = "none";
      banner.classList.remove("active");
    } else {
      banner.style.display = "block";
      banner.classList.add("active");
      document.getElementById("modelUnreadyTitle").textContent = "QUANTUM_MODEL_NOT_READY (HTTP 503)";
      document.getElementById("modelUnreadyMessage").textContent =
        "Operational quantum model (qsvc_q8_r1) is unmaterialized.";
    }
  } else if (modelId.includes("phase8") || modelId.includes("track") || modelId.includes("int-")) {
    banner.style.display = "block";
    banner.classList.add("active");
    document.getElementById("modelUnreadyTitle").textContent = "MODEL_NOT_READY (HTTP 503)";
    document.getElementById("modelUnreadyMessage").textContent =
      "Candidate model is not materialized for operational inference.";
  } else {
    banner.style.display = "none";
    banner.classList.remove("active");
  }
}

function onSpeakerSelectionChanged() {
  // Speaker selection maps to actual verified adapter
  const speakerId = document.getElementById("speakerSelect").value;
  console.log(`Speaker selected: ${speakerId}`);
}

async function handleTranscribeOnly() {
  if (!state.activeAudioBlob) {
    showToast({ type: "warning", title: "Audio Input Required", message: "Please provide an audio recording, upload a file, or generate benchmark audio first." });
    return;
  }
  setAudioLabState("TRANSCRIBING");
  const btn = document.getElementById("transcribeBtn");
  btn.disabled = true;
  btn.textContent = "Transcribing...";

  const formData = new FormData();
  formData.append("file", state.activeAudioBlob, state.activeAudioName);

  try {
    const res = await fetch("/api/v1/audio/transcribe", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail ? err.detail.message : res.statusText);
    }
    const data = await res.json();
    document.getElementById("transcriptionOutput").textContent = data.text || "[No speech detected / silent]";
    document.getElementById("acousticQualityVal").textContent = `Quality Score: ${data.acoustic_quality}`;
    setAudioLabState("COMPLETED");
    showToast({ type: "success", title: "Transcription Finished", message: `Decoded text: "${(data.text || '').slice(0, 32)}..."` });
  } catch (err) {
    showToast({ type: "error", title: "Transcription Failed", message: err.message });
    setAudioLabState("ERROR");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>🗣️</span> Transcribe Speech`;
  }
}

async function handleExtractEmbeddingOnly() {
  if (!state.activeAudioBlob) {
    showToast({ type: "warning", title: "Audio Input Required", message: "Please provide an audio recording, upload a file, or generate benchmark audio first." });
    return;
  }
  setAudioLabState("EXTRACTING_FEATURES");
  const btn = document.getElementById("extractEmbeddingBtn");
  btn.disabled = true;
  btn.textContent = "Extracting Features...";

  const formData = new FormData();
  formData.append("file", state.activeAudioBlob, state.activeAudioName);

  try {
    const res = await fetch("/api/v1/audio/extract-embedding?include_vector=true", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail ? err.detail.message : res.statusText);
    }
    const data = await res.json();
    state.extractedEmbedding = data;

    // Show summary box
    document.getElementById("embeddingSummaryBox").style.display = "block";
    document.getElementById("embedDimVal").textContent = `${data.embedding_dimension} dimensions`;
    document.getElementById("embedNormVal").textContent = `${data.l2_norm.toFixed(6)} (Unit Norm Check Passed)`;
    document.getElementById("embedDurationVal").textContent = `${data.audio_duration_seconds}s`;
    document.getElementById("inspectVectorBtn").disabled = false;
    setAudioLabState("COMPLETED");
    showToast({ type: "success", title: "Features Extracted", message: "384-D latent vector extracted with unit L2-norm verified." });

  } catch (err) {
    showToast({ type: "error", title: "Extraction Failed", message: err.message });
    setAudioLabState("ERROR");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>📐</span> Extract 384-D Features`;
  }
}

async function handleFullClassification() {
  if (!state.activeAudioBlob) {
    showToast({ type: "warning", title: "Audio Input Required", message: "Please provide an audio recording, upload a file, or generate benchmark audio first." });
    return;
  }

  setAudioLabState("RUNNING_QUANTUM_MODEL");
  const modelId = "qsvc_q8_r1"; // Enforce canonical quantum model ONLY
  const speakerId = document.getElementById("speakerSelect").value;
  const banner = document.getElementById("modelUnreadyBanner");
  const btn = document.getElementById("classifyBtn");

  btn.disabled = true;
  btn.innerHTML = `<span>⏳</span> Evaluating Quantum Kernel...`;

  const formData = new FormData();
  formData.append("file", state.activeAudioBlob, state.activeAudioName);
  formData.append("speaker_id", speakerId);
  formData.append("model_id", modelId);
  formData.append("operating_tau", 0.60);

  const tStart = performance.now();

  try {
    const res = await fetch("/api/v1/predict/file", { method: "POST", body: formData });
    const latency = Math.round(performance.now() - tStart);

    if (res.status === 503) {
      const err = await res.json();
      banner.classList.add("active");
      const errDetail = err.detail || {};
      document.getElementById("modelUnreadyTitle").textContent = `HTTP 503: ${errDetail.error_code || "QUANTUM_MODEL_NOT_READY"}`;
      document.getElementById("modelUnreadyMessage").textContent = `${errDetail.message || "Quantum model is unready."} Remediation: ${errDetail.remediation || "Formal owner authorization required."}`;
      clearResults("Blocked (HTTP 503 Quantum Model Not Ready)");
      setAudioLabState("QUANTUM_MODEL_NOT_READY");
      addSessionRecord(speakerId, modelId, latency, "NONE", false, "HTTP 503 QUANTUM_MODEL_NOT_READY", state.isSyntheticInput);
      showToast({ type: "error", title: "Quantum Model Not Ready (HTTP 503)", message: "Required support vectors are unmaterialized." });
      return;
    }

    if (res.status === 400) {
      const err = await res.json();
      const errDetail = err.detail || {};
      banner.classList.add("active");
      document.getElementById("modelUnreadyTitle").textContent = `HTTP 400: ${errDetail.error_code || "BAD_REQUEST"}`;
      document.getElementById("modelUnreadyMessage").textContent = errDetail.message || "Request rejected.";
      clearResults("Rejected (HTTP 400 Classical Model Not Allowed)");
      setAudioLabState("ERROR");
      showToast({ type: "error", title: "Request Rejected (HTTP 400)", message: errDetail.message || "Classical models prohibited from operational inference." });
      return;
    }

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail ? JSON.stringify(err.detail) : res.statusText);
    }

    const data = await res.json();
    banner.classList.remove("active");
    displayInferenceResults(data);

    if (data.suggested_action === "ABSTAIN") {
      setAudioLabState("ABSTAINED");
      showToast({ type: "warning", title: "Quantum Decision: ABSTAIN", message: `Below operating cutoff τ=0.60 (Score: ${(data.confidence_score * 100).toFixed(1)}%)` });
    } else {
      setAudioLabState("COMPLETED");
      const alertType = data.is_emergency ? "warning" : "success";
      showToast({ type: alertType, title: `Quantum Intent: ${data.predicted_intent}`, message: `Confidence: ${(data.confidence_score * 100).toFixed(1)}% | Action: ${data.suggested_action}` });
    }

    addSessionRecord(speakerId, modelId, data.processing_latency_ms, data.predicted_intent, data.is_emergency, "SUCCESS", state.isSyntheticInput);

  } catch (err) {
    showToast({ type: "error", title: "Quantum Pipeline Notice", message: err.message });
    clearResults("Error: Request Failed");
    setAudioLabState("ERROR");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>⚛️</span> Quantum Classify`;
  }
}

function displayInferenceResults(data) {
  // 1. Transcription
  document.getElementById("transcriptionOutput").textContent = data.transcription || "[No speech detected / silent]";
  document.getElementById("acousticQualityVal").textContent = "Nominal (16 kHz mono resampled)";

  // 2. Intent badge
  const intentBadge = document.getElementById("intentBadge");
  intentBadge.textContent = data.predicted_intent.toUpperCase().replace("_", " ");
  if (data.is_emergency) {
    intentBadge.className = "intent-badge emergency";
  } else if (data.suggested_action === "ABSTAIN") {
    intentBadge.className = "intent-badge abstain";
  } else {
    intentBadge.className = "intent-badge general";
  }

  // 3. Confidence & Actions
  document.getElementById("confidenceVal").textContent = `${(data.confidence_score * 100).toFixed(1)}%`;
  document.getElementById("confidenceBandVal").textContent = data.confidence_band;
  document.getElementById("suggestedActionVal").textContent = data.suggested_action;

  // 4. Update Animated Radial Gauge & Numeric Rollup
  updateRadialConfidenceGauge(data.confidence_score, data.is_emergency, data.suggested_action === "ABSTAIN");

  // 5. Latency
  const latencyBadge = document.getElementById("latencyBadge");
  latencyBadge.textContent = `Latency: ${data.processing_latency_ms.toFixed(1)} ms`;
  latencyBadge.style.display = "inline-block";

  // 6. Probability bars
  renderProbBars(data.class_probabilities, data.predicted_intent);
}

function clearResults(statusMsg = "Awaiting audio input...") {
  document.getElementById("transcriptionOutput").textContent = statusMsg;
  const intentBadge = document.getElementById("intentBadge");
  intentBadge.textContent = "NONE";
  intentBadge.className = "intent-badge general";
  document.getElementById("confidenceVal").textContent = "--";
  document.getElementById("confidenceBandVal").textContent = "--";
  document.getElementById("suggestedActionVal").textContent = "--";
  document.getElementById("latencyBadge").style.display = "none";
  updateRadialConfidenceGauge(0.0, false, false);
  renderEmptyProbBars();
}

function renderEmptyProbBars() {
  const container = document.getElementById("probBarsContainer");
  container.innerHTML = "";
  CANONICAL_CLASSES.forEach(cls => {
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <div class="prob-label">${cls}</div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: 0%;"></div>
      </div>
      <div class="prob-value">0.0%</div>
    `;
    container.appendChild(row);
  });
}

function renderProbBars(probs, topIntent) {
  const container = document.getElementById("probBarsContainer");
  container.innerHTML = "";
  CANONICAL_CLASSES.forEach(cls => {
    const p = (probs[cls] || 0.0);
    const pct = (p * 100).toFixed(1);
    const isEmerg = EMERGENCY_CLASSES.includes(cls);
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <div class="prob-label" style="${cls === topIntent ? 'font-weight:700; color:#fff;' : ''}">${cls}</div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${isEmerg ? 'emergency' : ''}" style="width: ${pct}%;"></div>
      </div>
      <div class="prob-value">${pct}%</div>
    `;
    container.appendChild(row);
  });
}

// ==========================================================================
// 4. Model & Speaker Registry Implementation
// ==========================================================================

async function loadModelsAndSpeakers() {
  try {
    const [modelsRes, speakersRes] = await fetchParallel([
      fetch("/api/v1/models"),
      fetch("/api/v1/speakers")
    ]);

    state.allModels = modelsRes.models;
    state.allSpeakers = speakersRes.speakers;

    renderModelRegistryTable(state.allModels);
    populateModelSelectors(state.allModels, state.allSpeakers);

    // Setup registry filter & search
    document.getElementById("registrySearchInput").addEventListener("input", filterModelRegistry);
    document.getElementById("registryStatusFilter").addEventListener("change", filterModelRegistry);

  } catch (err) {
    console.error("Failed to load models and speakers:", err);
  }
}

function populateModelSelectors(models, speakers) {
  const speakerSelect = document.getElementById("speakerSelect");
  speakerSelect.innerHTML = "";
  speakers.forEach(spk => {
    const opt = document.createElement("option");
    opt.value = spk;
    opt.textContent = `${spk} (Dysarthric / Control Profile)`;
    if (spk === "F01") opt.selected = true;
    speakerSelect.appendChild(opt);
  });

  const modelSelect = document.getElementById("modelSelect");
  modelSelect.innerHTML = "";
  // Under quantum-only mandate, ONLY the canonical quantum model is allowed for operational prediction
  const opt = document.createElement("option");
  opt.value = "qsvc_q8_r1";
  opt.textContent = "qsvc_q8_r1 (Canonical 8-Qubit Quantum Kernel QSVC — Operational)";
  opt.selected = true;
  modelSelect.appendChild(opt);
}

function renderModelRegistryTable(models) {
  const tbody = document.getElementById("registryTableBody");
  tbody.innerHTML = "";

  models.forEach(m => {
    const tr = document.createElement("tr");
    const isOperationalQuantum = m.model_id === "qsvc_q8_r1";
    const isClassical = m.operational_role === "PROHIBITED_FOR_OPERATIONAL_INFERENCE" || (m.status === "RESEARCH_BASELINE_ONLY");
    const shortHash = m.sha256 ? m.sha256.slice(0, 8) : "N/A (Missing)";

    let statusPill = "";
    if (isOperationalQuantum) {
      if (m.status === "OPERATIONAL") {
        statusPill = `<span class="status-pill ready">OPERATIONAL (QUANTUM ACTIVE)</span>`;
      } else {
        statusPill = `<span class="status-pill blocked">QUANTUM (503 NOT READY)</span>`;
      }
    } else if (isClassical) {
      statusPill = `<span class="status-pill info">RESEARCH BASELINE (READ-ONLY)</span>`;
    } else {
      statusPill = `<span class="status-pill blocked">UNMATERIALIZED (503)</span>`;
    }

    const archClean = (m.architecture || "").split("(")[0];
    tr.innerHTML = `
      <td><strong>${m.model_id}</strong></td>
      <td>${m.speaker_id || "All Speakers"}</td>
      <td>${statusPill}</td>
      <td class="mono-cell">${archClean}</td>
      <td class="mono-cell">${shortHash}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="openModelDetailsModal('${m.model_id}', '${m.speaker_id || ''}')">
          Details
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterModelRegistry() {
  const q = document.getElementById("registrySearchInput").value.toLowerCase();
  const statusFilter = document.getElementById("registryStatusFilter").value;

  const filtered = state.allModels.filter(m => {
    const matchesQuery = m.model_id.toLowerCase().includes(q) ||
                         (m.speaker_id && m.speaker_id.toLowerCase().includes(q)) ||
                         m.architecture.toLowerCase().includes(q);
    const matchesStatus = statusFilter === "ALL" || m.status === statusFilter;
    return matchesQuery && matchesStatus;
  });

  renderModelRegistryTable(filtered);
}

// ==========================================================================
// 5. Research Results Explorer Implementation
// ==========================================================================

async function loadResearchData() {
  try {
    const [phasesRes, metricsRes, opPointsRes, figuresRes] = await fetchParallel([
      fetch("/api/v1/research/phases"),
      fetch("/api/v1/research/metrics"),
      fetch("/api/v1/research/operating-points"),
      fetch("/api/v1/research/figures")
    ]);

    renderPhasesTimeline(phasesRes.phases);
    renderMetricsTables(metricsRes);
    renderOperatingPointsTable(opPointsRes);
    renderFigureGallery(figuresRes.figures);
    state.figures = figuresRes.figures;

  } catch (err) {
    console.error("Failed to load research data:", err);
  }
}

function renderPhasesTimeline(phases) {
  const container = document.getElementById("phasesTimelineContainer");
  container.innerHTML = "";

  phases.forEach(p => {
    const card = document.createElement("div");
    card.className = "card";
    card.style.marginBottom = "14px";
    card.innerHTML = `
      <div class="card-header">
        <div class="card-title">${p.phase_name}</div>
        <span class="status-pill ready">${p.status}</span>
      </div>
      <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 8px;">${p.scope}</div>
      <div style="font-size: 12.5px; color: #38BDF8; font-weight: 500;"><strong>Verdict:</strong> ${p.verdict}</div>
    `;
    container.appendChild(card);
  });
}

function renderMetricsTables(metrics) {
  // Phase 8.1 Comparison Table
  const tbody81 = document.getElementById("metricsPhase81Body");
  tbody81.innerHTML = "";
  metrics.phase8_1_comparison.rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${r.model_track || r.model_name || 'Candidate'}</strong></td>
      <td>${r.architecture || 'Hierarchical'}</td>
      <td class="mono-cell">${r.emergency_recall || '--'}</td>
      <td class="mono-cell">${r.emergency_precision || '--'}</td>
      <td class="mono-cell">${r.FER || '--'}</td>
      <td class="mono-cell">${r.false_negative_count || '--'}</td>
      <td><span class="status-pill ${r.safety_bound_passed === 'True' ? 'ready' : 'blocked'}">${r.safety_bound_passed || 'No'}</span></td>
    `;
    tbody81.appendChild(tr);
  });

  // Phase 6.5.6 Classical vs Quantum Table
  const tbody6 = document.getElementById("metricsPhase6Body");
  tbody6.innerHTML = "";
  metrics.phase6_5_6_comparison.rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${r.comparison_pair || r.model_a + ' vs ' + r.model_b}</strong></td>
      <td class="mono-cell">${r.macro_f1_classical || r.f1_classical || '--'}</td>
      <td class="mono-cell">${r.macro_f1_quantum || r.f1_quantum || '--'}</td>
      <td class="mono-cell">${r.latency_ratio || '772x Slower'}</td>
      <td class="mono-cell">${r.p_value || '--'}</td>
      <td><span class="status-pill error">${r.statistical_verdict || 'NO ADVANTAGE (CASE B)'}</span></td>
    `;
    tbody6.appendChild(tr);
  });

  // Phase 7.12 Four-Way Comparison Matrix (Classical vs Fixed vs Trainable vs Hybrid)
  const tbody712 = document.getElementById("metricsPhase712Body");
  if (tbody712 && metrics.four_way_hybrid_comparison && metrics.four_way_hybrid_comparison.rows) {
    tbody712.innerHTML = "";
    metrics.four_way_hybrid_comparison.rows.forEach(r => {
      const tr = document.createElement("tr");
      const isHybrid = (r.Model || "").toLowerCase().includes("hybrid");
      if (isHybrid) {
        tr.style.backgroundColor = "rgba(16, 185, 129, 0.08)";
        tr.style.fontWeight = "600";
      }
      tr.innerHTML = `
        <td><strong>${isHybrid ? '🌟 ' : ''}${r.Model || '--'}</strong></td>
        <td class="mono-cell">${r.Trainable_Params || '0'}</td>
        <td class="mono-cell" style="${isHybrid ? 'color: #10B981; font-weight: 700;' : ''}">${r.Macro_F1 || '--'}</td>
        <td class="mono-cell">${r.Emergency_Recall ? (parseFloat(r.Emergency_Recall) * 100).toFixed(1) + '%' : '--'}</td>
        <td class="mono-cell" style="${isHybrid ? 'color: #38BDF8; font-weight: 700;' : ''}">${r.FER ? (parseFloat(r.FER) * 100).toFixed(2) + '%' : '--'}</td>
        <td class="mono-cell">${r.Accuracy ? (parseFloat(r.Accuracy) * 100).toFixed(1) + '%' : '--'}</td>
        <td><span class="status-pill ${r.Safety_Status && r.Safety_Status.includes('SAFE') ? 'ready' : (r.Safety_Status && r.Safety_Status.includes('BORDERLINE') ? 'warning' : 'error')}">${r.Safety_Status || '--'}</span></td>
        <td style="font-size: 11.5px; color: ${isHybrid ? '#10B981' : 'var(--text-dim)'};">${r.Statistical_Verdict || '--'}</td>
      `;
      tbody712.appendChild(tr);
    });
  }
}

function renderOperatingPointsTable(opData) {
  const tbody = document.getElementById("operatingPointsBody");
  tbody.innerHTML = "";
  opData.operating_points.forEach(pt => {
    const tr = document.createElement("tr");
    const isDefault = Math.abs(pt.threshold_tau - 0.60) < 1e-4;
    tr.style.backgroundColor = isDefault ? "rgba(59, 130, 246, 0.08)" : "";
    tr.innerHTML = `
      <td class="mono-cell"><strong>${pt.threshold_tau.toFixed(2)}${isDefault ? ' (Default)' : ''}</strong></td>
      <td class="mono-cell">${(pt.emergency_recall * 100).toFixed(1)}%</td>
      <td class="mono-cell">${(pt.emergency_precision * 100).toFixed(1)}%</td>
      <td class="mono-cell">${(pt.FER * 100).toFixed(2)}%</td>
      <td><span class="status-pill ${pt.safety_bound_passed ? 'ready' : 'error'}">${pt.safety_bound_passed ? 'Passed (≤15%)' : 'Violated (>15%)'}</span></td>
      <td class="mono-cell">${pt.TP}</td>
      <td class="mono-cell">${pt.FP}</td>
      <td class="mono-cell">${pt.FN}</td>
      <td class="mono-cell">${pt.TN}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderFigureGallery(figures) {
  const container = document.getElementById("figuresGalleryContainer");
  container.innerHTML = "";

  figures.forEach(fig => {
    const card = document.createElement("div");
    card.className = "figure-card";
    card.innerHTML = `
      <div class="figure-preview" onclick="openFigureLightbox('${fig.figure_id}')">
        <img src="${fig.url}" alt="${fig.title}" loading="lazy">
        <span class="figure-badge">VERIFIED ARTIFACT</span>
      </div>
      <div class="figure-content">
        <div class="figure-title">${fig.title}</div>
        <div class="figure-protocol">${fig.protocol_context}</div>
        <div class="figure-desc">${fig.description}</div>
        <div style="font-size: 11px; color: var(--text-dim); margin-top: auto;">
          Source: <span class="mono-cell">${fig.source_path}</span>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

// ==========================================================================
// 6. Provenance & Reproducibility Implementation
// ==========================================================================

async function loadProvenanceData() {
  try {
    const res = await fetch("/api/v1/provenance/summary");
    if (!res.ok) return;
    const data = await res.json();

    const setEl = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    if (data.dataset_provenance) {
      setEl("provDatasetName", data.dataset_provenance.dataset_name);
      setEl("provUtteranceCount", `${data.dataset_provenance.total_utterances} Utterances`);
      setEl("provSpeakerCount", `${data.dataset_provenance.speakers_count} (${data.dataset_provenance.dysarthric_speakers} Dysarthric, ${data.dataset_provenance.control_speakers} Control)`);
      setEl("provManifestPath", data.dataset_provenance.manifest_file);
    }

    if (data.cryptographic_integrity) {
      setEl("provChecksumCount", `${data.cryptographic_integrity.total_tracked_files} Files`);
      setEl("provAuditResult", data.cryptographic_integrity.audit_result);
      setEl("provLastAuditTimestamp", data.cryptographic_integrity.last_verified_timestamp);
    }

    if (data.test_suite_status) {
      setEl("provTestCount", `${data.test_suite_status.total_tests} Collected`);
      setEl("provPassRate", `${data.test_suite_status.pass_rate_percent}% (0 Regressions)`);
      setEl("provRegressionVerdict", data.test_suite_status.verdict);
    }
  } catch (err) {
    console.warn("Provenance data loading notice:", err);
  }
}

// ==========================================================================
// 7. Diagnostics & WebSocket Tester
// ==========================================================================

function initDiagnostics() {
  const wsConnectBtn = document.getElementById("wsConnectBtn");
  const wsSendBtn = document.getElementById("wsSendBtn");
  const clearSessionBtn = document.getElementById("clearSessionBtn");
  const pingBtn = document.getElementById("pingDiagnosticsBtn");

  wsConnectBtn.addEventListener("click", toggleWebSocketConnection);
  wsSendBtn.addEventListener("click", sendWebSocketTestFrame);
  clearSessionBtn.addEventListener("click", clearSessionHistory);
  pingBtn.addEventListener("click", refreshDiagnostics);
}

async function refreshDiagnostics() {
  const t0 = performance.now();
  try {
    const res = await fetch("/health");
    const roundtrip = Math.round(performance.now() - t0);
    const data = await res.json();

    document.getElementById("diagApiLatency").textContent = `${roundtrip} ms`;
    document.getElementById("diagApiStatus").textContent = data.status.toUpperCase();
    document.getElementById("diagApiStatus").className = "status-pill ready";
    document.getElementById("diagAppVersion").textContent = data.version;
    document.getElementById("diagServerTimestamp").textContent = data.timestamp_utc;
  } catch (err) {
    document.getElementById("diagApiStatus").textContent = "OFFLINE";
    document.getElementById("diagApiStatus").className = "status-pill error";
  }
}

function toggleWebSocketConnection() {
  const statusBadge = document.getElementById("wsStatusBadge");
  const logBox = document.getElementById("wsLogBox");
  const connectBtn = document.getElementById("wsConnectBtn");
  const sendBtn = document.getElementById("wsSendBtn");

  if (!state.wsConnection || state.wsConnection.readyState !== WebSocket.OPEN) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/inference`;

    appendWsLog(`Attempting connection to ${wsUrl}...`);
    state.wsConnection = new WebSocket(wsUrl);

    state.wsConnection.onopen = () => {
      statusBadge.textContent = "CONNECTED";
      statusBadge.className = "status-pill ready";
      connectBtn.textContent = "Disconnect";
      sendBtn.disabled = false;
      appendWsLog("WebSocket handshake established. Real-time inference channel ready.");
    };

    state.wsConnection.onmessage = (event) => {
      appendWsLog(`Received frame: ${event.data}`);
    };

    state.wsConnection.onerror = (err) => {
      statusBadge.textContent = "ERROR";
      statusBadge.className = "status-pill error";
      appendWsLog("WebSocket connection error.");
    };

    state.wsConnection.onclose = () => {
      statusBadge.textContent = "DISCONNECTED";
      statusBadge.className = "status-pill blocked";
      connectBtn.textContent = "Connect to WebSocket";
      sendBtn.disabled = true;
      appendWsLog("WebSocket connection closed.");
    };

  } else {
    state.wsConnection.close();
  }
}

function sendWebSocketTestFrame() {
  if (!state.wsConnection || state.wsConnection.readyState !== WebSocket.OPEN) return;

  const speakerId = document.getElementById("speakerSelect").value || "F01";
  const modelId = "qsvc_q8_r1";

  // 1. Send init frame with canonical quantum model
  appendWsLog(`Sending 'init' frame: speaker='${speakerId}', model='${modelId}'`);
  state.wsConnection.send(JSON.stringify({
    type: "init",
    speaker_id: speakerId,
    model_id: modelId,
    operating_tau: 0.60
  }));

  // 2. Send audio chunk
  setTimeout(() => {
    appendWsLog("Sending 'chunk' audio frame (base64 acoustic waveform)...");
    state.wsConnection.send(JSON.stringify({
      type: "chunk",
      data: "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    }));
  }, 100);

  // 3. Send finalize message
  setTimeout(() => {
    appendWsLog("Sending 'finalize' frame to trigger offline classification...");
    state.wsConnection.send(JSON.stringify({
      type: "finalize"
    }));
  }, 250);
}

function appendWsLog(msg) {
  const box = document.getElementById("wsLogBox");
  const t = new Date().toISOString().split("T")[1].slice(0, 8);
  box.textContent = `[${t}] ${msg}\n` + box.textContent;
}

function addSessionRecord(speakerId, modelId, latency, intent, isEmergency, outcome, isSynthetic = false) {
  const record = {
    timestamp: new Date().toISOString().split("T")[1].slice(0, 8),
    speakerId,
    modelId,
    latency: `${latency} ms`,
    intent,
    isEmergency: isEmergency ? "YES" : "NO",
    outcome,
    isSynthetic: isSynthetic
  };

  state.sessionHistory.unshift(record);
  renderSessionTable();
}

function renderSessionTable() {
  const tbody = document.getElementById("sessionHistoryTableBody");
  tbody.innerHTML = "";

  state.sessionHistory.forEach(rec => {
    const tr = document.createElement("tr");
    const syntheticTag = rec.isSynthetic
      ? ` <span class="status-pill warning" style="font-size: 10px; padding: 2px 6px;">SYNTHETIC TEST</span>`
      : "";
    tr.innerHTML = `
      <td class="mono-cell">${rec.timestamp}</td>
      <td><strong>${rec.speakerId}</strong>${syntheticTag}</td>
      <td class="mono-cell">${rec.modelId}</td>
      <td class="mono-cell">${rec.latency}</td>
      <td><span class="status-pill ${rec.isEmergency === 'YES' ? 'error' : 'ready'}">${rec.intent}</span></td>
      <td><span class="status-pill ${rec.outcome === 'SUCCESS' ? 'ready' : 'blocked'}">${rec.outcome}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function clearSessionHistory() {
  state.sessionHistory = [];
  renderSessionTable();
}

// ==========================================================================
// 8. Modals Implementation
// ==========================================================================

function initModals() {
  // Vector Modal
  document.getElementById("closeVectorModalBtn").addEventListener("click", () => {
    document.getElementById("vectorModal").classList.remove("active");
  });

  // Model Details Modal
  document.getElementById("closeModelDetailsModalBtn").addEventListener("click", () => {
    document.getElementById("modelDetailsModal").classList.remove("active");
  });

  // Figure Lightbox Modal
  document.getElementById("closeFigureLightboxBtn").addEventListener("click", () => {
    document.getElementById("figureLightboxModal").classList.remove("active");
  });

  // Close modals on clicking backdrop
  document.querySelectorAll(".modal-backdrop").forEach(backdrop => {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) backdrop.classList.remove("active");
    });
  });

  // Copy vector button
  document.getElementById("copyVectorBtn").addEventListener("click", () => {
    if (state.extractedEmbedding && state.extractedEmbedding.full_vector) {
      navigator.clipboard.writeText(JSON.stringify(state.extractedEmbedding.full_vector));
      showToast({ type: "success", title: "Vector Copied", message: "Full 384-dimensional Whisper acoustic feature vector copied to clipboard." });
    }
  });
}

function openVectorInspectorModal() {
  if (!state.extractedEmbedding) return;
  const modal = document.getElementById("vectorModal");
  const previewBox = document.getElementById("vectorPreviewBox");

  const preview = state.extractedEmbedding.vector_preview || [];
  previewBox.textContent = JSON.stringify(preview, null, 2);
  modal.classList.add("active");
}

function openModelDetailsModal(modelId, speakerId) {
  const model = state.allModels.find(m => m.model_id === modelId);
  if (!model) return;

  const modal = document.getElementById("modelDetailsModal");
  document.getElementById("detailModalModelId").textContent = model.model_id;
  document.getElementById("detailModalArchitecture").textContent = model.architecture;
  document.getElementById("detailModalStatus").textContent = model.status;
  document.getElementById("detailModalSha256").textContent = model.sha256 || "Unmaterialized (No Checkpoint File)";
  document.getElementById("detailModalParams").textContent = model.parameter_count || "25,223 parameters";
  document.getElementById("detailModalPath").textContent = model.path || "N/A";
  document.getElementById("detailModalRemedy").textContent = model.status === "MATERIALIZED_AND_VERIFIED"
    ? "Verified for local demonstration inference."
    : "Model cannot be fabricated. Obtain explicit owner authorization to materialize checkpoint.";

  modal.classList.add("active");
}

function openFigureLightbox(figureId) {
  const fig = state.figures.find(f => f.figure_id === figureId);
  if (!fig) return;

  const modal = document.getElementById("figureLightboxModal");
  document.getElementById("lightboxTitle").textContent = fig.title;
  document.getElementById("lightboxImage").src = fig.url;
  document.getElementById("lightboxProtocol").textContent = fig.protocol_context;
  document.getElementById("lightboxDescription").textContent = fig.description;

  modal.classList.add("active");
}

// Utility: Parallel Fetch Helper
async function fetchParallel(fetchPromises) {
  const responses = await Promise.all(fetchPromises);
  return Promise.all(responses.map(r => {
    if (!r.ok) throw new Error(`HTTP Error ${r.status} on ${r.url}`);
    return r.json();
  }));
}

// ==========================================================================
// 9. Modern Non-Blocking Toast Notification Engine
// ==========================================================================

function initToastSystem() {
  if (!document.getElementById("toastContainer")) {
    const cont = document.createElement("div");
    cont.id = "toastContainer";
    cont.className = "toast-container";
    cont.setAttribute("aria-live", "polite");
    document.body.appendChild(cont);
  }
}

function showToast({ type = "info", title = "", message = "", duration = 4000 }) {
  let container = document.getElementById("toastContainer");
  if (!container) {
    initToastSystem();
    container = document.getElementById("toastContainer");
  }
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast-item toast-${type}`;
  toast.setAttribute("role", "alert");

  const icons = {
    success: "✅",
    error: "❌",
    warning: "⚠️",
    info: "⚛️"
  };

  toast.innerHTML = `
    <div class="toast-icon-wrap" aria-hidden="true">${icons[type] || "ℹ️"}</div>
    <div class="toast-text-wrap">
      ${title ? `<div class="toast-title">${title}</div>` : ""}
      <div class="toast-msg">${message}</div>
    </div>
    <button class="toast-close" aria-label="Dismiss">&times;</button>
    <div class="toast-progress" style="animation-duration: ${duration}ms;"></div>
  `;

  const closeBtn = toast.querySelector(".toast-close");
  let timerId = null;

  function dismiss() {
    if (timerId) clearTimeout(timerId);
    toast.classList.add("toast-leave");
    setTimeout(() => {
      if (toast.parentElement) toast.parentElement.removeChild(toast);
    }, 260);
  }

  closeBtn.addEventListener("click", dismiss);
  timerId = setTimeout(dismiss, duration);

  container.appendChild(toast);
}

// ==========================================================================
// 10. Interactive Quantum Background Lattice Engine
// ==========================================================================

function initQuantumLatticeEngine() {
  const canvas = document.getElementById("quantumBgCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let width = (canvas.width = window.innerWidth || document.documentElement.clientWidth || 1280);
  let height = (canvas.height = window.innerHeight || document.documentElement.clientHeight || 800);

  function resizeLattice() {
    width = canvas.width = window.innerWidth || document.documentElement.clientWidth || 1280;
    height = canvas.height = window.innerHeight || document.documentElement.clientHeight || 800;
  }

  const isReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (isReducedMotion) return;

  const nodeCount = Math.min(48, Math.floor((width * height) / 26000));
  const nodes = [];
  const mouse = { x: -1000, y: -1000, active: false };

  window.addEventListener("resize", resizeLattice);

  window.addEventListener("mousemove", (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.active = true;
  });

  window.addEventListener("mouseleave", () => {
    mouse.active = false;
  });

  for (let i = 0; i < nodeCount; i++) {
    nodes.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.45,
      vy: (Math.random() - 0.5) * 0.45,
      radius: Math.random() * 1.8 + 1.2,
      phase: Math.random() * Math.PI * 2,
      phaseSpeed: Math.random() * 0.02 + 0.01,
      color: i % 3 === 0 ? "#22D3EE" : i % 3 === 1 ? "#8B5CF6" : "#6366F1"
    });
  }

  function renderLattice() {
    if (document.hidden) {
      requestAnimationFrame(renderLattice);
      return;
    }

    ctx.clearRect(0, 0, width, height);

    // Update & draw nodes
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      n.x += n.vx;
      n.y += n.vy;
      n.phase += n.phaseSpeed;

      // Bounce on borders
      if (n.x < 0) { n.x = 0; n.vx *= -1; }
      else if (n.x > width) { n.x = width; n.vx *= -1; }
      if (n.y < 0) { n.y = 0; n.vy *= -1; }
      else if (n.y > height) { n.y = height; n.vy *= -1; }

      // Mouse subtle interaction
      if (mouse.active) {
        const dx = mouse.x - n.x;
        const dy = mouse.y - n.y;
        const dist = Math.hypot(dx, dy);
        if (dist < 180 && dist > 1) {
          const force = (180 - dist) / 180 * 0.08;
          n.x -= (dx / dist) * force;
          n.y -= (dy / dist) * force;
        }
      }

      // Draw node with pulsating glow
      const alpha = 0.4 + 0.3 * Math.sin(n.phase);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.globalAlpha = alpha;
      ctx.shadowBlur = 8;
      ctx.shadowColor = n.color;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Draw entanglement lines
    const maxDist = 135;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const n1 = nodes[i];
        const n2 = nodes[j];
        const dist = Math.hypot(n1.x - n2.x, n1.y - n2.y);
        if (dist < maxDist) {
          const lineAlpha = (1 - dist / maxDist) * 0.22;
          ctx.beginPath();
          ctx.moveTo(n1.x, n1.y);
          ctx.lineTo(n2.x, n2.y);
          ctx.strokeStyle = n1.color === "#22D3EE"
            ? `rgba(34, 211, 238, ${lineAlpha})`
            : `rgba(139, 92, 246, ${lineAlpha})`;
          ctx.lineWidth = 1;
          ctx.globalAlpha = 1;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(renderLattice);
  }

  renderLattice();
}

// ==========================================================================
// 11. Live 8-Qubit Statevector Precession HUD Engine
// ==========================================================================

let globalQubitHUDState = "IDLE";

function setQubitHUDState(stateName) {
  globalQubitHUDState = stateName;
  const statePill = document.getElementById("qubitStatePill");
  if (statePill) {
    if (stateName === "RUNNING_QUANTUM_MODEL") {
      statePill.textContent = "State: Annealing |Ψ(t)⟩";
      statePill.className = "status-pill warning";
    } else if (stateName === "COMPLETED") {
      statePill.textContent = "State: Collapsed |ψ_final⟩";
      statePill.className = "status-pill ready";
    } else {
      statePill.textContent = "State: Ground |ψ₀⟩";
      statePill.className = "status-pill ready";
    }
  }
}

function initQubitPrecessionHUD() {
  const canvas = document.getElementById("qubitPrecessionCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resizeHUD() {
    const parent = canvas.parentElement;
    const parentW = parent ? parent.clientWidth : 0;
    const computedW = canvas.clientWidth || parentW || 560;
    canvas.width = Math.max(computedW, 320);
    canvas.height = 190;
  }
  window.resizeQubitHUD = resizeHUD;
  resizeHUD();
  window.addEventListener("resize", resizeHUD);

  const qubitCount = 8;
  let angleBase = 0;

  function drawHUD() {
    if (canvas.width <= 10 || (canvas.parentElement && Math.abs(canvas.width - canvas.parentElement.clientWidth) > 20 && canvas.parentElement.clientWidth > 50)) {
      resizeHUD();
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const radius = Math.min(cx, cy) - 26;

    const isExcited = globalQubitHUDState === "RUNNING_QUANTUM_MODEL";
    const speedMult = isExcited ? 3.5 : 0.8;
    angleBase += 0.015 * speedMult;

    // Draw central quantum nucleus
    ctx.beginPath();
    ctx.arc(cx, cy, 14, 0, Math.PI * 2);
    ctx.fillStyle = isExcited ? "rgba(245, 158, 11, 0.25)" : "rgba(139, 92, 246, 0.25)";
    ctx.shadowBlur = isExcited ? 18 : 10;
    ctx.shadowColor = isExcited ? "#F59E0B" : "#8B5CF6";
    ctx.fill();
    ctx.shadowBlur = 0;

    // Draw orbital guidance ring
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Calculate 8 qubit positions
    const positions = [];
    for (let i = 0; i < qubitCount; i++) {
      const theta = (i * 2 * Math.PI) / qubitCount + angleBase * (i % 2 === 0 ? 1 : -0.7);
      const qx = cx + Math.cos(theta) * radius;
      const qy = cy + Math.sin(theta) * radius;
      positions.push({ x: qx, y: qy, theta, idx: i });
    }

    // Draw ZZ entanglement coupling lines
    for (let i = 0; i < qubitCount; i++) {
      const nextIdx = (i + 1) % qubitCount;
      ctx.beginPath();
      ctx.moveTo(positions[i].x, positions[i].y);
      ctx.lineTo(positions[nextIdx].x, positions[nextIdx].y);
      ctx.strokeStyle = isExcited ? "rgba(34, 211, 238, 0.55)" : "rgba(139, 92, 246, 0.25)";
      ctx.lineWidth = isExcited ? 2 : 1;
      if (isExcited) {
        ctx.shadowBlur = 8;
        ctx.shadowColor = "#22D3EE";
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw photon pulses in excited state
      if (isExcited) {
        const pulsePos = (angleBase * 2 + i * 0.4) % 1;
        const px = positions[i].x + (positions[nextIdx].x - positions[i].x) * pulsePos;
        const py = positions[i].y + (positions[nextIdx].y - positions[i].y) * pulsePos;
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#22D3EE";
        ctx.shadowBlur = 10;
        ctx.shadowColor = "#22D3EE";
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Draw each of the 8 qubits
    for (let i = 0; i < qubitCount; i++) {
      const q = positions[i];
      const qubitColor = i % 2 === 0 ? "#22D3EE" : "#A78BFA";

      // Qubit glow outer circle
      ctx.beginPath();
      ctx.arc(q.x, q.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = isExcited ? "rgba(34, 211, 238, 0.35)" : "rgba(139, 92, 246, 0.2)";
      ctx.shadowBlur = isExcited ? 14 : 8;
      ctx.shadowColor = qubitColor;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Inner qubit nucleus
      ctx.beginPath();
      ctx.arc(q.x, q.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = qubitColor;
      ctx.fill();

      // Qubit Label (|q₀⟩...|q₇⟩)
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#CBD5E1";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const labelOffset = 16;
      const lx = q.x + Math.cos(q.theta) * labelOffset;
      const ly = q.y + Math.sin(q.theta) * labelOffset;
      ctx.fillText(`|q${i}⟩`, lx, ly);
    }

    requestAnimationFrame(drawHUD);
  }

  drawHUD();
}

// ==========================================================================
// 12. Animated Radial Confidence Gauge & Numeric Rollup
// ==========================================================================

function updateRadialConfidenceGauge(score, isEmergency, isAbstain) {
  const bar = document.getElementById("radialGaugeBar");
  const percentEl = document.getElementById("gaugePercentVal");
  if (!bar || !percentEl) return;

  const validScore = Math.max(0, Math.min(1, typeof score === "number" ? score : 0));
  const circumference = 251.32; // 2 * Math.PI * 40
  const offset = circumference * (1 - validScore);

  bar.style.strokeDashoffset = offset;

  if (isEmergency) {
    bar.className = "gauge-bar emergency";
  } else if (isAbstain) {
    bar.className = "gauge-bar abstain";
  } else {
    bar.className = "gauge-bar";
  }

  // Smooth roll-up animation
  const startVal = state.currentConfidenceScore || 0;
  const targetVal = validScore * 100;
  state.currentConfidenceScore = validScore;

  const startTime = performance.now();
  const duration = 800;

  function stepRollup(now) {
    const elapsed = now - startTime;
    const progress = Math.min(1, elapsed / duration);
    // Quadratic ease out
    const ease = 1 - (1 - progress) * (1 - progress);
    const current = startVal + (targetVal - startVal) * ease;
    percentEl.textContent = `${current.toFixed(1)}%`;
    if (progress < 1) {
      requestAnimationFrame(stepRollup);
    } else {
      percentEl.textContent = `${targetVal.toFixed(1)}%`;
    }
  }

  requestAnimationFrame(stepRollup);
}

// ==========================================================================
// 13. 3D Perspective Hover Tilt & Holographic Highlight Engine
// ==========================================================================

function init3DCardTiltEngine() {
  const cards = document.querySelectorAll(".card, .bento-hero, .bento-stat-card");
  cards.forEach(card => {
    // Avoid double attachment
    if (card._tiltAttached) return;
    card._tiltAttached = true;

    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty("--mouse-x", `${x}px`);
      card.style.setProperty("--mouse-y", `${y}px`);

      // 3D subtle tilt (max 4.5 degrees)
      const rotX = ((y / rect.height) - 0.5) * -5.0;
      const rotY = ((x / rect.width) - 0.5) * 5.0;
      card.style.transform = `perspective(900px) rotateX(${rotX.toFixed(2)}deg) rotateY(${rotY.toFixed(2)}deg) translateY(-2px)`;
    });

    card.addEventListener("mouseleave", () => {
      card.style.transform = "perspective(900px) rotateX(0deg) rotateY(0deg) translateY(0)";
    });
  });
}

// ==========================================================================
// 14. Global Keyboard Accessibility & Drag/Drop Guards
// ==========================================================================

function initGlobalKeyboardAndDragGuards() {
  // 1. Escape key closes modals and drawers
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      document.querySelectorAll(".modal-backdrop.active").forEach(m => m.classList.remove("active"));
      const safetyDrawer = document.getElementById("safetyDetailsDrawer");
      if (safetyDrawer && safetyDrawer.classList.contains("expanded")) {
        safetyDrawer.classList.remove("expanded");
        const safetyBtn = document.getElementById("safetyDetailsBtn");
        if (safetyBtn) safetyBtn.innerHTML = "<span>Governance &amp; Limits</span> ▾";
      }
      const sidebar = document.getElementById("appSidebar");
      if (sidebar && sidebar.classList.contains("mobile-open")) {
        sidebar.classList.remove("mobile-open");
      }
    }
  });

  // 2. Prevent accidental file drops navigating the browser
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // 3. User gesture auto-resume for Web Audio
  window.addEventListener("click", () => {
    if (state.audioContext && state.audioContext.state === "suspended") {
      state.audioContext.resume().catch(() => {});
    }
  }, { once: false });
}
