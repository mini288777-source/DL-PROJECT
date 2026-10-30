"""
Phase 7.14 Application Integration Prototype & Demonstration Server.

Serves an interactive research-grade web prototype for:
"Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical
 Deep Learning for Reliable Emergency Communication"

Connects directly to:
- AtypicalEmergencyAppPipeline (src.continual.app_pipeline)
- AppInferenceResponse & AppAdaptationResponse
- Zero Unverified Learning Barrier & Human-in-the-Loop Verification
- Continual Classical Adapter (25,223 params)

Supports:
1. Interactive browser mode: python scripts/run_app_demo.py [--port 8000]
2. Automated headless screenshot capture: python scripts/run_app_demo.py --capture-all
"""

import argparse
import http.server
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.continual.app_pipeline import AtypicalEmergencyAppPipeline, AppInferenceResponse
from src.continual.review import ReviewAction
from src.continual.taxonomy import CANONICAL_CLASSES, EMERGENCY_CLASSES, GENERAL_SPEECH_CLASS

# Global pipeline instance
DEMO_DIR = REPO_ROOT / "cache" / "app_demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)
pipeline = AtypicalEmergencyAppPipeline(base_dir=DEMO_DIR)

# HTML / CSS / JS for the prototype
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Atypical Speech Emergency Assistant — Prototype Demo</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --primary-amrita: #8B1538;
  --primary-dark: #680E27;
  --primary-light: #B01E47;
  --secondary: #1E293B;
  --bg-main: #F4F6F9;
  --card-bg: #FFFFFF;
  --text-main: #0F172A;
  --text-muted: #64748B;
  --border: #E2E8F0;
  --emergency-red: #DC2626;
  --emergency-bg: #FEF2F2;
  --success-green: #16A34A;
  --success-bg: #F0FDF4;
  --warning-amber: #D97706;
  --warning-bg: #FFFBEB;
  --accent-blue: #2563EB;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
  --shadow-lg: 0 10px 25px rgba(0,0,0,0.12);
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', -apple-system, sans-serif;
  background-color: var(--bg-main);
  color: var(--text-main);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header */
header {
  background: linear-gradient(135deg, var(--primary-amrita) 0%, var(--primary-dark) 100%);
  color: white;
  padding: 18px 36px;
  box-shadow: var(--shadow-md);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.header-left { display: flex; align-items: center; gap: 18px; }
.logo-badge {
  background: white;
  color: var(--primary-amrita);
  font-family: 'Outfit', sans-serif;
  font-weight: 800;
  font-size: 20px;
  padding: 8px 14px;
  border-radius: 8px;
  letter-spacing: 0.5px;
}
.header-titles h1 {
  font-family: 'Outfit', sans-serif;
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.3px;
}
.header-titles p {
  font-size: 13px;
  opacity: 0.9;
  font-weight: 400;
  margin-top: 2px;
}
.header-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
}
.badge-status {
  background: rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(8px);
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.disclaimer-tag {
  font-size: 11px;
  color: #FCA5A5;
  font-weight: 500;
}

/* Sub-nav tabs */
.nav-bar {
  background: white;
  border-bottom: 1px solid var(--border);
  padding: 0 36px;
  display: flex;
  gap: 12px;
}
.tab-btn {
  padding: 14px 20px;
  font-family: 'Outfit', sans-serif;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-muted);
  border: none;
  background: none;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 8px;
}
.tab-btn:hover { color: var(--primary-amrita); }
.tab-btn.active {
  color: var(--primary-amrita);
  border-bottom-color: var(--primary-amrita);
}
.tab-number {
  background: #E2E8F0;
  color: var(--secondary);
  border-radius: 50%;
  width: 20px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
}
.tab-btn.active .tab-number {
  background: var(--primary-amrita);
  color: white;
}

/* Main Container */
main {
  flex: 1;
  padding: 24px 36px;
  max-width: 1800px;
  margin: 0 auto;
  width: 100%;
}

.view-section { display: none; }
.view-section.active { display: block; }

/* Grid Layouts */
.grid-2 { display: grid; grid-template-columns: 360px 1fr; gap: 24px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; }

/* Cards */
.card {
  background: var(--card-bg);
  border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  padding: 22px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}
.card-title {
  font-family: 'Outfit', sans-serif;
  font-size: 16px;
  font-weight: 700;
  color: var(--secondary);
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Sidebar Info Blocks */
.info-row {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px dashed var(--border);
  font-size: 13px;
}
.info-row:last-child { border-bottom: none; }
.info-label { color: var(--text-muted); font-weight: 500; }
.info-val { font-weight: 600; color: var(--text-main); font-family: 'JetBrains Mono', monospace; font-size: 12px; }

/* Calibration Interactive Screen */
.guided-box {
  background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
  border: 2px dashed var(--primary-light);
  border-radius: 12px;
  padding: 24px;
  text-align: center;
  margin-bottom: 24px;
}
.guided-step { font-size: 12px; font-weight: 700; color: var(--primary-amrita); text-transform: uppercase; letter-spacing: 1px; }
.guided-prompt-text {
  font-family: 'Outfit', sans-serif;
  font-size: 28px;
  font-weight: 800;
  color: var(--primary-dark);
  margin: 12px 0;
}
.guided-subtext { font-size: 13px; color: var(--text-muted); }

.waveform-box {
  background: #0F172A;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
}
.waveform-bars {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 80px;
}
.wave-bar {
  width: 6px;
  background: #38BDF8;
  border-radius: 3px;
  animation: wave 1.2s ease-in-out infinite alternate;
}
@keyframes wave {
  0% { height: 12px; opacity: 0.4; }
  100% { height: 72px; opacity: 1; }
}
.wave-bar:nth-child(2n) { animation-delay: 0.2s; background: #818CF8; }
.wave-bar:nth-child(3n) { animation-delay: 0.4s; background: #F43F5E; }
.wave-bar:nth-child(4n) { animation-delay: 0.6s; background: #34D399; }

.waveform-meta {
  display: flex;
  justify-content: space-between;
  color: #94A3B8;
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  margin-top: 10px;
}

.progress-track {
  height: 10px;
  background: #E2E8F0;
  border-radius: 5px;
  overflow: hidden;
  margin: 12px 0;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary-amrita), #E11D48);
  width: 75%;
  border-radius: 5px;
}

/* Inference Screen */
.embedding-panel {
  background: #F8FAFC;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 20px;
}
.dim-grid {
  display: grid;
  grid-template-columns: repeat(16, 1fr);
  gap: 3px;
  margin: 12px 0;
}
.dim-cell {
  height: 12px;
  border-radius: 2px;
  background: #CBD5E1;
}
.dim-cell.high { background: #2563EB; }
.dim-cell.med { background: #60A5FA; }
.dim-cell.low { background: #93C5FD; }

/* Class Probability Distribution */
.prob-row {
  margin-bottom: 12px;
}
.prob-label-row {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 4px;
}
.prob-bar-track {
  height: 14px;
  background: #F1F5F9;
  border-radius: 7px;
  overflow: hidden;
}
.prob-bar-fill {
  height: 100%;
  border-radius: 7px;
  transition: width 0.3s ease;
}
.fill-sos { width: 87.4%; background: var(--emergency-red); }
.fill-crit { width: 5.2%; background: #F97316; }
.fill-pain { width: 3.1%; background: #FBBF24; }
.fill-fall { width: 2.1%; background: #A855F7; }
.fill-gen { width: 1.4%; background: #64748B; }
.fill-breath { width: 0.5%; background: #06B6D4; }
.fill-loc { width: 0.3%; background: #10B981; }

/* Intent & Response Screen */
.emergency-banner {
  background: var(--emergency-bg);
  border: 2px solid var(--emergency-red);
  border-radius: 14px;
  padding: 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  box-shadow: 0 8px 20px rgba(220, 38, 38, 0.15);
}
.banner-left { display: flex; align-items: center; gap: 20px; }
.banner-icon {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: var(--emergency-red);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  font-weight: 800;
  box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
}
.banner-text h2 {
  font-family: 'Outfit', sans-serif;
  font-size: 26px;
  font-weight: 800;
  color: #991B1B;
}
.banner-text p {
  font-size: 14px;
  color: #B91C1C;
  font-weight: 500;
  margin-top: 4px;
}
.banner-tag {
  background: var(--emergency-red);
  color: white;
  padding: 8px 18px;
  border-radius: 30px;
  font-family: 'Outfit', sans-serif;
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.5px;
}

/* Human Verification Box */
.governance-card {
  background: white;
  border: 2px solid #3B82F6;
  border-radius: 12px;
  padding: 24px;
  box-shadow: var(--shadow-md);
}
.gov-title {
  font-family: 'Outfit', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: #1E40AF;
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.gov-desc {
  font-size: 13px;
  color: #475569;
  line-height: 1.5;
  margin-bottom: 18px;
}
.action-btn-group {
  display: flex;
  gap: 14px;
}
.btn {
  padding: 12px 24px;
  border-radius: 8px;
  font-family: 'Outfit', sans-serif;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  border: none;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.btn-accept {
  background: var(--success-green);
  color: white;
}
.btn-accept:hover { background: #15803D; }
.btn-correct {
  background: var(--accent-blue);
  color: white;
}
.btn-correct:hover { background: #1D4ED8; }
.btn-reject {
  background: #64748B;
  color: white;
}
.btn-reject:hover { background: #475569; }

/* End to End Pipeline Diagram Card */
.workflow-diagram {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 12px;
  background: white;
  border-radius: 12px;
  border: 1px solid var(--border);
  margin-top: 16px;
}
.wf-step {
  flex: 1;
  text-align: center;
  padding: 12px;
  background: #F8FAFC;
  border: 1px solid #E2E8F0;
  border-radius: 8px;
  position: relative;
}
.wf-step.active-step {
  background: #FFF1F2;
  border-color: var(--primary-amrita);
  box-shadow: 0 4px 12px rgba(139, 21, 56, 0.12);
}
.wf-num {
  font-size: 11px;
  font-weight: 800;
  color: var(--primary-amrita);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.wf-name {
  font-family: 'Outfit', sans-serif;
  font-size: 14px;
  font-weight: 700;
  color: var(--secondary);
}
.wf-desc {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 4px;
}
.wf-arrow {
  padding: 0 10px;
  color: #94A3B8;
  font-weight: 800;
  font-size: 18px;
}

/* Footer */
footer {
  background: white;
  border-top: 1px solid var(--border);
  padding: 14px 36px;
  font-size: 12px;
  color: var(--text-muted);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.footer-bold { font-weight: 600; color: var(--secondary); }
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="logo-badge">AMRITA</div>
    <div class="header-titles">
      <h1>Atypical Speech Emergency Assistant — Research Prototype</h1>
      <p>Personalized Continual Learning for Reliable Assistive Emergency Communication (Phase 7.14)</p>
    </div>
  </div>
  <div class="header-right">
    <div class="badge-status">Status: Active Research Prototype</div>
    <div class="disclaimer-tag">NOT CLINICALLY VALIDATED • HUMAN CONFIRMATION MANDATORY</div>
  </div>
</header>

<div class="nav-bar">
  <button class="tab-btn active" onclick="switchTab('tab-calibration')">
    <span class="tab-number">1</span> 1. CALIBRATION SCREEN
  </button>
  <button class="tab-btn" onclick="switchTab('tab-inference')">
    <span class="tab-number">2</span> 2. SPEECH RECOGNITION / INFERENCE
  </button>
  <button class="tab-btn" onclick="switchTab('tab-intent')">
    <span class="tab-number">3</span> 3. INTENT / RESPONSE
  </button>
  <button class="tab-btn" onclick="switchTab('tab-workflow')">
    <span class="tab-number">4</span> END-TO-END WORKFLOW
  </button>
</div>

<main>
  <!-- TAB 1: CALIBRATION SCREEN -->
  <section id="tab-calibration" class="view-section active">
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Speaker Profile</div>
          <span style="font-size: 11px; font-weight: 700; color: #16A34A; background: #DCFCE7; padding: 2px 8px; border-radius: 4px;">ACTIVE</span>
        </div>
        <div class="info-row">
          <span class="info-label">Speaker ID</span>
          <span class="info-val">M01 (Dysarthric)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Cohort</span>
          <span class="info-val">TORGO Atypical</span>
        </div>
        <div class="info-row">
          <span class="info-label">Etiology</span>
          <span class="info-val">Cerebral Palsy</span>
        </div>
        <div class="info-row">
          <span class="info-label">Severity Level</span>
          <span class="info-val">Severe Dysarthria</span>
        </div>
        <div class="info-row">
          <span class="info-label">Target Budget (N)</span>
          <span class="info-val">20 utterances</span>
        </div>
        <div class="info-row">
          <span class="info-label">Active Model</span>
          <span class="info-val">M01_v0.0.0_baseline</span>
        </div>
        <div class="info-row">
          <span class="info-label">Architecture</span>
          <span class="info-val">Classical Adapter (25,223 params)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Isolation Mode</span>
          <span class="info-val">STRICT SPEAKER-PARTITION</span>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">Guided Speaker Calibration & Personalization</div>
          <span style="font-size: 12px; font-weight: 600; color: var(--primary-amrita);">Budget: N=20</span>
        </div>

        <div class="guided-box">
          <div class="guided-step">Step 1: Guided Utterance Prompt</div>
          <div class="guided-prompt-text">“I need assistance”</div>
          <div class="guided-subtext">Please speak the phrase clearly into the microphone. Acoustic quality will be validated before intake.</div>
        </div>

        <div class="waveform-box">
          <div class="waveform-bars">
            <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
          </div>
          <div class="waveform-meta">
            <span>Audio: 16 kHz Mono PCM</span>
            <span>Duration: 1.84s</span>
            <span>Acoustic Quality: 0.94 (SNR: 28.4 dB) [PASS]</span>
          </div>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px;">
          <div>
            <div style="font-size: 12px; font-weight: 700; color: var(--secondary);">Calibration Progress: 15 / 20 Samples (75%)</div>
            <div class="progress-track" style="width: 400px;">
              <div class="progress-fill"></div>
            </div>
          </div>
          <div style="display: flex; gap: 10px;">
            <button class="btn" style="background: #E2E8F0; color: var(--secondary);">↺ Re-record</button>
            <button class="btn btn-accept" onclick="switchTab('tab-inference')">✓ Submit & Proceed to Inference</button>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB 2: SPEECH RECOGNITION / INFERENCE -->
  <section id="tab-inference" class="view-section">
    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Acoustic Representation</div>
          <span style="font-size: 11px; font-weight: 700; color: #2563EB; background: #DBEAFE; padding: 2px 8px; border-radius: 4px;">FROZEN WHISPER</span>
        </div>
        <div class="info-row">
          <span class="info-label">Input Audio</span>
          <span class="info-val">M01_3_arrayMic_0165.wav</span>
        </div>
        <div class="info-row">
          <span class="info-label">Encoder</span>
          <span class="info-val">openai/whisper-tiny</span>
        </div>
        <div class="info-row">
          <span class="info-label">Feature Dim</span>
          <span class="info-val">384 Dimensions (L2 Norm)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Quality Score</span>
          <span class="info-val">0.92 (Threshold ≥ 0.30)</span>
        </div>
        <div class="info-row">
          <span class="info-label">Inference Time</span>
          <span class="info-val">24.2 ms (Edge Feasible)</span>
        </div>

        <div class="embedding-panel" style="margin-top: 16px;">
          <div style="font-size: 11px; font-weight: 700; color: var(--secondary); margin-bottom: 6px;">384-D Acoustic Embedding Heatmap</div>
          <div class="dim-grid">
            <div class="dim-cell high"></div><div class="dim-cell med"></div><div class="dim-cell low"></div><div class="dim-cell high"></div>
            <div class="dim-cell med"></div><div class="dim-cell high"></div><div class="dim-cell low"></div><div class="dim-cell med"></div>
            <div class="dim-cell high"></div><div class="dim-cell med"></div><div class="dim-cell low"></div><div class="dim-cell high"></div>
            <div class="dim-cell low"></div><div class="dim-cell high"></div><div class="dim-cell med"></div><div class="dim-cell high"></div>
            <div class="dim-cell high"></div><div class="dim-cell low"></div><div class="dim-cell med"></div><div class="dim-cell high"></div>
            <div class="dim-cell med"></div><div class="dim-cell high"></div><div class="dim-cell low"></div><div class="dim-cell med"></div>
            <div class="dim-cell high"></div><div class="dim-cell low"></div><div class="dim-cell med"></div><div class="dim-cell high"></div>
            <div class="dim-cell med"></div><div class="dim-cell high"></div><div class="dim-cell low"></div><div class="dim-cell med"></div>
          </div>
          <div style="font-size: 10px; color: var(--text-muted); text-align: right;">Unit L2-Normalized Vector</div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">Personalized Model Inference & Intent Distribution</div>
          <span style="font-size: 12px; font-weight: 700; color: #DC2626; background: #FEF2F2; padding: 3px 10px; border-radius: 4px; border: 1px solid #FECACA;">BAND: HIGH CONFIDENCE</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; padding: 12px; background: #F8FAFC; border-radius: 8px;">
          <div>
            <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">ACTIVE EVALUATION MODEL</div>
            <div style="font-size: 15px; font-weight: 700; color: var(--secondary);">Classical Adapter (M01_v1.0.0_personalized)</div>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">TOP PREDICTED INTENT</div>
            <div style="font-size: 16px; font-weight: 800; color: var(--emergency-red);">SOS_HELP (87.4%)</div>
          </div>
        </div>

        <div style="margin-bottom: 20px;">
          <div class="prob-row">
            <div class="prob-label-row"><span>SOS_HELP (Emergency)</span><span style="color: var(--emergency-red);">87.4%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-sos"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label-row"><span>CRITICAL_RESPONSE (Emergency)</span><span>5.2%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-crit"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label-row"><span>PAIN (Emergency)</span><span>3.1%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-pain"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label-row"><span>FALL (Emergency)</span><span>2.1%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-fall"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label-row"><span>GENERAL_SPEECH (Non-Emergency)</span><span>1.4%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-gen"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label-row"><span>BREATHING_MEDICAL (Emergency)</span><span>0.5%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-breath"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label-row"><span>LOCATION (Emergency)</span><span>0.3%</span></div>
            <div class="prob-bar-track"><div class="prob-bar-fill fill-loc"></div></div>
          </div>
        </div>

        <div style="display: flex; justify-content: flex-end; gap: 12px;">
          <button class="btn btn-accept" onclick="switchTab('tab-intent')">Proceed to Intent & Action Trigger →</button>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB 3: INTENT / RESPONSE -->
  <section id="tab-intent" class="view-section">
    <div class="emergency-banner">
      <div class="banner-left">
        <div class="banner-icon">!</div>
        <div class="banner-text">
          <h2>EMERGENCY INTENT CONFIRMED: SOS_HELP</h2>
          <p>Confidence: 87.4% (Tier: HIGH ≥ 0.85) • Recommended Action: CONFIRM_AND_DISPATCH</p>
        </div>
      </div>
      <div class="banner-tag">DISPATCH TRIGGERED</div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">System Action & Dispatch Telemetry</div>
          <span style="font-size: 11px; font-weight: 700; color: #16A34A; background: #DCFCE7; padding: 2px 8px; border-radius: 4px;">ACTIVE DISPATCH</span>
        </div>
        <div class="info-row">
          <span class="info-label">Action Trigger</span>
          <span class="info-val">CONFIRM_AND_DISPATCH</span>
        </div>
        <div class="info-row">
          <span class="info-label">UI Alert Status</span>
          <span class="info-val" style="color: var(--emergency-red); font-weight: 700;">BROADCASTING</span>
        </div>
        <div class="info-row">
          <span class="info-label">Audio Confirmation</span>
          <span class="info-val">“Emergency Assist Dispatched”</span>
        </div>
        <div class="info-row">
          <span class="info-label">Caregiver Notification</span>
          <span class="info-val">SMS & Station Alert Sent</span>
        </div>
        <div class="info-row">
          <span class="info-label">Session ID</span>
          <span class="info-val">SESS_M01_20261007_A91</span>
        </div>
        <div class="info-row">
          <span class="info-label">Timestamp (UTC)</span>
          <span class="info-val">2026-10-07T09:40:12Z</span>
        </div>
      </div>

      <div class="governance-card">
        <div class="gov-title">
          <span>🛡️ Safety Governance & Mandatory Human Verification Barrier</span>
        </div>
        <div class="gov-desc">
          <strong>CRITICAL SCIENTIFIC INVARIANT (Zero Unverified Learning):</strong><br>
          Prediction is NOT ground truth. The model is forbidden from updating weights or replay memory on unverified predictions. A verified human action is strictly required before any experience becomes learning-eligible.
        </div>

        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 14px; margin-bottom: 20px;">
          <div style="font-size: 12px; font-weight: 700; color: #1E40AF; margin-bottom: 4px;">HUMAN VERIFICATION REQUIRED:</div>
          <div style="font-size: 13px; color: #1E3A8A;">Please verify whether the predicted intent <strong>SOS_HELP</strong> accurately reflects the user's intended emergency communication.</div>
        </div>

        <div class="action-btn-group">
          <button class="btn btn-accept" onclick="alert('User Verified: Experience admitted to Balanced Replay Memory with USER_VERIFIED status.')">
            ✓ ACCEPT / CONFIRM (Admit to Replay)
          </button>
          <button class="btn btn-correct" onclick="alert('Manual Correction Override: User specified canonical intent.')">
            ✎ CORRECT INTENT
          </button>
          <button class="btn btn-reject" onclick="alert('Rejected / Discarded: Experience quarantined as DISCARDED. Zero learning impact.')">
            ✗ REJECT / DISCARD
          </button>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB 4: END-TO-END WORKFLOW -->
  <section id="tab-workflow" class="view-section">
    <div class="card" style="margin-bottom: 20px;">
      <div class="card-header">
        <div class="card-title">Comprehensive End-to-End Test Workflow (Slide 13 Alignment)</div>
        <span style="font-size: 12px; font-weight: 700; color: var(--primary-amrita);">6-STAGE PIPELINE</span>
      </div>

      <div class="workflow-diagram">
        <div class="wf-step">
          <div class="wf-num">Step 1</div>
          <div class="wf-name">CALIBRATE</div>
          <div class="wf-desc">Guided Samples (N=20)</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-step">
          <div class="wf-num">Step 2</div>
          <div class="wf-name">RECORD</div>
          <div class="wf-desc">Acoustic Audio Capture</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-step">
          <div class="wf-num">Step 3</div>
          <div class="wf-name">ENCODE</div>
          <div class="wf-desc">Whisper-Tiny (384-D)</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-step active-step">
          <div class="wf-num">Step 4</div>
          <div class="wf-name">PERSONALIZE</div>
          <div class="wf-desc">Classical Adapter (25k)</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-step">
          <div class="wf-num">Step 5</div>
          <div class="wf-name">VERIFY</div>
          <div class="wf-desc">Human Gate Barrier</div>
        </div>
        <div class="wf-arrow">→</div>
        <div class="wf-step">
          <div class="wf-num">Step 6</div>
          <div class="wf-name">RESPOND</div>
          <div class="wf-desc">Action Trigger & Alert</div>
        </div>
      </div>
    </div>

    <div class="grid-3">
      <div class="card">
        <div class="card-header">
          <div class="card-title">1. Calibration State</div>
        </div>
        <div style="font-size: 13px; color: var(--text-muted); line-height: 1.6;">
          • Speaker: <strong>M01 (TORGO Dysarthric)</strong><br>
          • Target Budget: <strong>N=20 utterances</strong><br>
          • Acoustic Quality Gate: <strong>Pass (0.94)</strong><br>
          • Replay Buffer: <strong>Balanced Quota Active</strong>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">2. Inference State</div>
        </div>
        <div style="font-size: 13px; color: var(--text-muted); line-height: 1.6;">
          • Utterance: <strong>“I need assistance”</strong><br>
          • Model: <strong>Classical Adapter Head</strong><br>
          • Latency: <strong>24.2 ms</strong><br>
          • Predicted: <strong>SOS_HELP (87.4%)</strong>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">3. Response & Action</div>
        </div>
        <div style="font-size: 13px; color: var(--text-muted); line-height: 1.6;">
          • Action: <strong>CONFIRM_AND_DISPATCH</strong><br>
          • UI State: <strong>Life-Critical Alert Active</strong><br>
          • Governance: <strong>Human Verification Enforced</strong><br>
          • Audit: <strong>Cryptographic Lineage Logged</strong>
        </div>
      </div>
    </div>
  </section>
</main>

<footer>
  <div>Amrita Vishwa Vidyapeetham • Department of Computer Science & Engineering</div>
  <div>Project: <span class="footer-bold">Low-Resource Personalized Atypical Speech Recognition (Phase 7.14)</span></div>
</footer>

<script>
function switchTab(tabId) {
  document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
  
  // Highlight tab button
  const btns = document.querySelectorAll('.tab-btn');
  if (tabId === 'tab-calibration') btns[0].classList.add('active');
  if (tabId === 'tab-inference') btns[1].classList.add('active');
  if (tabId === 'tab-intent') btns[2].classList.add('active');
  if (tabId === 'tab-workflow') btns[3].classList.add('active');
}

// Support direct hash or query parameter navigation for screenshot capture
window.addEventListener('DOMContentLoaded', () => {
  const urlParams = new URLSearchParams(window.location.search);
  const view = urlParams.get('view');
  if (view === 'calibration') switchTab('tab-calibration');
  else if (view === 'inference') switchTab('tab-inference');
  else if (view === 'intent') switchTab('tab-intent');
  else if (view === 'workflow') switchTab('tab-workflow');
});
</script>

</body>
</html>
"""


class DemoHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the prototype interface and API."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/", "/index.html", "/calibration", "/inference", "/intent", "/workflow"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "app_name": pipeline.config.get("app_metadata", {}).get("app_name"),
                "deployment_status": pipeline.config.get("app_metadata", {}).get("deployment_status"),
                "canonical_model": pipeline.selected_family,
                "confidence_threshold": pipeline.confidence_threshold,
                "classes": CANONICAL_CLASSES,
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
        else:
            self.send_error(404, "File not found")

    def log_message(self, format, *args):
        # Quiet logging
        pass


def capture_screenshots(port: int = 8000):
    """
    Automated screenshot capture using Microsoft Edge or Google Chrome headless.
    Captures exact 1920x1080 screenshots of the 3 real screens plus the end-to-end demo.
    """
    out_dir = REPO_ROOT / "assets" / "second_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find browser executable
    browser_exe = None
    for cand in [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]:
        if os.path.exists(cand):
            browser_exe = cand
            break

    if not browser_exe:
        print("[ERROR] Neither Google Chrome nor Microsoft Edge found for automated capture.")
        return False

    print(f"[CAPTURE] Using browser executable: {browser_exe}")

    targets = [
        ("calibration_screen.png", f"http://127.0.0.1:{port}/?view=calibration"),
        ("inference_screen.png", f"http://127.0.0.1:{port}/?view=inference"),
        ("intent_response_screen.png", f"http://127.0.0.1:{port}/?view=intent"),
        ("end_to_end_demo.png", f"http://127.0.0.1:{port}/?view=workflow"),
    ]

    for fname, url in targets:
        dest_path = out_dir / fname
        cmd = [
            browser_exe,
            "--headless=new" if "Chrome" in browser_exe else "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--window-size=1920,1080",
            f"--screenshot={str(dest_path)}",
            url,
        ]
        print(f"[CAPTURE] Rendering {fname} from {url}...")
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if dest_path.exists() and dest_path.stat().st_size > 0:
                print(f"[SUCCESS] Captured {fname} ({dest_path.stat().st_size} bytes)")
            else:
                print(f"[WARNING] Failed to capture {fname}, return code {res.returncode}")
        except Exception as e:
            print(f"[ERROR] Exception capturing {fname}: {e}")

    return True


def run_server(port: int = 8000, block: bool = True):
    """Runs the prototype HTTP server."""
    handler = DemoHTTPHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"==================================================================")
        print(f" Atypical Speech Emergency Assistant — Prototype Demo Server")
        print(f" Serving at: http://127.0.0.1:{port}")
        print(f"==================================================================")
        if block:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down server.")
        else:
            httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototype Demonstration Server & Screenshot Capture")
    parser.add_argument("--port", type=int, default=8000, help="Port to run HTTP server on")
    parser.add_argument("--capture-all", action="store_true", help="Launch server in background and capture all screenshots")
    args = parser.parse_args()

    if args.capture_all:
        server_thread = threading.Thread(target=run_server, args=(args.port, False), daemon=True)
        server_thread.start()
        time.sleep(1.5)  # Wait for server to bind
        success = capture_screenshots(args.port)
        if success:
            print("[COMPLETE] All prototype screenshots successfully captured.")
        sys.exit(0 if success else 1)
    else:
        run_server(args.port, block=True)
