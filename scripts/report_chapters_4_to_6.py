"""
Chapter 4 Content Generator for Final Project Report.
Written in natural, human, authentic student engineering style for Amrita School of Computing.
"""

def build_chapter_4(builder):
    print("Building Chapter 4: System Design and Implementation (Student & Human Style)...")
    builder.add_chapter_heading(4, "SYSTEM DESIGN AND IMPLEMENTATION")

    builder.add_division_heading("4.1. Overall System Architecture")
    builder.add_paragraph(
        "Our system architecture consists of two interconnected parts: an offline scientific experimental pipeline and an interactive online web application. To avoid methodological confusion, our implementation enforces a strict division between research benchmarking (where classical and quantum models are compared under matched conditions) and the live application (which runs exclusively on the canonical quantum model)."
    )
    builder.add_paragraph(
        "Figure 4.1 illustrates the complete system architecture. When an audio file is processed, digital signal processing resamples the audio to 16 kHz mono. The frozen Whisper-tiny acoustic encoder extracts 384-dimensional bottleneck latent vectors, which are then compressed via PCA to 8 dimensions and fed into the downstream quantum and classical classifiers."
    )

    builder.add_figure("reports/phase6_5_6/figures/fig8_evidence_chain_workflow.png", "Figure 4.1: End-to-end system architecture and experimental research workflow.")

    builder.add_division_heading("4.2. Training and Evaluation Pipeline")
    builder.add_paragraph(
        "During our initial tests in Phases 3 and 4, we trained models directly on a flat 7-class formulation. However, extreme class imbalance (95.8% general speech) caused all models to collapse into predicting the majority class 100% of the time, resulting in 0% emergency recall. To overcome this, our team implemented a two-stage hierarchical architecture in Phase 6.5:"
    )
    builder.add_paragraph(
        "1. Stage 1 (Binary Emergency Triage Gate): A binary classifier trained with cost-sensitive loss separates general conversational speech from potential emergencies."
    )
    builder.add_paragraph(
        "2. Stage 2 (Multiclass Subtype Classification): If an utterance is flagged as an emergency in Stage 1, it is routed to a multiclass classifier that identifies the specific emergency intent (Fall, Breathing, Pain, SOS Help, Critical Response)."
    )
    builder.add_paragraph(
        "This hierarchical structure significantly improved minority-class sensitivity, increasing emergency recall from 0.00% to over 26% across both quantum and classical models."
    )

    builder.add_division_heading("4.3. Quantum-Only Operational Web Application Architecture")
    builder.add_paragraph(
        "To demonstrate our system in an interactive environment, we developed a local full-stack web application called the Quantum Research Command Center. The application backend is built using FastAPI and Uvicorn running on Python 3.10. Table 4.1 summarizes the model architectures, and Table 4.2 outlines the API endpoints."
    )

    mod_headers = ["MODEL IDENTIFIER", "MODEL FAMILY", "INPUT DIM", "CIRCUIT / KERNEL", "DEPLOYMENT STATUS"]
    mod_data = [
        ["qsvc_q8_r1", "Quantum Kernel QSVC", "8 (PCA)", "8-Qubit ZZFeatureMap (R=1)", "CANONICAL OPERATIONAL PREDICTOR"],
        ["rbf_svm_d8", "Classical Kernel SVM", "8 (PCA)", "Gaussian RBF (gamma=0.125)", "RESEARCH BENCHMARK (HTTP 400 REJECTED)"],
        ["linear_svm", "Classical Linear SVM", "8 (PCA)", "Linear Kernel (C=1.0)", "RESEARCH BENCHMARK (HTTP 400 REJECTED)"],
        ["knn_k5", "K-Nearest Neighbors", "8 (PCA)", "k=5 Euclidean Distance", "RESEARCH BENCHMARK (HTTP 400 REJECTED)"],
        ["qsvc_q4_r1", "Quantum Kernel QSVC", "4 (PCA)", "4-Qubit ZZFeatureMap (R=1)", "RESEARCH ABLATION ONLY"]
    ]
    builder.add_table(mod_headers, mod_data, col_widths=[90, 85, 60, 115, 90], caption="Table 4.1: Evaluated Classical Baseline and Quantum Model Architecture Specifications")

    api_headers = ["ENDPOINT ROUTE", "HTTP METHOD", "FUNCTIONALITY", "ERROR HANDLING CONTRACT"]
    api_data = [
        ["/health", "GET", "System vitals and service status", "200 OK with runtime telemetry and version info"],
        ["/ready", "GET", "Operational model readiness check", "200 OK if qsvc_q8_r1 artifacts loaded; 503 if missing"],
        ["/predict", "POST", "JSON base64 audio classification", "200 OK; 400 if classical requested; 503 if unready"],
        ["/predict/file", "POST", "Multipart form-data audio upload", "200 OK; 400 for bad audio; 404 for unknown speaker"],
        ["/api/audio/transcribe", "POST", "Offline Whisper-tiny ASR transcription", "200 OK with transcript text and duration"]
    ]
    builder.add_table(api_headers, api_data, col_widths=[95, 60, 140, 145], caption="Table 4.2: FastAPI Operational Endpoints and Request Contracts")

    builder.add_division_heading("4.4. Strict Fail-Closed Isolation Contract")
    builder.add_paragraph(
        "A key design decision in our application is fail-closed security. The live web application executes exclusively on the canonical quantum model (qsvc_q8_r1). Classical models exist solely for offline research benchmarking. If an incoming API request specifies a classical model identifier (e.g., 'rbf_svm'), the backend explicitly raises an exception and returns HTTP 400 Bad Request. There is no silent classical fallback; if quantum model artifacts are missing, the system strictly returns HTTP 503 Service Unavailable."
    )

    builder.add_division_heading("4.5. Interactive User Interface and Demonstration Workflow")
    builder.add_paragraph(
        "The frontend is implemented with modern HTML5, CSS3, and JavaScript (ES2022). It includes real-time audio visualizers, interactive speaker selection, and an 8-qubit statevector orbital precession HUD that visually illustrates quantum state rotations during inference. Figures 4.2 and 4.3 show the main command center dashboard and the interactive audio lab interface."
    )

    builder.add_figure("reports/application/figures/01_overview_dashboard.png", "Figure 4.2: Quantum Research Command Center overview dashboard displaying system vitals and model status.")
    builder.add_figure("reports/application/figures/02_quantum_audio_lab.png", "Figure 4.3: Quantum Audio Lab 5-stage audio ingestion, feature extraction, and prediction interface.")
    builder.add_figure("reports/application/figures/03_model_speaker_registry.png", "Figure 4.4: Model and speaker registry displaying verified TORGO speaker profiles and operational quantum status.")
    builder.add_figure("reports/application/figures/04_research_results_explorer.png", "Figure 4.5: Interactive research results explorer displaying empirical metrics and statistical tests.")

    builder.add_division_heading("4.6. Software Verification, Automated Testing, and Safety Gates")
    builder.add_paragraph(
        "We validated our implementation with an automated test suite containing 47 targeted Gate B.2 tests (located in tests/gate_b2/). The suite tests model weight immutability, speaker isolation, rejection of classical model requests (HTTP 400), audio file validation, and WebSocket telemetry. All 47 tests pass 100% in 28.9 seconds."
    )
    builder.add_paragraph(
        "To ensure responsible engineering, we established clear safety gate boundaries: Gate B.2 (Engineering Prototype) is OPERATIONAL. However, Gate C (Independent Clinical Pre-Registration) and Gate D (Automated Emergency Dispatch Deployment) remain formally BLOCKED. Because the False Emergency Rate is 25.77%, the system must not be deployed for autonomous emergency calls without human verification."
    )
    builder.add_page_break()


print("Chapter 4 content builder updated in student-human style.")
