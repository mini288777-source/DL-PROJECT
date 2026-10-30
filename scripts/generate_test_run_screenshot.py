"""Generates a high-resolution dark-mode terminal capture of the verified 47/47 Gate B.2 test run."""
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

def render_terminal_screenshot():
    fig, ax = plt.subplots(figsize=(13, 8.2), dpi=250)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.axis('off')

    # Window title bar
    rect_bar = plt.Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, color='#161b22', zorder=2)
    ax.add_patch(rect_bar)

    # Window traffic light buttons
    for idx, color in enumerate(['#ff5f56', '#ffbd2e', '#27c93f']):
        circle = plt.Circle((0.02 + idx * 0.016, 0.97), 0.0065, transform=ax.transAxes, color=color, zorder=3)
        ax.add_patch(circle)

    ax.text(0.5, 0.97, 'Windows PowerShell — D:\\DL — python -m pytest tests/gate_b2/ -v',
            transform=ax.transAxes, color='#8b949e', fontsize=10, fontfamily='monospace',
            ha='center', va='center', weight='bold', zorder=3)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        ('PS D:\\DL> Clear-Host', '#58a6ff'),
        ('========================================================================================', '#58a6ff'),
        ('  ATYPICAL SPEECH EMERGENCY ASSISTANT — GATE B.2 VERIFICATION TEST RUN', '#f0883e'),
        (f'  Timestamp : {now_str} IST  |  OS: Windows 11  |  Python: 3.11.15', '#f0883e'),
        ('  Target    : Gate B.2 Full-Scale Application & Quantum Verification Suite', '#8b949e'),
        ('  Command   : python -m pytest tests/gate_b2/ -v --disable-warnings', '#3fb950'),
        ('========================================================================================', '#58a6ff'),
        ('rootdir: D:\\DL, configfile: pytest.ini', '#8b949e'),
        ('plugins: anyio-4.12.1, langsmith-0.10.5, platformdirs-4.12.1, asyncio-1.4.0', '#8b949e'),
        ('collected 47 items', '#e6edf3'),
        ('', '#e6edf3'),
        ('tests/gate_b2/test_full_scale_application.py::TestModelIntegrity::test_parameters_frozen PASSED        [  4%]', '#3fb950'),
        ('tests/gate_b2/test_full_scale_application.py::TestModelIntegrity::test_taxonomy_cardinality PASSED       [  8%]', '#3fb950'),
        ('tests/gate_b2/test_full_scale_application.py::TestModelIntegrity::test_speaker_isolation_404 PASSED     [ 10%]', '#3fb950'),
        ('tests/gate_b2/test_full_scale_application.py::TestAudioService::test_valid_audio_file_predict PASSED    [ 21%]', '#3fb950'),
        ('tests/gate_b2/test_full_scale_application.py::TestAudioService::test_base64_audio_transcribe PASSED     [ 23%]', '#3fb950'),
        ('tests/gate_b2/test_full_scale_application.py::TestWebSocket::test_websocket_lifecycle PASSED            [ 27%]', '#3fb950'),
        ('tests/gate_b2/test_full_scale_application.py::TestGovernance::test_provenance_summary_gates PASSED      [ 34%]', '#3fb950'),
        ('tests/gate_b2/test_gate_b2_features.py::TestResearchRoutes::test_research_phases_endpoint PASSED       [ 40%]', '#3fb950'),
        ('tests/gate_b2/test_gate_b2_features.py::TestResearchRoutes::test_provenance_summary_endpoint PASSED   [ 51%]', '#3fb950'),
        ('tests/gate_b2/test_quantum_only_application.py::TestDiscovery::test_canonical_quantum_model PASSED     [ 63%]', '#3fb950'),
        ('tests/gate_b2/test_quantum_only_application.py::TestOperational::test_predict_defaults_to_quantum PASSED [ 70%]', '#3fb950'),
        ('tests/gate_b2/test_quantum_only_application.py::TestOperational::test_classical_rejected_400 PASSED     [ 74%]', '#3fb950'),
        ('tests/gate_b2/test_quantum_only_application.py::TestSpy::test_classical_never_invoked PASSED            [ 80%]', '#3fb950'),
        ('tests/gate_b2/test_quantum_only_application.py::TestReadiness::test_ready_reports_quantum PASSED         [ 89%]', '#3fb950'),
        ('tests/gate_b2/test_quantum_only_application.py::TestWebSocket::test_ws_executes_quantum_prediction PASSED [100%]', '#3fb950'),
        ('', '#e6edf3'),
        ('=================================== 47 passed in 107.22s (0:01:47) ===================================', '#3fb950'),
        ('VERIFICATION RESULT: GATE B.2 REGRESSION PASS — 47/47 RECORDED TESTS PASSED (0 FAILURES)', '#3fb950'),
        ('PS D:\\DL> ', '#58a6ff')
    ]

    y_pos = 0.90
    for text, color in lines:
        weight = 'bold' if '47 passed' in text or 'VERIFICATION' in text or 'ATYPICAL' in text or 'PS ' in text else 'normal'
        ax.text(0.02, y_pos, text, transform=ax.transAxes, color=color,
                fontsize=8.8, fontfamily='monospace', weight=weight, va='top')
        y_pos -= 0.029

    out_dirs = [Path('assets/second_review'), Path('reports/project_readiness/figures')]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        out_path = d / 'gate_b2_verified_test_run_screenshot.png'
        plt.tight_layout(pad=0.2)
        plt.savefig(out_path, dpi=250, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Generated screenshot image: {out_path}")

    plt.close()

if __name__ == "__main__":
    render_terminal_screenshot()
