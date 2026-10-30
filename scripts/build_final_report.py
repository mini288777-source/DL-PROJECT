"""
Master Builder Script for Assembling the Complete Final Project Report.

Generates:
1. D:\DL\reports\final\APPY_FIZZ_Final_Project_Report.docx
2. D:\DL\reports\final\APPY_FIZZ_Final_Project_Report.pdf

Follows the official Amrita School of Computing template format,
the audited empirical evidence, and the complete 18-section content blueprint.
"""

import os
import sys
import time
from pathlib import Path
import win32com.client

REPO_ROOT = Path(r"D:\DL")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
OUTPUT_DIR = REPO_ROOT / "reports" / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCX_OUTPUT = OUTPUT_DIR / "APPY_FIZZ_Final_Project_Report.docx"
PDF_OUTPUT = OUTPUT_DIR / "APPY_FIZZ_Final_Project_Report.pdf"

WD_ALIGN_PARAGRAPH_LEFT = 0
WD_ALIGN_PARAGRAPH_CENTER = 1
WD_ALIGN_PARAGRAPH_RIGHT = 2
WD_ALIGN_PARAGRAPH_JUSTIFY = 3

WD_PAGE_BREAK = 7
WD_FORMAT_DOCUMENT_DEFAULT = 16
WD_FORMAT_PDF = 17

ONE_CM_PTS = 28.35
ONE_INCH_PTS = 72.0
MAX_CONTENT_WIDTH = 440.0


class ReportBuilder:
    def __init__(self):
        print("Launching Microsoft Word COM Automation...")
        self.word = win32com.client.Dispatch("Word.Application")
        self.word.Visible = False
        self.doc = self.word.Documents.Add()
        self._configure_page_setup()
        
    def _configure_page_setup(self):
        ps = self.doc.PageSetup
        ps.PaperSize = 7 # wdPaperA4
        ps.TopMargin = ONE_INCH_PTS
        ps.BottomMargin = ONE_INCH_PTS
        ps.LeftMargin = ONE_INCH_PTS
        ps.RightMargin = ONE_INCH_PTS
        ps.HeaderDistance = 36.0
        ps.FooterDistance = 36.0

    def add_page_break(self):
        p = self.doc.Paragraphs.Add()
        p.Range.InsertBreak(WD_PAGE_BREAK)

    def add_chapter_heading(self, chapter_num, chapter_title):
        p1 = self.doc.Paragraphs.Add()
        p1.Range.Text = f"CHAPTER {chapter_num}\n"
        p1.Range.Font.Name = "Times New Roman"
        p1.Range.Font.Size = 14
        p1.Range.Font.Bold = True
        p1.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
        p1.Range.ParagraphFormat.SpaceBefore = 18
        p1.Range.ParagraphFormat.SpaceAfter = 6
        p1.Range.ParagraphFormat.LineSpacingRule = 0
        p1.Range.ParagraphFormat.LineSpacing = 18

        p2 = self.doc.Paragraphs.Add()
        p2.Range.Text = f"{chapter_title.upper()}\n"
        p2.Range.Font.Name = "Times New Roman"
        p2.Range.Font.Size = 14
        p2.Range.Font.Bold = True
        p2.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
        p2.Range.ParagraphFormat.SpaceBefore = 0
        p2.Range.ParagraphFormat.SpaceAfter = 20
        p2.Range.ParagraphFormat.LineSpacingRule = 0
        p2.Range.ParagraphFormat.LineSpacing = 18

    def add_preliminary_heading(self, title):
        p = self.doc.Paragraphs.Add()
        p.Range.Text = f"{title.upper()}\n"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 14
        p.Range.Font.Bold = True
        p.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
        p.Range.ParagraphFormat.SpaceBefore = 18
        p.Range.ParagraphFormat.SpaceAfter = 18
        p.Range.ParagraphFormat.LineSpacingRule = 0
        p.Range.ParagraphFormat.LineSpacing = 18

    def add_division_heading(self, heading_text):
        p = self.doc.Paragraphs.Add()
        p.Range.Text = f"{heading_text.upper()}\n"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = True
        p.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_LEFT
        p.Range.ParagraphFormat.FirstLineIndent = 0
        p.Range.ParagraphFormat.SpaceBefore = 14
        p.Range.ParagraphFormat.SpaceAfter = 6
        p.Range.ParagraphFormat.LineSpacingRule = 0
        p.Range.ParagraphFormat.LineSpacing = 18

    def add_subdivision_heading(self, heading_text):
        p = self.doc.Paragraphs.Add()
        p.Range.Text = f"{heading_text}\n"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = True
        p.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_LEFT
        p.Range.ParagraphFormat.FirstLineIndent = 0
        p.Range.ParagraphFormat.SpaceBefore = 10
        p.Range.ParagraphFormat.SpaceAfter = 4
        p.Range.ParagraphFormat.LineSpacingRule = 0
        p.Range.ParagraphFormat.LineSpacing = 18

    def add_paragraph(self, text, indent=True):
        p = self.doc.Paragraphs.Add()
        p.Range.Text = f"{text}\n"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = 12
        p.Range.Font.Bold = False
        p.Range.Font.Italic = False
        p.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_JUSTIFY
        p.Range.ParagraphFormat.FirstLineIndent = ONE_CM_PTS if indent else 0
        p.Range.ParagraphFormat.SpaceBefore = 0
        p.Range.ParagraphFormat.SpaceAfter = 6
        p.Range.ParagraphFormat.LineSpacingRule = 0
        p.Range.ParagraphFormat.LineSpacing = 18

    def add_centered_text(self, text, font_size=12, bold=False, italic=False, space_before=0, space_after=6):
        p = self.doc.Paragraphs.Add()
        p.Range.Text = f"{text}\n"
        p.Range.Font.Name = "Times New Roman"
        p.Range.Font.Size = font_size
        p.Range.Font.Bold = bold
        p.Range.Font.Italic = italic
        p.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
        p.Range.ParagraphFormat.FirstLineIndent = 0
        p.Range.ParagraphFormat.SpaceBefore = space_before
        p.Range.ParagraphFormat.SpaceAfter = space_after
        p.Range.ParagraphFormat.LineSpacingRule = 0
        p.Range.ParagraphFormat.LineSpacing = font_size * 1.5

    def add_table(self, headers, data, col_widths=None, caption=None):
        if caption:
            p_cap = self.doc.Paragraphs.Add()
            p_cap.Range.Text = f"{caption}\n"
            p_cap.Range.Font.Name = "Times New Roman"
            p_cap.Range.Font.Size = 11
            p_cap.Range.Font.Bold = True
            p_cap.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
            p_cap.Range.ParagraphFormat.FirstLineIndent = 0
            p_cap.Range.ParagraphFormat.SpaceBefore = 8
            p_cap.Range.ParagraphFormat.SpaceAfter = 4

        p = self.doc.Paragraphs.Add()
        num_rows = len(data) + 1
        num_cols = len(headers)
        table = self.doc.Tables.Add(p.Range, num_rows, num_cols)
        table.Borders.Enable = True
        table.Borders.InsideLineStyle = 1
        table.Borders.OutsideLineStyle = 1
        table.Rows.Alignment = WD_ALIGN_PARAGRAPH_CENTER

        for col_idx, h_text in enumerate(headers, start=1):
            cell = table.Cell(1, col_idx)
            cell.Range.Text = str(h_text)
            cell.Range.Font.Name = "Times New Roman"
            cell.Range.Font.Size = 10
            cell.Range.Font.Bold = True
            cell.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
            cell.Range.ParagraphFormat.SpaceBefore = 3
            cell.Range.ParagraphFormat.SpaceAfter = 3
            cell.Shading.BackgroundPatternColor = 0xF2F2F2

        for row_idx, row_data in enumerate(data, start=2):
            for col_idx, val in enumerate(row_data, start=1):
                cell = table.Cell(row_idx, col_idx)
                cell.Range.Text = str(val)
                cell.Range.Font.Name = "Times New Roman"
                cell.Range.Font.Size = 9.5
                cell.Range.Font.Bold = False
                cell.Range.ParagraphFormat.SpaceBefore = 2
                cell.Range.ParagraphFormat.SpaceAfter = 2
                if col_idx == 1 and num_cols > 3:
                    cell.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_LEFT
                else:
                    cell.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER

        if col_widths and len(col_widths) == num_cols:
            for col_idx, w in enumerate(col_widths, start=1):
                table.Columns(col_idx).Width = w

        p_after = self.doc.Paragraphs.Add()
        p_after.Range.ParagraphFormat.SpaceBefore = 0
        p_after.Range.ParagraphFormat.SpaceAfter = 6

    def add_figure(self, image_path, caption, max_width=MAX_CONTENT_WIDTH, max_height=260):
        abs_path = str(REPO_ROOT / image_path) if not os.path.isabs(image_path) else image_path
        if not os.path.exists(abs_path):
            print(f"Warning: Figure not found at {abs_path}")
            return

        p_img = self.doc.Paragraphs.Add()
        p_img.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
        p_img.Range.ParagraphFormat.FirstLineIndent = 0
        p_img.Range.ParagraphFormat.SpaceBefore = 8
        p_img.Range.ParagraphFormat.SpaceAfter = 4

        try:
            shape = self.doc.InlineShapes.AddPicture(
                FileName=abs_path,
                LinkToFile=False,
                SaveWithDocument=True,
                Range=p_img.Range
            )
            orig_w = shape.Width
            orig_h = shape.Height
            aspect = orig_w / max(orig_h, 1.0)
            
            target_w = min(orig_w, max_width)
            target_h = target_w / aspect
            if target_h > max_height:
                target_h = max_height
                target_w = target_h * aspect

            shape.Width = target_w
            shape.Height = target_h
        except Exception as e:
            print(f"Error adding picture {abs_path}: {e}")

        p_cap = self.doc.Paragraphs.Add()
        p_cap.Range.Text = f"{caption}\n"
        p_cap.Range.Font.Name = "Times New Roman"
        p_cap.Range.Font.Size = 10.5
        p_cap.Range.Font.Italic = True
        p_cap.Range.Font.Bold = False
        p_cap.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
        p_cap.Range.ParagraphFormat.FirstLineIndent = 0
        p_cap.Range.ParagraphFormat.SpaceBefore = 2
        p_cap.Range.ParagraphFormat.SpaceAfter = 12

    def save_and_export(self):
        print(f"Saving Word document to: {DOCX_OUTPUT}")
        self.doc.SaveAs2(str(DOCX_OUTPUT), WD_FORMAT_DOCUMENT_DEFAULT)
        print(f"Exporting PDF document to: {PDF_OUTPUT}")
        self.doc.SaveAs2(str(PDF_OUTPUT), WD_FORMAT_PDF)
        self.doc.Close(False)
        self.word.Quit()
        print("Report generation successfully completed!")


def build_preliminary_pages(builder: ReportBuilder):
    print("Building Preliminary Pages...")

    # ==========================================
    # 1. TITLE PAGE (Specimen 1)
    # ==========================================
    builder.add_centered_text("LOW-RESOURCE PERSONALIZED ATYPICAL SPEECH RECOGNITION USING HYBRID QUANTUM-CLASSICAL DEEP LEARNING FOR RELIABLE EMERGENCY COMMUNICATION", font_size=15, bold=True, space_before=24, space_after=36)
    builder.add_centered_text("A PROJECT REPORT", font_size=13, bold=True, space_after=18)
    builder.add_centered_text("Submitted by", font_size=12, italic=True, space_after=18)
    builder.add_centered_text("[STUDENT_NAME_1] (Reg. No. [REG_NO_1])\n[STUDENT_NAME_2] (Reg. No. [REG_NO_2])\n[STUDENT_NAME_3] (Reg. No. [REG_NO_3])", font_size=12, bold=True, space_after=24)
    builder.add_centered_text("in partial fulfillment for the award of the degree of", font_size=12, italic=True, space_after=18)
    builder.add_centered_text("BACHELOR OF TECHNOLOGY\nIN\nCOMPUTER SCIENCE AND ENGINEERING\n(ARTIFICIAL INTELLIGENCE)", font_size=13, bold=True, space_after=24)
    builder.add_centered_text("Under the guidance of", font_size=12, italic=True, space_after=12)
    builder.add_centered_text("[SUPERVISOR_NAME_AND_DESIGNATION]", font_size=12, bold=True, space_after=30)
    builder.add_centered_text("AMRITA VISHWA VIDYAPEETHAM\nAMRITA SCHOOL OF COMPUTING\nCHENNAI – 601 103, TAMIL NADU, INDIA", font_size=13, bold=True, space_after=18)
    builder.add_centered_text("NOVEMBER 2026", font_size=12, bold=True, space_after=0)
    builder.add_page_break()

    # ==========================================
    # 2. COVER PAGE (Duplicate of Title Page)
    # ==========================================
    builder.add_centered_text("LOW-RESOURCE PERSONALIZED ATYPICAL SPEECH RECOGNITION USING HYBRID QUANTUM-CLASSICAL DEEP LEARNING FOR RELIABLE EMERGENCY COMMUNICATION", font_size=15, bold=True, space_before=24, space_after=36)
    builder.add_centered_text("A PROJECT REPORT", font_size=13, bold=True, space_after=18)
    builder.add_centered_text("Submitted by", font_size=12, italic=True, space_after=18)
    builder.add_centered_text("[STUDENT_NAME_1] (Reg. No. [REG_NO_1])\n[STUDENT_NAME_2] (Reg. No. [REG_NO_2])\n[STUDENT_NAME_3] (Reg. No. [REG_NO_3])", font_size=12, bold=True, space_after=24)
    builder.add_centered_text("in partial fulfillment for the award of the degree of", font_size=12, italic=True, space_after=18)
    builder.add_centered_text("BACHELOR OF TECHNOLOGY\nIN\nCOMPUTER SCIENCE AND ENGINEERING\n(ARTIFICIAL INTELLIGENCE)", font_size=13, bold=True, space_after=24)
    builder.add_centered_text("Under the guidance of", font_size=12, italic=True, space_after=12)
    builder.add_centered_text("[SUPERVISOR_NAME_AND_DESIGNATION]", font_size=12, bold=True, space_after=30)
    builder.add_centered_text("AMRITA VISHWA VIDYAPEETHAM\nAMRITA SCHOOL OF COMPUTING\nCHENNAI – 601 103, TAMIL NADU, INDIA", font_size=13, bold=True, space_after=18)
    builder.add_centered_text("NOVEMBER 2026", font_size=12, bold=True, space_after=0)
    builder.add_page_break()

    # ==========================================
    # 3. BONAFIDE CERTIFICATE (Specimen 2)
    # ==========================================
    builder.add_preliminary_heading("BONAFIDE CERTIFICATE")
    builder.add_paragraph(
        'This is to Certify that this project report entitled "LOW-RESOURCE PERSONALIZED ATYPICAL SPEECH RECOGNITION USING HYBRID QUANTUM-CLASSICAL DEEP LEARNING FOR RELIABLE EMERGENCY COMMUNICATION" is the bonafide work of [STUDENT_NAME_1] ([REG_NO_1]), [STUDENT_NAME_2] ([REG_NO_2]), and [STUDENT_NAME_3] ([REG_NO_3]), who carried out the project work under my supervision.',
        indent=True
    )
    builder.add_paragraph(
        'This report represents the original work of the candidates and has been prepared in accordance with the academic guidelines prescribed by Amrita Vishwa Vidyapeetham for the degree of Bachelor of Technology in Computer Science and Engineering (Artificial Intelligence).',
        indent=True
    )
    
    # Signatures table
    sig_headers = ["SUPERVISOR", "PROGRAM HEAD"]
    sig_data = [
        [
            "SIGNATURE: _____________________\n\n[SUPERVISOR_NAME]\nSupervisor & Associate Professor,\nDepartment of CSE,\nAmrita School of Computing,\nChennai – 601 103.",
            "SIGNATURE: _____________________\n\n[PROGRAM_HEAD_NAME]\nProgram Head, Department of CSE,\nAmrita School of Computing,\nChennai – 601 103."
        ],
        [
            "INTERNAL EXAMINER:\nSIGNATURE: _____________________\n\nDate: ________________________",
            "EXTERNAL EXAMINER:\nSIGNATURE: _____________________\n\nDate: ________________________"
        ]
    ]
    builder.add_table(sig_headers, sig_data, col_widths=[220, 220])
    builder.add_page_break()

    # ==========================================
    # 4. DECLARATION BY CANDIDATE (Specimen 3)
    # ==========================================
    builder.add_preliminary_heading("DECLARATION BY THE CANDIDATE")
    builder.add_paragraph(
        'We declare that the project report entitled "LOW-RESOURCE PERSONALIZED ATYPICAL SPEECH RECOGNITION USING HYBRID QUANTUM-CLASSICAL DEEP LEARNING FOR RELIABLE EMERGENCY COMMUNICATION" submitted by us for the degree of Bachelor of Technology in Computer Science and Engineering (Artificial Intelligence) is the record of the project work carried out by us under the guidance of [SUPERVISOR_NAME_AND_DESIGNATION], Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai.',
        indent=True
    )
    builder.add_paragraph(
        'We further declare that this work has not formed the basis for the award of any degree, diploma, associateship, fellowship, or similar title in this or any other University or Institution of higher learning.',
        indent=True
    )
    
    cand_headers = ["CANDIDATE NAME", "REGISTER NUMBER", "SIGNATURE"]
    cand_data = [
        ["[STUDENT_NAME_1]", "[REG_NO_1]", "________________________"],
        ["[STUDENT_NAME_2]", "[REG_NO_2]", "________________________"],
        ["[STUDENT_NAME_3]", "[REG_NO_3]", "________________________"],
    ]
    builder.add_table(cand_headers, cand_data, col_widths=[160, 140, 140])
    builder.add_centered_text("\nPlace: Chennai\nDate: November 2026", font_size=12, space_before=18)
    builder.add_page_break()

    # ==========================================
    # 5. ABSTRACT (Specimen 4)
    # ==========================================
    builder.add_preliminary_heading("ABSTRACT")
    builder.add_paragraph(
        "Atypical and dysarthric speech, resulting from neuromotor conditions such as cerebral palsy, amyotrophic lateral sclerosis (ALS), Parkinson's disease, and stroke, exhibits severe acoustic distortions including articulatory imprecision, hypernasality, pitch irregularities, and reduced vocal intensity. Consequently, commercial automatic speech recognition (ASR) systems exhibit catastrophic failure rates, frequently exceeding 60% to 80% word error rates, rendering voice-driven emergency alerting inaccessible for affected individuals. While speaker personalization provides a viable pathway to restore communication efficacy, individuals with severe motor-speech impairments experience acute vocal fatigue, restricting speaker calibration data to extreme few-shot sample budgets (N in {20, 30, 40, 50} utterances).",
        indent=True
    )
    builder.add_paragraph(
        "This project presents a rigorous experimental investigation evaluating whether quantum machine learning—specifically an 8-qubit quantum statevector fidelity kernel utilizing a parameter-free ZZFeatureMap paired with a Support Vector Classifier (qsvc_q8_r1)—can provide a measurable, statistically defensible advantage in few-shot personalization over matched classical kernels (RBF-SVM) and lightweight neural adaptation baselines. Upstream acoustic representations are extracted using a frozen, self-supervised Whisper-tiny foundation encoder to produce 384-dimensional bottleneck latent embeddings, which are subsequently projected into an 8-dimensional Principal Component Analysis (PCA) space retaining 89.73% cumulative variance. Evaluated across the 15-speaker TORGO atypical speech corpus (8 dysarthric, 7 control) under a strictly matched 60-condition protocol (15 speakers x 4 sample budgets), the empirical findings demonstrate that the quantum model achieved an End-to-End Macro-F1 of 0.1051 +/- 0.0575 compared to 0.1087 +/- 0.0487 for the classical RBF-SVM baseline (p = 0.3428, Wilcoxon signed-rank test with Holm-Bonferroni correction). Emergency recall was 27.52% +/- 22.84% for QSVC vs. 31.89% +/- 16.42% for RBF-SVM (p = 0.0689). Under exact classical statevector CPU simulation, the quantum kernel incurred a 772.6x per-sample test latency penalty and a 212.8x experiment wall-clock cost compared to the classical baseline. The authoritative scientific verdict confirms Case B: No statistically supported quantum advantage under the evaluated representation and few-shot constraints.",
        indent=True
    )
    builder.add_paragraph(
        "To demonstrate the operational workflow, a production-grade Quantum Research Command Center web application was engineered, exposing the canonical qsvc_q8_r1 model exclusively while strictly rejecting classical models (HTTP 400) and enforcing zero classical fallback. Owing to naturalistic class imbalance (approximately 95.8% general speech prevalence) and high false-emergency rates (28.12%), autonomous emergency dispatch remains infeasible, and external deployment gates (Gate C and Gate D) remain formally blocked. This research establishes an irreproachably transparent, reproducible foundation for low-resource assistive bio-acoustic machine learning.",
        indent=True
    )
    builder.add_paragraph(
        "Keywords: Atypical Speech Recognition, Dysarthria, Quantum Machine Learning, Quantum Kernel Estimation, Few-Shot Personalization, Whisper Embeddings, Emergency Intent Classification, Safety Governance.",
        indent=False
    )
    builder.add_page_break()

    # ==========================================
    # 6. ACKNOWLEDGEMENT (Specimen 5)
    # ==========================================
    builder.add_preliminary_heading("ACKNOWLEDGEMENT")
    builder.add_paragraph(
        "This project work would not have been possible without the guidance, encouragement, and support of many esteemed individuals and institutions. It gives us immense pleasure to express our profound gratitude to our honorable Chancellor, Sri Mata Amritanandamayi Devi, for her divine blessings, grace, and for being a continuous source of inspiration throughout our academic journey.",
        indent=True
    )
    builder.add_paragraph(
        "We are deeply indebted to extend our gratitude to our Director, Mr. I B Manikantan, Amrita School of Computing, for facilitating the state-of-the-art computational infrastructure, laboratories, and extended administrative support that enabled us to gain valuable learning experiences and conduct advanced machine learning research.",
        indent=True
    )
    builder.add_paragraph(
        "We register our special thanks to Dr. V. Jayakumar, Principal, Amrita School of Computing, for his administrative support, academic leadership, and encouragement during the successful conduct of this final year project.",
        indent=True
    )
    builder.add_paragraph(
        "We wish to express our heartfelt and sincere gratitude to our project supervisor, [SUPERVISOR_NAME_AND_DESIGNATION], Department of Computer Science and Engineering, for their invaluable guidance, intellectual rigor, constant encouragement, and insightful feedback during every phase of this investigation.",
        indent=True
    )
    builder.add_paragraph(
        "We express our gratitude to the Program Head, Department of Computer Science and Engineering (Artificial Intelligence), for his inspiring vision, academic direction, and continuous support. We also extend our sincere appreciation to the Project Coordinators, Review Panel Members, and the entire faculty of the Department of Computer Science and Engineering for their constructive critiques, critical questioning, and valuable recommendations, which significantly elevated the quality, transparency, and scientific integrity of this research report.",
        indent=True
    )
    builder.add_centered_text("[STUDENT_NAME_1]\n[STUDENT_NAME_2]\n[STUDENT_NAME_3]\n(APPY FIZZ Team)", font_size=12, bold=True, space_before=18)
    builder.add_page_break()

    # ==========================================
    # 7. TABLE OF CONTENTS (Specimen 6)
    # ==========================================
    builder.add_preliminary_heading("TABLE OF CONTENTS")
    toc_headers = ["CHAPTER NO.", "TITLE", "PAGE NO."]
    toc_data = [
        ["", "ABSTRACT", "v"],
        ["", "ACKNOWLEDGEMENT", "vi"],
        ["", "LIST OF TABLES", "viii"],
        ["", "LIST OF FIGURES", "ix"],
        ["", "LIST OF SYMBOLS AND ABBREVIATIONS", "xi"],
        ["1", "INTRODUCTION", "1"],
        ["1.1", "Background and Motivation", "1"],
        ["1.2", "Neuromotor Speech Disorders and Dysarthria", "3"],
        ["1.3", "Challenges of Low-Resource Personalization", "5"],
        ["1.4", "Emergency-Related Speech Classification and Safety Imperatives", "7"],
        ["1.5", "Quantum Machine Learning: Hypotheses vs. Empirical Realities", "9"],
        ["1.6", "Problem Definition", "11"],
        ["1.7", "Research Aim and Specific Objectives", "12"],
        ["1.8", "Scope and Limitations", "14"],
        ["1.9", "Summary of Main Contributions", "15"],
        ["1.10", "Organization of the Report", "17"],
        ["2", "LITERATURE REVIEW / PROBLEM BACKGROUND", "19"],
        ["2.1", "Atypical and Dysarthric Speech Recognition", "19"],
        ["2.2", "Acoustic Feature Extraction: From MFCCs to Whisper Embeddings", "21"],
        ["2.3", "Few-Shot Learning and Personalization in Bio-Acoustics", "24"],
        ["2.4", "Classical Classification Baselines in Low-Sample Regimes", "26"],
        ["2.5", "Principles of Quantum Machine Learning and Quantum Kernels", "28"],
        ["2.6", "Critical Examination of Quantum Advantage Claims", "31"],
        ["2.7", "Structured Literature Survey Comparison Table", "33"],
        ["2.8", "Critical Synthesis of Literature", "35"],
        ["2.9", "Identification of the Core Research Gap", "37"],
        ["2.10", "Position and Methodological Stance of the Current Work", "39"],
        ["3", "PROBLEM STATEMENT AND METHODOLOGY", "41"],
        ["3.1", "Problem Formulation and Formal Hypotheses", "41"],
        ["3.2", "The TORGO Dysarthric Speech Database", "43"],
        ["3.3", "Proxy Emergency Intent Taxonomy Formulation", "46"],
        ["3.4", "Acoustic Preprocessing and Frozen Whisper Feature Extraction", "48"],
        ["3.5", "Dimensionality Reduction and Representation Space Audit", "51"],
        ["3.6", "Classical Baseline Model Formulations", "54"],
        ["3.7", "Quantum Kernel Formulation and Circuit Topology", "57"],
        ["3.8", "Personalization Calibration Protocol and Data Leakage Controls", "60"],
        ["3.9", "Evaluation Metrics Formulation", "62"],
        ["3.10", "Statistical Hypothesis Testing and Multiplicity Control", "65"],
        ["3.11", "Hardware and Software Execution Environment", "67"],
        ["4", "SYSTEM DESIGN, EXPERIMENTAL WORK AND IMPLEMENTATION", "69"],
        ["4.1", "Overall Research Architecture and Longitudinal Workflow", "69"],
        ["4.2", "Longitudinal Experimental Progression: Track 1 vs. Track 2", "72"],
        ["4.3", "Environmental Noise Robustness Protocol", "75"],
        ["4.4", "Continual Adaptation and Safety Governance Framework", "77"],
        ["4.5", "Quantum-Only Operational Command Center Architecture", "80"],
        ["4.6", "Strict Separation Contract: Research Baselines vs. Live Operational Classifier", "83"],
        ["4.7", "Interactive Web Interface Implementation", "85"],
        ["4.8", "Verified Application Walkthrough and Visual Telemetry", "88"],
        ["4.9", "Software Verification, Automated Regression Suite, and Gate Status", "91"],
        ["5", "RESULTS AND DISCUSSIONS", "94"],
        ["5.1", "Historical Benchmark Results and Majority Collapse Diagnosis", "94"],
        ["5.2", "Definitive Matched Quantum vs. Classical Baseline Evaluation", "97"],
        ["5.3", "Few-Shot Calibration Sample-Budget Trajectories", "101"],
        ["5.4", "Acoustic Noise Perturbation Robustness Findings", "104"],
        ["5.5", "Secondary Baseline Comparison: Weighted Adapter vs. QSVC", "107"],
        ["5.6", "Quantum Architecture Ablations: Qubits and Circuit Depth", "110"],
        ["5.7", "Computational Complexity and Statevector Latency Audit", "113"],
        ["5.8", "Speaker Heterogeneity and Individual Variance", "116"],
        ["5.9", "Forensic Metric Reconciliation and Historical Reporting Audits", "119"],
        ["5.10", "Out-of-Fold Calibration vs. Test FER Generalization Gap", "122"],
        ["5.11", "Exhaustive Disclosure of Research Limitations", "125"],
        ["5.12", "Canonical Scientific Interpretation: Case B Affirmation", "128"],
        ["6", "CONCLUSIONS AND SCOPE FOR FURTHER WORK", "131"],
        ["6.1", "Summary of Completed Research", "131"],
        ["6.2", "Principal Empirical Conclusions", "133"],
        ["6.3", "Engineering Contributions", "135"],
        ["6.4", "Limitations of the Current Study", "137"],
        ["6.5", "Scope for Future Work", "138"],
        ["6.6", "Concluding Remarks", "140"],
        ["", "REFERENCES", "142"],
        ["", "APPENDICES", "147"],
        ["A", "Mathematical Derivations and Quantum Circuit Topology", "147"],
        ["B", "Hyperparameter and Configuration Registry", "150"],
        ["C", "Detailed Speaker-Wise Experimental Performance Tables", "153"],
        ["D", "Software Environment, Dependency Manifest, and Hardware Specifications", "156"],
        ["E", "Plagiarism Clearance and Institutional Certification Notice", "159"]
    ]
    builder.add_table(toc_headers, toc_data, col_widths=[80, 290, 70])
    builder.add_page_break()

    # ==========================================
    # 8. LIST OF TABLES (Specimen 7)
    # ==========================================
    builder.add_preliminary_heading("LIST OF TABLES")
    lot_headers = ["TABLE NO.", "TITLE", "PAGE NO."]
    lot_data = [
        ["2.1", "Structured Literature Survey in Dysarthric Speech Recognition and QML", "34"],
        ["3.1", "Demographic, Clinical, and Acoustic Profile of the 15-Speaker TORGO Cohort", "45"],
        ["3.2", "Canonical 7-Class Emergency Intent Taxonomy and Prevalence Distribution", "47"],
        ["3.3", "Cumulative Explained Variance across PCA Dimensions (d in [2, 64])", "53"],
        ["4.1", "Classical Baseline and Quantum Model Architecture Specifications", "82"],
        ["4.2", "FastAPI Operational Endpoints, Request Schemas, and Error Contracts", "84"],
        ["5.1", "Performance Summary of the Original Flat 7-Class Benchmark", "96"],
        ["5.2", "Definitive Matched Comparison: QSVC Q8-R1 vs. Classical RBF-SVM (d=8)", "99"],
        ["5.3", "Secondary Model Comparison: QSVC Q8-R1 vs. Weighted Classical Adapter", "108"],
        ["5.4", "Quantum Architecture Ablation: Qubit Scaling and Variational Circuit Depth", "111"],
        ["5.5", "Computational Latency and Wall-Clock Execution Time Audit", "114"],
        ["5.6", "Forensic Metric Reconciliation and Historical Reporting Corrections", "120"],
        ["5.7", "Macro-F1 and Intent Preservation across Acoustic Noise Perturbations", "105"]
    ]
    builder.add_table(lot_headers, lot_data, col_widths=[80, 290, 70])
    builder.add_page_break()

    # ==========================================
    # 9. LIST OF FIGURES (Specimen 8)
    # ==========================================
    builder.add_preliminary_heading("LIST OF FIGURES")
    lof_headers = ["FIGURE NO.", "TITLE", "PAGE NO."]
    lof_data = [
        ["3.1", "Acoustic Preprocessing and Log-Mel Spectrogram Extraction Pipeline", "49"],
        ["3.2", "Whisper-Tiny Bottleneck Feature Extraction and Temporal Mean Pooling", "50"],
        ["4.1", "Longitudinal Research Architecture: Track 1 Benchmark to Track 2 Recovery", "71"],
        ["4.2", "Cumulative Explained Variance vs. PCA Feature Dimensionality Curve", "52"],
        ["4.3", "Eigenvalue Decay Spectrum of the 8-Qubit Quantum Fidelity Kernel Matrix", "59"],
        ["4.4", "Quantum Research Command Center Overview Dashboard Interface", "87"],
        ["4.5", "Quantum Audio Lab 5-Stage Ingestion and Inference Interface", "88"],
        ["4.6", "Verified Model and Speaker Registry Interface", "89"],
        ["4.7", "Interactive Research Results Explorer Interface", "89"],
        ["4.8", "Cryptographic Provenance and Checksum Verification Workspace", "90"],
        ["4.9", "Runtime System Diagnostics and Telemetry Interface", "90"],
        ["4.10", "Safety Governance, Regulatory Boundaries, and Help Interface", "91"],
        ["5.1", "Longitudinal Research Performance Trajectory across Milestones", "95"],
        ["5.2", "Primary Matched Comparison: QSVC Q8-R1 vs. Classical RBF-SVM (d=8)", "98"],
        ["5.3", "Cross-Model Emergency Recall vs. False-Emergency Rate Trade-off Curve", "109"],
        ["5.4", "Statevector CPU Simulation Latency and Wall-Clock Scaling Breakdown", "115"],
        ["5.5", "Historical Flat 7-Class Benchmark Comparison Showing Majority Collapse", "97"],
        ["5.6", "Macro-F1 Degradation across Environmental Acoustic Noise Conditions", "106"],
        ["5.7", "Intent Preservation Error Rate (IPER) Response to Acoustic Noise", "106"],
        ["5.8", "Speaker-Wise Performance Heterogeneity across 15 TORGO Speakers", "117"],
        ["5.9", "End-to-End Hierarchical Triage and Subtype Confusion Matrix", "118"],
        ["5.10", "Out-of-Fold Calibration vs. Test FER Generalization Curve", "124"]
    ]
    builder.add_table(lof_headers, lof_data, col_widths=[80, 290, 70])
    builder.add_page_break()

    # ==========================================
    # 10. LIST OF SYMBOLS AND ABBREVIATIONS (Specimen 9)
    # ==========================================
    builder.add_preliminary_heading("LIST OF SYMBOLS AND ABBREVIATIONS")
    abbr_headers = ["ABBREVIATION / SYMBOL", "DESCRIPTION / MEANING"]
    abbr_data = [
        ["ALS", "Amyotrophic Lateral Sclerosis"],
        ["ASR", "Automatic Speech Recognition"],
        ["AWGN", "Additive White Gaussian Noise"],
        ["CER", "Character Error Rate"],
        ["CPU", "Central Processing Unit"],
        ["DSP", "Digital Signal Processing"],
        ["FER", "False Emergency Rate"],
        ["FFT", "Fast Fourier Transform"],
        ["IPER", "Intent Preservation Error Rate"],
        ["KNN", "K-Nearest Neighbors"],
        ["Log-Mel", "Logarithmic Mel-Frequency Filterbank Energy"],
        ["MFCC", "Mel-Frequency Cepstral Coefficients"],
        ["MLP", "Multilayer Perceptron (Neural Adapter)"],
        ["NISQ", "Noisy Intermediate-Scale Quantum"],
        ["OOF", "Out-of-Fold (Cross-Validation Calibration)"],
        ["PCA", "Principal Component Analysis"],
        ["QML", "Quantum Machine Learning"],
        ["QPU", "Quantum Processing Unit"],
        ["QSVC", "Quantum Support Vector Classifier"],
        ["RBF", "Radial Basis Function (Gaussian Kernel)"],
        ["ReLU", "Rectified Linear Unit Activation Function"],
        ["SNR", "Signal-to-Noise Ratio (Decibels, dB)"],
        ["SVM", "Support Vector Machine"],
        ["TORGO", "Toronto-Bloorview Cerebral Palsy Speech Database"],
        ["WER", "Word Error Rate"],
        ["Whisper", "OpenAI Self-Supervised Weakly-Supervised ASR Model"],
        ["ZZFeatureMap", "Second-Order Pauli-Z Quantum Variational Feature Map"],
        ["tau", "Decision Cutoff Operating Threshold (demonstration tau = 0.6000)"],
        ["delta", "Mathematical Difference / Margin Metric"],
        ["gamma", "Kernel Bandwidth Scaling Hyperparameter"]
    ]
    builder.add_table(abbr_headers, abbr_data, col_widths=[140, 300])
    builder.add_page_break()
print("build_preliminary_pages defined.")


def main():
    print("=" * 70)
    print("STARTING FULL PROJECT REPORT ASSEMBLY: APPY FIZZ")
    print("=" * 70)
    
    # Import chapter and section builders
    from scripts.report_chapters_1_to_3 import build_chapter_1, build_chapter_2, build_chapter_3
    from scripts.report_chapters_4_to_6 import build_chapter_4, build_chapter_5, build_chapter_6
    from scripts.report_appendices_and_refs import build_references, build_appendices

    builder = ReportBuilder()
    try:
        build_preliminary_pages(builder)
        build_chapter_1(builder)
        build_chapter_2(builder)
        build_chapter_3(builder)
        build_chapter_4(builder)
        build_chapter_5(builder)
        build_chapter_6(builder)
        build_references(builder)
        build_appendices(builder)
        builder.save_and_export()
        print("=" * 70)
        print("REPORT GENERATION COMPLETE!")
        print(f"Word Document: {DOCX_OUTPUT}")
        print(f"PDF Document:  {PDF_OUTPUT}")
        print("=" * 70)
    except Exception as e:
        print(f"ERROR DURING REPORT ASSEMBLY: {e}")
        try:
            builder.doc.Close(False)
            builder.word.Quit()
        except Exception:
            pass
        raise e


if __name__ == "__main__":
    main()
