"""
Generate the dissertation PowerPoint presentation.

Creates a 16-slide deck for:
  "AI-Assisted Proactive Microservice Deployment Impact Forecasting
   using Runtime Observability, Dependency Graphs, and Machine Learning"

Slides include placeholder boxes where dissertation figures should be inserted.

Usage:
  python scripts/build_ppt.py
  python scripts/build_ppt.py --out results/presentation.pptx
"""
import argparse
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE       = RGBColor(0x1F, 0x49, 0x7D)   # dark navy
BLUE_LIGHT = RGBColor(0x2E, 0x75, 0xB6)   # Microsoft-style accent blue
ORANGE     = RGBColor(0xC0, 0x55, 0x22)   # accent orange
GREEN      = RGBColor(0x38, 0x86, 0x38)   # accent green
GREY_BG    = RGBColor(0xF2, 0xF2, 0xF2)   # slide background grey
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
BLACK      = RGBColor(0x00, 0x00, 0x00)
PLACEHOLDER_BG = RGBColor(0xD9, 0xE2, 0xF3)   # light blue for chart boxes

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "presentation.pptx"


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _rgb(r, g, b):
    return RGBColor(r, g, b)


def _add_textbox(slide, text, left, top, width, height,
                 font_size=18, bold=False, italic=False,
                 color=BLACK, align=PP_ALIGN.LEFT, wrap=True,
                 bg_color=None):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    if bg_color:
        fill = txBox.fill
        fill.solid()
        fill.fore_color.rgb = bg_color
    return txBox


def _add_rect(slide, left, top, width, height, fill_color, line_color=None, line_width=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        if line_width:
            shape.line.width = Pt(line_width)
    else:
        shape.line.fill.background()
    return shape


def _header_bar(slide, title_text, subtitle_text=None):
    """Dark blue top banner with white title."""
    _add_rect(slide, 0, 0, 13.33, 1.3, BLUE)
    _add_textbox(slide, title_text,
                 left=0.3, top=0.1, width=12.5, height=0.8,
                 font_size=24, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle_text:
        _add_textbox(slide, subtitle_text,
                     left=0.3, top=0.8, width=12.5, height=0.45,
                     font_size=13, bold=False, italic=True, color=WHITE,
                     align=PP_ALIGN.LEFT)


def _placeholder_box(slide, label, left, top, width, height):
    """Light blue box acting as a chart/figure placeholder."""
    _add_rect(slide, left, top, width, height,
              fill_color=PLACEHOLDER_BG, line_color=BLUE_LIGHT, line_width=1.5)
    _add_textbox(slide, f"[ INSERT FIGURE: {label} ]",
                 left=left + 0.1, top=top + height / 2 - 0.25,
                 width=width - 0.2, height=0.5,
                 font_size=13, italic=True, color=BLUE_LIGHT,
                 align=PP_ALIGN.CENTER)


def _bullet_list(slide, items, left, top, width, height,
                 font_size=16, indent_items=None, color=BLACK):
    """
    Render a simple bullet list.  indent_items is a list of booleans
    (True = sub-bullet, indented further).
    """
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for i, item in enumerate(items):
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        is_sub = indent_items[i] if indent_items else False
        p.level = 1 if is_sub else 0
        run = p.add_run()
        bullet = "    –  " if is_sub else "•  "
        run.text = bullet + item
        run.font.size = Pt(font_size - 2 if is_sub else font_size)
        run.font.color.rgb = color


def _table_slide(slide, headers, rows, left, top, width, col_widths=None):
    """Add a simple formatted table."""
    n_cols = len(headers)
    n_rows = len(rows) + 1
    row_h  = 0.38

    tbl = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(left), Inches(top),
        Inches(width), Inches(row_h * n_rows)
    ).table

    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)

    # Header row
    for ci, hdr in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.text = hdr
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.runs[0]
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = WHITE

    # Data rows
    for ri, row in enumerate(rows):
        bg = GREY_BG if ri % 2 == 0 else WHITE
        for ci, val in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.runs[0]
            run.font.size = Pt(11)
            run.font.color.rgb = BLACK


# ── Slide builders ─────────────────────────────────────────────────────────────

def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # Full background
    _add_rect(slide, 0, 0, 13.33, 7.5, BLUE)

    # Accent strip
    _add_rect(slide, 0, 5.5, 13.33, 0.15, ORANGE)

    # Main title
    _add_textbox(
        slide,
        "AI-Assisted Proactive Microservice\nDeployment Impact Forecasting",
        left=0.5, top=0.8, width=12.0, height=2.2,
        font_size=32, bold=True, color=WHITE, align=PP_ALIGN.CENTER
    )

    # Subtitle
    _add_textbox(
        slide,
        "using Runtime Observability, Dependency Graphs, and Machine Learning",
        left=0.5, top=2.9, width=12.0, height=0.9,
        font_size=18, italic=True, color=WHITE, align=PP_ALIGN.CENTER
    )

    # Divider line
    _add_rect(slide, 2.5, 3.9, 8.33, 0.04, WHITE)

    # Student info
    _add_textbox(slide, "MSc Dissertation",
                 left=0.5, top=4.1, width=12.0, height=0.5,
                 font_size=15, color=WHITE, align=PP_ALIGN.CENTER)
    _add_textbox(slide, "[Your Name]  |  [University]  |  2025",
                 left=0.5, top=4.55, width=12.0, height=0.5,
                 font_size=14, italic=True, color=WHITE, align=PP_ALIGN.CENTER)

    # Stack labels at bottom
    _add_textbox(slide,
                 "GKE  ·  Online Boutique  ·  Helm  ·  Jaeger  ·  Prometheus  ·  "
                 "RandomForest  ·  XGBoost  ·  LightGBM  ·  Optuna",
                 left=0.5, top=5.9, width=12.0, height=0.6,
                 font_size=11, italic=True, color=WHITE, align=PP_ALIGN.CENTER)


def slide_agenda(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Agenda")

    items = [
        "Problem Statement & Motivation",
        "Research Objectives",
        "System Architecture & Experimental Setup",
        "Dataset Overview",
        "Three-Stage ML Pipeline",
        "Stage 1 – Baseline Models",
        "Stage 2 – Feature Engineering",
        "Stage 3 – Hyperparameter Tuning (Optuna)",
        "Results: Severity Classification",
        "Results: Blast Radius Regression",
        "Proactive Window Analysis",
        "Key Findings & Contributions",
        "Conclusion & Current Results",
        "Future Phase 4 – RAG-Augmented Incident Intelligence",
        "Future Phase 5 – Autonomous Remediation Agent",
    ]
    _bullet_list(slide, items,
                 left=0.5, top=1.4, width=12.0, height=5.8,
                 font_size=15)


def slide_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Problem Statement", "Why proactive impact forecasting?")

    _add_textbox(slide, "The Challenge",
                 left=0.5, top=1.4, width=6.0, height=0.4,
                 font_size=15, bold=True, color=BLUE)
    _bullet_list(slide, [
        "Modern applications rely on dozens of tightly-coupled microservices",
        "A single misconfigured deployment (CPU limit, memory cap, replica count, latency) "
        "can cascade into system-wide outages",
        "Existing approaches are REACTIVE — they alert AFTER impact has propagated",
        "Operators have no early warning of downstream blast radius",
    ], left=0.5, top=1.8, width=6.0, height=2.8, font_size=14)

    _add_textbox(slide, "The Opportunity",
                 left=0.5, top=4.65, width=6.0, height=0.4,
                 font_size=15, bold=True, color=GREEN)
    _bullet_list(slide, [
        "Runtime observability (traces, metrics) generates rich early signals",
        "ML models can learn fault propagation patterns before they fully manifest",
        "Proactive forecasting enables automated rollback / traffic rerouting",
    ], left=0.5, top=5.05, width=6.0, height=1.9, font_size=14)

    # Right: reactive vs proactive diagram placeholder
    _placeholder_box(slide, "Reactive vs Proactive Timeline Diagram",
                     left=6.8, top=1.4, width=6.0, height=5.6)


def slide_objectives(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Research Objectives")

    objectives = [
        ("RO1", "Design a proactive formulation that predicts deployment impact "
                "using only early fault-window signals (T = 15–60s into fault)"),
        ("RO2", "Build a reproducible experiment harness on GKE / Online Boutique "
                "covering four fault types: cpu, memory, latency, replicas"),
        ("RO3", "Develop a three-stage ML pipeline — baseline → feature engineering "
                "→ hyperparameter tuning — with leakage-free GroupKFold evaluation"),
        ("RO4", "Forecast severity class (Low / Medium / High) with high accuracy "
                "and explainability via feature importance"),
        ("RO5", "Forecast blast radius (number of downstream services affected) "
                "as a regression task alongside the classifier"),
    ]

    for i, (label, text) in enumerate(objectives):
        y = 1.45 + i * 1.06
        _add_rect(slide, 0.4, y, 1.0, 0.65, BLUE)
        _add_textbox(slide, label,
                     left=0.4, top=y, width=1.0, height=0.65,
                     font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _add_rect(slide, 1.5, y, 11.3, 0.65, GREY_BG)
        _add_textbox(slide, text,
                     left=1.6, top=y + 0.05, width=11.0, height=0.6,
                     font_size=13, color=BLACK)


def slide_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "System Architecture", "GKE Standard cluster with Online Boutique")

    # Left column: stack description
    _add_textbox(slide, "Infrastructure",
                 left=0.4, top=1.4, width=4.5, height=0.4,
                 font_size=14, bold=True, color=BLUE)
    _bullet_list(slide, [
        "GKE Standard cluster (n1-standard-4 nodes)",
        "Online Boutique — 12 polyglot microservices",
        "Helm release: online-boutique (namespace: online-boutique)",
        "Fault injection via kubectl patch (Helm values override)",
    ], left=0.4, top=1.8, width=4.5, height=2.2, font_size=13)

    _add_textbox(slide, "Observability Stack",
                 left=0.4, top=4.1, width=4.5, height=0.4,
                 font_size=14, bold=True, color=ORANGE)
    _bullet_list(slide, [
        "Prometheus — CPU, memory, replica, request rate metrics",
        "Jaeger — distributed traces for all 12 services",
        "Custom collector — scrapes every 15 s during fault phase",
        "GCS bucket (ff-dataset-bucket) — raw experiment storage",
    ], left=0.4, top=4.5, width=4.5, height=2.5, font_size=13)

    _placeholder_box(slide, "System Architecture Diagram\n(GKE → Services → Observability → ML Pipeline)",
                     left=5.1, top=1.4, width=7.8, height=5.6)


def slide_dataset(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Dataset Overview", "Controlled fault injection experiments on GKE")

    _bullet_list(slide, [
        "89 fault-injection experiments across 10 target microservices",
        "4 fault types: cpu throttle, memory limit, latency injection, replica scale-down",
        "3 traffic levels: low, medium, high (Locust load generator)",
        "4 snapshot times per experiment: T = 15, 30, 45, 60 s into fault",
        "356 total rows (89 experiments × 4 time snapshots)",
        "Labels: severity class (Low / Medium / High) + blast_radius_final (regression)",
    ], left=0.4, top=1.4, width=6.0, height=3.2, font_size=14)

    _add_textbox(slide, "Label Distribution",
                 left=0.4, top=4.7, width=6.0, height=0.35,
                 font_size=13, bold=True, color=BLUE)
    _table_slide(slide,
                 headers=["Class", "Count", "Fraction"],
                 rows=[
                     ["Low",    "102", "28.7%"],
                     ["Medium", " 96", "27.0%"],
                     ["High",   "158", "44.4%"],
                 ],
                 left=0.4, top=5.05, width=5.5,
                 col_widths=[2.0, 1.6, 1.9])

    _placeholder_box(slide, "Fault Type Distribution / Experiment Grid",
                     left=6.5, top=1.4, width=6.4, height=5.6)


def slide_pipeline_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Three-Stage ML Pipeline")

    stages = [
        ("Stage 1", "Baseline", "24 base features\nRF · XGBoost · LightGBM\nGroupKFold(5) by experiment",
         BLUE),
        ("Stage 2", "Feature Engineering", "+ 9 engineered features\nTopology · Log transforms\nFault severity ratio · Traces",
         ORANGE),
        ("Stage 3", "Hyperparameter\nTuning", "Optuna Bayesian search\n50 trials per model\nMacro-F1 / MAE objective",
         GREEN),
    ]

    for i, (label, title, body, color) in enumerate(stages):
        x = 0.5 + i * 4.25
        _add_rect(slide, x, 1.5, 3.8, 0.55, color)
        _add_textbox(slide, f"{label}: {title}",
                     left=x, top=1.5, width=3.8, height=0.55,
                     font_size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _add_rect(slide, x, 2.05, 3.8, 2.2, GREY_BG)
        _add_textbox(slide, body,
                     left=x + 0.15, top=2.15, width=3.5, height=2.0,
                     font_size=13, color=BLACK, align=PP_ALIGN.CENTER)
        if i < 2:
            _add_textbox(slide, "→",
                         left=x + 3.85, top=2.4, width=0.4, height=0.5,
                         font_size=20, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    _add_textbox(slide, "Evaluation (all stages)",
                 left=0.4, top=4.55, width=12.5, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _bullet_list(slide, [
        "GroupKFold(n_splits=5) grouped by experiment_id — all 4 T-snapshots of one experiment "
        "land in the same fold, preventing time-leakage",
        "OOF (out-of-fold) predictions assembled by original row index for unbiased accuracy",
        "Metrics: OOF accuracy, macro-F1, per-class F1 (Low / Medium / High), MAE, RMSE",
    ], left=0.4, top=4.9, width=12.5, height=2.2, font_size=13)


def slide_baseline(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Stage 1 — Baseline Models", "24 raw features, default hyperparameters")

    _add_textbox(slide, "Feature Set (24 columns from preprocess.py — frozen)",
                 left=0.4, top=1.4, width=12.5, height=0.38,
                 font_size=14, bold=True, color=BLUE)

    cols = [
        ["fault_value_numeric", "time_into_fault_s", "target_cpu", "target_cpu_delta",
         "target_memory_bytes", "target_memory_delta"],
        ["target_replicas", "target_restart_count", "system_cpu_mean", "system_memory_mean",
         "target_request_rate", "target_error_rate"],
        ["target_latency_mean_ms", "target_latency_delta", "blast_radius_early",
         "max_trace_depth", "fault_type", "service", "traffic_level"],
    ]
    for ci, col in enumerate(cols):
        _bullet_list(slide, col,
                     left=0.4 + ci * 4.25, top=1.8, width=4.1, height=2.0,
                     font_size=11)

    _add_textbox(slide, "Baseline Results (OOF, GroupKFold-5)",
                 left=0.4, top=3.9, width=12.5, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _table_slide(slide,
                 headers=["Model", "Accuracy", "Macro-F1", "Low F1", "Medium F1", "High F1", "Blast MAE"],
                 rows=[
                     ["RandomForest", "69.7%", "0.610", "0.758", "0.266", "0.807", "1.639"],
                     ["XGBoost",      "72.2%", "0.634", "0.757", "0.329", "0.814", "1.717"],
                     ["LightGBM",     "75.8%", "0.684", "0.781", "0.414", "0.857", "1.642"],
                 ],
                 left=0.4, top=4.3, width=12.5,
                 col_widths=[2.0, 1.6, 1.6, 1.5, 1.7, 1.5, 1.6])

    _add_textbox(slide,
                 "Key observation: Medium-class F1 is consistently the lowest — boundary label "
                 "between Low and High; inherent label ambiguity.",
                 left=0.4, top=6.5, width=12.5, height=0.6,
                 font_size=12, italic=True, color=ORANGE)


def slide_feature_engineering(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Stage 2 — Feature Engineering", "+ 9 domain-informed features")

    groups = [
        ("1. Service Topology", BLUE,
         ["service_depth — call-graph depth (1 = entry, 3 = leaf)",
          "service_downstream_count — services this node calls",
          "service_is_gateway — binary: is frontend?"],
         "Encodes Online Boutique call-graph structure; a fault at depth-1 propagates further"),
        ("2. Fault Severity Ratio", ORANGE,
         ["fault_severity_ratio — fault magnitude / original Helm resource limit",
          "cpu: 1 − fval/limit_m  |  memory: 1 − fval/limit_Mi",
          "latency: min(ms/1000, 1)  |  replicas: 1 − replica_count"],
         "Normalises heterogeneous fault types to [0, 1] scale"),
        ("3. Log Transforms", GREEN,
         ["log_target_latency_mean_ms = log1p(latency)",
          "log_target_memory_bytes    = log1p(memory)",
          "log_fault_value_numeric    = log1p(fault_value)"],
         "Reduces right-skew; NaN preserved (not filled with 0)"),
        ("4. System Trace Aggregates", _rgb(0x70, 0x30, 0xA0),
         ["system_error_rate   — mean span error rate across all services",
          "system_latency_ratio — fault-window latency / baseline latency"],
         "97.8% row coverage from Jaeger raw traces"),
    ]

    for i, (title, color, bullets, note) in enumerate(groups):
        col = i % 2
        row = i // 2
        x = 0.4 + col * 6.5
        y = 1.45 + row * 2.85

        _add_rect(slide, x, y, 6.1, 0.38, color)
        _add_textbox(slide, title,
                     left=x + 0.1, top=y, width=5.9, height=0.38,
                     font_size=13, bold=True, color=WHITE)
        _bullet_list(slide, bullets,
                     left=x + 0.1, top=y + 0.4, width=5.9, height=1.3,
                     font_size=11)
        _add_textbox(slide, f"↳ {note}",
                     left=x + 0.1, top=y + 1.7, width=5.9, height=0.45,
                     font_size=10, italic=True, color=color)


def slide_fe_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Stage 2 — Feature Engineering Results",
                "Comparison: baseline vs FE dataset")

    _table_slide(slide,
                 headers=["Stage", "Model", "Accuracy", "Macro-F1", "Low F1",
                           "Medium F1", "High F1", "Blast MAE"],
                 rows=[
                     ["Baseline", "RandomForest", "69.7%", "0.610", "0.758", "0.266", "0.807", "1.639"],
                     ["Baseline", "LightGBM",     "75.8%", "0.684", "0.781", "0.414", "0.857", "1.642"],
                     ["FE",       "RandomForest", "75.3%", "0.690", "0.755", "0.432", "0.885", "1.549"],
                     ["FE",       "XGBoost",      "73.6%", "0.661", "0.737", "0.403", "0.842", "1.662"],
                     ["FE",       "LightGBM",     "74.4%", "0.669", "0.758", "0.392", "0.857", "1.613"],
                 ],
                 left=0.4, top=1.45, width=12.5,
                 col_widths=[1.6, 1.9, 1.5, 1.5, 1.3, 1.5, 1.3, 1.5])

    _add_textbox(slide, "Observations",
                 left=0.4, top=4.2, width=12.5, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _bullet_list(slide, [
        "RandomForest gains +8.0 pp accuracy and +0.080 macro-F1 over its own baseline",
        "Medium-class F1 improves most: +0.166 for RF (+0.432 vs +0.266) — topology features help disambiguate",
        "Blast MAE drops: RF 1.639 → 1.549 (−5.5%), showing regression also benefits",
        "LightGBM FE slightly lower than its baseline due to smaller search space interactions "
        "(resolved by tuning in Stage 3)",
    ], left=0.4, top=4.6, width=12.5, height=2.5, font_size=13)

    _placeholder_box(slide, "06_full_model_comparison.png — Accuracy + Macro-F1 grouped bars",
                     left=0.4, top=6.2, width=12.5, height=1.1)


def slide_tuning(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Stage 3 — Hyperparameter Tuning",
                "Optuna Bayesian optimisation, 50 trials per model")

    _add_textbox(slide, "Approach",
                 left=0.4, top=1.4, width=5.8, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _bullet_list(slide, [
        "Framework: Optuna (Tree-structured Parzen Estimator sampler)",
        "Inner loop: same GroupKFold-5 as training — no leakage",
        "Classification objective: maximise macro-F1",
        "Regression objective: minimise MAE",
        "Models tuned: RandomForest and LightGBM",
    ], left=0.4, top=1.8, width=5.8, height=2.4, font_size=13)

    _add_textbox(slide, "Search Spaces",
                 left=6.4, top=1.4, width=6.5, height=0.38,
                 font_size=14, bold=True, color=ORANGE)
    _bullet_list(slide,
                 [
                     "RF: n_estimators [200–1000], max_depth [5–30],",
                     "min_samples_leaf [1–10], max_features {sqrt, log2, 0.3, 0.5}",
                     "LGBM: n_estimators [100–1000], num_leaves [20–150],",
                     "learning_rate [0.01–0.2 log], subsample [0.5–1.0],",
                     "colsample_bytree [0.5–1.0], min_child_samples [5–50]",
                 ],
                 left=6.4, top=1.8, width=6.5, height=2.4,
                 font_size=12)

    _add_textbox(slide, "Tuning Improvements (50 trials)",
                 left=0.4, top=4.35, width=12.5, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _table_slide(slide,
                 headers=["Model", "Task", "Metric", "Before Tuning", "After Tuning", "Improvement"],
                 rows=[
                     ["RandomForest", "Classification", "Macro-F1", "0.690", "0.708", "+2.6%"],
                     ["RandomForest", "Regression",     "MAE",      "1.549", "1.495", "−3.5%"],
                     ["LightGBM",     "Classification", "Macro-F1", "0.684", "0.710", "+3.8%"],
                     ["LightGBM",     "Regression",     "MAE",      "1.642", "1.473", "−10.3%"],
                 ],
                 left=0.4, top=4.75, width=12.5,
                 col_widths=[2.1, 2.0, 1.5, 2.0, 2.0, 2.0])


def slide_classification_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Results — Severity Classification",
                "Three-stage OOF comparison (GroupKFold-5 by experiment_id)")

    _table_slide(slide,
                 headers=["Stage", "Model", "Accuracy", "Macro-F1", "Low F1", "Medium F1", "High F1"],
                 rows=[
                     ["Baseline",  "LightGBM",     "75.8%", "0.684", "0.781", "0.414", "0.857"],
                     ["FE",        "RandomForest",  "75.3%", "0.690", "0.755", "0.432", "0.885"],
                     ["Tuned FE",  "LightGBM",      "78.1%", "0.710", "0.792", "0.463", "0.876"],
                     ["Tuned FE",  "RandomForest",  "76.7%", "0.708", "0.769", "0.485", "0.871"],
                 ],
                 left=0.4, top=1.45, width=12.5,
                 col_widths=[1.9, 2.0, 1.6, 1.5, 1.4, 1.6, 1.5])

    _placeholder_box(slide, "01_stage_progression.png — Accuracy / Macro-F1 / Medium-F1 line chart",
                     left=0.4, top=3.5, width=6.1, height=3.7)
    _placeholder_box(slide, "02_per_class_f1_heatmap.png — F1 heatmap by stage × class",
                     left=6.7, top=3.5, width=6.1, height=3.7)


def slide_regression_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Results — Blast Radius Regression",
                "Predicting number of downstream services affected")

    _add_textbox(slide, "What is blast_radius_final?",
                 left=0.4, top=1.4, width=5.8, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _bullet_list(slide, [
        "Count of services degraded in the final 60 s of the fault window",
        "Computed from Prometheus error-rate and latency thresholds",
        "Range: 0 (no propagation) to 11 (all services degraded)",
        "Task: predict this value from early fault signals (T = 15–60 s)",
    ], left=0.4, top=1.8, width=5.8, height=2.2, font_size=13)

    _table_slide(slide,
                 headers=["Stage", "Model", "MAE", "RMSE"],
                 rows=[
                     ["Baseline", "RandomForest", "1.639", "2.016"],
                     ["Baseline", "LightGBM",     "1.642", "2.055"],
                     ["FE",       "RandomForest", "1.549", "1.946"],
                     ["Tuned FE", "LightGBM",     "1.473", "—"],
                     ["Tuned FE", "RandomForest", "1.495", "—"],
                 ],
                 left=0.4, top=4.15, width=5.8,
                 col_widths=[1.7, 1.8, 1.2, 1.1])

    _add_textbox(slide,
                 "Best result: LightGBM tuned MAE = 1.473\n"
                 "(−10.3% vs FE baseline; predictions within ~1.5 services of true value)",
                 left=0.4, top=6.3, width=5.8, height=0.8,
                 font_size=12, italic=True, color=GREEN)

    _placeholder_box(slide, "05_blast_radius_mae.png — MAE grouped bars by stage × model",
                     left=6.5, top=1.4, width=6.5, height=5.8)


def slide_proactive_window(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Proactive Window Analysis",
                "Accuracy at T = 15, 30, 45, 60 s into the fault phase")

    _add_textbox(slide, "Proactive Formulation",
                 left=0.4, top=1.4, width=5.8, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _bullet_list(slide, [
        "Features from early window: [fault_start → fault_start + T]",
        "Label from final 60 s: severity class + blast_radius_final",
        "Model predicts BEFORE impact fully propagates",
        "T = 15 s → forecast 45 s before full severity is known",
    ], left=0.4, top=1.8, width=5.8, height=2.2, font_size=13)

    _table_slide(slide,
                 headers=["Time (T)", "LightGBM Baseline", "RF Tuned", "LGBM Tuned"],
                 rows=[
                     ["T = 15 s", "~0.745", "—", "0.775"],
                     ["T = 30 s", "~0.758", "—", "0.787"],
                     ["T = 45 s", "~0.745", "—", "0.775"],
                     ["T = 60 s", "~0.758", "—", "0.787"],
                 ],
                 left=0.4, top=4.15, width=5.8,
                 col_widths=[1.5, 2.0, 1.4, 1.9])

    _add_textbox(slide,
                 "Key finding: accuracy is consistent across all T values — the model "
                 "generalises even at T = 15 s (very early in the fault lifecycle).",
                 left=0.4, top=6.3, width=5.8, height=0.8,
                 font_size=12, italic=True, color=GREEN)

    _placeholder_box(slide, "04_accuracy_vs_time.png — Proactive accuracy vs T (line chart)",
                     left=6.5, top=1.4, width=6.5, height=5.8)


def slide_feature_importance(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Feature Importance", "Top-15 features — RandomForest classifier (FE dataset)")

    _placeholder_box(slide, "03_feature_importance.png — Top-15 feature importance bar chart\n"
                     "(orange = engineered feature, blue = baseline feature)",
                     left=0.4, top=1.4, width=7.0, height=5.7)

    _add_textbox(slide, "Key Observations",
                 left=7.6, top=1.4, width=5.3, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _bullet_list(slide, [
        "blast_radius_early ranks #1 — early propagation is the strongest predictor",
        "fault_severity_ratio (engineered) enters top-5 — domain normalisation adds signal",
        "service_depth surfaces — faults closer to frontend propagate more",
        "log_target_latency_mean_ms outranks raw latency — log transform reduces noise",
        "system_error_rate captures cross-service cascade not visible in single-service metrics",
        "time_into_fault_s ranks high — later snapshots carry richer signal",
        "Categorical: fault_type and service are both informative",
    ], left=7.6, top=1.85, width=5.3, height=5.0, font_size=13)


def slide_findings(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Key Findings & Contributions")

    findings = [
        (BLUE,   "F1",
         "Proactive forecasting is feasible at T=15 s",
         "LightGBM tuned achieves 77.5% accuracy with only 15 seconds of fault data — "
         "demonstrating that early signals are sufficient for reliable prediction."),
        (ORANGE, "F2",
         "Three-stage pipeline delivers consistent gains",
         "Baseline → FE → Tuned: accuracy +2.3 pp, macro-F1 +0.026, blast MAE −10.3%. "
         "Each stage contributes incremental improvement."),
        (GREEN,  "F3",
         "Feature engineering improves the hardest class",
         "Medium-class F1 improves most (+0.071 for RF) — topology and severity ratio "
         "features reduce ambiguity at the Low/Medium boundary."),
        (_rgb(0x70, 0x30, 0xA0), "F4",
         "Dependency graph encodes propagation risk",
         "service_depth and service_downstream_count rank in the top-10 — structural "
         "position in the call graph predicts blast radius better than raw metrics alone."),
    ]

    for i, (color, label, title, body) in enumerate(findings):
        col = i % 2
        row = i // 2
        x = 0.4 + col * 6.5
        y = 1.45 + row * 2.7

        _add_rect(slide, x, y, 0.55, 0.9, color)
        _add_textbox(slide, label,
                     left=x, top=y + 0.1, width=0.55, height=0.7,
                     font_size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        _add_textbox(slide, title,
                     left=x + 0.65, top=y, width=5.6, height=0.5,
                     font_size=14, bold=True, color=color)
        _add_textbox(slide, body,
                     left=x + 0.65, top=y + 0.5, width=5.6, height=1.6,
                     font_size=12, color=BLACK)


def slide_conclusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Conclusion & Future Work")

    _add_textbox(slide, "Conclusion",
                 left=0.4, top=1.4, width=5.8, height=0.38,
                 font_size=14, bold=True, color=BLUE)
    _bullet_list(slide, [
        "Demonstrated that proactive ML-based impact forecasting is achievable on a real "
        "GKE microservice platform using only runtime observability",
        "Final best model: LightGBM (tuned) — 78.1% accuracy, 0.710 macro-F1, 1.473 MAE",
        "All four fault types (cpu, memory, latency, replicas) predicted with consistent accuracy",
        "GroupKFold evaluation prevents temporal leakage — results are generalisable",
        "Open-source pipeline: preprocess → feature engineer → train → tune → visualise",
    ], left=0.4, top=1.8, width=5.8, height=3.0, font_size=13)

    _add_textbox(slide, "Future Work",
                 left=0.4, top=5.0, width=5.8, height=0.38,
                 font_size=14, bold=True, color=ORANGE)
    _bullet_list(slide, [
        "Graph Neural Networks — encode call-graph as GNN input directly",
        "Online / streaming inference — real-time Kubernetes admission webhook",
        "Multi-fault scenarios — compound simultaneous faults",
        "Larger dataset — more services, more fault types, more traffic patterns",
        "SHAP explainability — per-prediction explanations for operators",
    ], left=0.4, top=5.4, width=5.8, height=1.8, font_size=13)

    _placeholder_box(slide, "06_full_model_comparison.png — Final model comparison",
                     left=6.5, top=1.4, width=6.5, height=5.8)


def slide_future_phases_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Future Phases — From Forecasting to Autonomous Remediation",
                "Extending the three-stage ML pipeline with AI reasoning and action")

    phases = [
        ("Phase 1–3\n(Current)", "Offline ML Pipeline",
         "preprocess → FE → tune\nGroupKFold evaluation\n78.1% accuracy achieved",
         BLUE, "DONE"),
        ("Phase 4", "RAG-Augmented\nIncident Intelligence",
         "Embed past experiments\nRetrieve similar incidents\nLLM-generated explanation",
         ORANGE, "NEXT"),
        ("Phase 5", "Autonomous\nRemediation Agent",
         "Live Prometheus/Jaeger feed\nAgent calls ML + RAG tools\nAuto rollback / alert",
         GREEN, "FUTURE"),
    ]

    for i, (phase, title, body, color, badge) in enumerate(phases):
        x = 0.5 + i * 4.2

        # Phase box
        _add_rect(slide, x, 1.45, 3.7, 0.55, color)
        _add_textbox(slide, phase,
                     left=x, top=1.45, width=3.7, height=0.55,
                     font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        # Title box
        _add_rect(slide, x, 2.0, 3.7, 0.7, GREY_BG)
        _add_textbox(slide, title,
                     left=x + 0.1, top=2.05, width=3.5, height=0.6,
                     font_size=13, bold=True, color=color, align=PP_ALIGN.CENTER)

        # Body box
        _add_rect(slide, x, 2.7, 3.7, 1.5, GREY_BG)
        _add_textbox(slide, body,
                     left=x + 0.1, top=2.75, width=3.5, height=1.4,
                     font_size=12, color=BLACK, align=PP_ALIGN.CENTER)

        # Badge
        badge_color = GREEN if badge == "DONE" else ORANGE if badge == "NEXT" else _rgb(0x70, 0x30, 0xA0)
        _add_rect(slide, x + 1.35, 4.3, 1.0, 0.32, badge_color)
        _add_textbox(slide, badge,
                     left=x + 1.35, top=4.3, width=1.0, height=0.32,
                     font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

        if i < 2:
            _add_textbox(slide, "→",
                         left=x + 3.75, top=2.7, width=0.45, height=0.5,
                         font_size=22, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    # Architecture flow diagram below
    _add_textbox(slide, "Combined Architecture (Phase 5)",
                 left=0.4, top=4.85, width=12.5, height=0.38,
                 font_size=14, bold=True, color=BLUE)

    flow = [
        ("[Prometheus\nJaeger]", BLUE),
        ("[Feature\nEngineer]", BLUE),
        ("[ML Model\nLightGBM]", BLUE),
        ("[RAG:\nIncidents]", ORANGE),
        ("[Agent\nOrchestrator]", GREEN),
        ("[Rollback /\nAlert]", _rgb(0x70, 0x30, 0xA0)),
    ]
    for i, (label, color) in enumerate(flow):
        x = 0.35 + i * 2.15
        _add_rect(slide, x, 5.3, 1.9, 0.75, color)
        _add_textbox(slide, label,
                     left=x, top=5.3, width=1.9, height=0.75,
                     font_size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        if i < len(flow) - 1:
            _add_textbox(slide, "→",
                         left=x + 1.9, top=5.5, width=0.25, height=0.35,
                         font_size=14, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    _add_textbox(slide,
                 "Phases 1–3 produce the trained model. Phase 4 adds memory (similar incident retrieval). "
                 "Phase 5 closes the loop: detect → predict → reason → act.",
                 left=0.4, top=6.2, width=12.5, height=0.7,
                 font_size=12, italic=True, color=BLUE)


def slide_future_rag_agent(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _header_bar(slide, "Phase 4 & 5 — RAG + Autonomous Agent",
                "Turning static forecasts into live, explainable, self-healing deployments")

    # ── Left column: RAG ──────────────────────────────────────────────────────
    _add_rect(slide, 0.35, 1.4, 5.95, 0.45, ORANGE)
    _add_textbox(slide, "Phase 4 — RAG: Incident Intelligence",
                 left=0.35, top=1.4, width=5.95, height=0.45,
                 font_size=14, bold=True, color=WHITE)

    _add_textbox(slide, "Knowledge Base",
                 left=0.35, top=1.95, width=5.95, height=0.32,
                 font_size=12, bold=True, color=ORANGE)
    _bullet_list(slide, [
        "89 experiments embedded as vectors (service, fault_type, features, outcome)",
        "Remediation runbooks (rollback steps per fault type)",
        "Online Boutique architecture docs (call graph, resource limits)",
        "results/final_results.csv — model performance history",
    ], left=0.35, top=2.27, width=5.95, height=1.4, font_size=12)

    _add_textbox(slide, "At inference time (new fault detected)",
                 left=0.35, top=3.75, width=5.95, height=0.32,
                 font_size=12, bold=True, color=ORANGE)
    _bullet_list(slide, [
        "Embed early feature vector → cosine similarity search",
        "Retrieve top-3 similar past experiments + outcomes",
        "LLM prompt: prediction + retrieved context → natural language explanation",
        'Example output: "Similar to EXP-042 (Nov 2024, checkoutservice cpu). '
        'Expect cascade to paymentservice in ~30 s."',
    ], left=0.35, top=4.07, width=5.95, height=2.3, font_size=12)

    # ── Right column: Agent ───────────────────────────────────────────────────
    _add_rect(slide, 6.7, 1.4, 6.2, 0.45, GREEN)
    _add_textbox(slide, "Phase 5 — Autonomous Remediation Agent",
                 left=6.7, top=1.4, width=6.2, height=0.45,
                 font_size=14, bold=True, color=WHITE)

    _add_textbox(slide, "Agent Tools",
                 left=6.7, top=1.95, width=6.2, height=0.32,
                 font_size=12, bold=True, color=GREEN)
    _bullet_list(slide, [
        "query_prometheus(service, window_s) → early metrics",
        "predict_severity(features) → calls clf_lgbm_tuned.joblib",
        "retrieve_similar_incidents(embedding) → RAG lookup",
        "rollback_helm(service, namespace) → kubectl / Helm",
        "alert_slack(message, severity) → operator notification",
    ], left=6.7, top=2.27, width=6.2, height=1.7, font_size=12)

    _add_textbox(slide, "Agent Reasoning Loop (every 15 s)",
                 left=6.7, top=4.05, width=6.2, height=0.32,
                 font_size=12, bold=True, color=GREEN)
    _bullet_list(slide, [
        "Thought: collect T=15s metrics for affected service",
        "Action: predict_severity() → High (confidence 0.87)",
        "Thought: blast_radius_early=4, service_depth=2 → high propagation risk",
        "Action: retrieve_similar_incidents() → 3/3 past cases cascaded",
        "Action: rollback_helm(checkoutservice) + alert_slack()",
        "Outcome: rollback triggered at T=18 s (before full propagation at T=60 s)",
    ], left=6.7, top=4.37, width=6.2, height=2.7, font_size=11,
        indent_items=[False, False, False, False, False, False])

    # ── Bottom comparison bar ─────────────────────────────────────────────────
    _add_rect(slide, 0.35, 6.85, 12.55, 0.42, BLUE)
    _add_textbox(slide,
                 "Impact: reactive MTTM ~90 s  →  Phase 5 proactive MTTM ~18 s  "
                 "(−80% mean time to mitigation)",
                 left=0.35, top=6.85, width=12.55, height=0.42,
                 font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


def slide_thank_you(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    _add_rect(slide, 0, 0, 13.33, 7.5, BLUE)
    _add_rect(slide, 0, 3.5, 13.33, 0.12, ORANGE)

    _add_textbox(slide, "Thank You",
                 left=0.5, top=1.2, width=12.0, height=1.2,
                 font_size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    _add_textbox(slide, "Questions?",
                 left=0.5, top=2.3, width=12.0, height=0.7,
                 font_size=24, italic=True, color=WHITE, align=PP_ALIGN.CENTER)

    _add_textbox(slide,
                 "Code & data: GCS bucket ff-dataset-bucket\n"
                 "Scripts: preprocess.py · feature_engineer.py · train_model.py · "
                 "tune_model.py · visualize.py\n"
                 "Results: results/final_results.csv · results/plots/",
                 left=1.0, top=3.9, width=11.0, height=1.5,
                 font_size=13, color=WHITE, align=PP_ALIGN.CENTER)

    _add_textbox(slide, "[Your Name]  |  [University]  |  2025",
                 left=0.5, top=6.5, width=12.0, height=0.5,
                 font_size=13, italic=True, color=WHITE, align=PP_ALIGN.CENTER)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(out_path: Path):
    prs = Presentation()
    prs.slide_width  = Inches(13.33)   # 16:9 widescreen
    prs.slide_height = Inches(7.5)

    slide_title(prs)
    slide_agenda(prs)
    slide_problem(prs)
    slide_objectives(prs)
    slide_architecture(prs)
    slide_dataset(prs)
    slide_pipeline_overview(prs)
    slide_baseline(prs)
    slide_feature_engineering(prs)
    slide_fe_results(prs)
    slide_tuning(prs)
    slide_classification_results(prs)
    slide_regression_results(prs)
    slide_proactive_window(prs)
    slide_feature_importance(prs)
    slide_findings(prs)
    slide_conclusion(prs)
    slide_future_phases_overview(prs)
    slide_future_rag_agent(prs)
    slide_thank_you(prs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"Saved → {out_path}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dissertation PowerPoint")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output .pptx path (default: results/presentation.pptx)")
    args = parser.parse_args()
    main(args.out)
