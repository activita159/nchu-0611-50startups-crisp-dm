"""
Generate CRISP-DM + Expert Discussion workflow diagram as PNG.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

font_manager.fontManager.addfont("C:\\Windows\\Fonts\\msyh.ttc")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle="arc3,rad=0"))


def draw_box(ax, x, y, w, h, text, facecolor, edgecolor="#333333", fontsize=11,
             fontcolor="white", bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08", facecolor=facecolor,
                          edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=fontcolor, weight=weight)


def draw_sub_box(ax, x, y, w, h, text, facecolor, edgecolor="#888888",
                 fontsize=9, fontcolor="#333333"):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05", facecolor=facecolor,
                          edgecolor=edgecolor, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=fontcolor)


def draw_phase_bg(ax, x, y, w, h, label, color):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1", facecolor=color,
                          edgecolor="none", alpha=0.12, zorder=0)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h + 0.15, label, ha="center", va="bottom",
            fontsize=13, color="#666666", weight="bold", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # Title
    ax.text(11, 13.3, "50 Startups CRISP-DM + Expert Discussion Workflow",
            ha="center", va="center", fontsize=18, weight="bold", color="#2C3E50")
    ax.text(11, 12.7, "Kaggle Dataset · 5 Domain Experts · 6 CRISP-DM Phases",
            ha="center", va="center", fontsize=11, color="#7F8C8D")

    # ═══════════════════════════════════════════════════
    # CRISP-DM Phases (background bands, left to right)
    # ═══════════════════════════════════════════════════
    phase_colors = ["#E8F8F5", "#FDEBD0", "#EBF5FB", "#F5EEF8", "#FDEDEC", "#E8F6F3"]
    phase_labels = [
        "Phase 1\nBusiness\nUnderstanding",
        "Phase 2\nData\nUnderstanding",
        "Phase 3\nData\nPreparation",
        "Phase 4\nModeling",
        "Phase 5\nEvaluation",
        "Phase 6\nDeployment",
    ]
    phase_x = [0.5, 3.7, 6.9, 10.1, 13.3, 16.5]
    phase_w = 3.2

    for i in range(6):
        draw_phase_bg(ax, phase_x[i], 0.3, phase_w, 12.0, phase_labels[i], phase_colors[i])

    # ═══════════════════════════════════════════════════
    # Row 1: Data Pipeline (y ≈ 10.5)
    # ═══════════════════════════════════════════════════
    row1_y = 10.5
    ax.text(0.3, row1_y + 0.7, "Data\nPipeline", ha="center", va="center",
            fontsize=10, weight="bold", color="#555555", rotation=0)

    box_h = 1.1
    arrow_color = "#555555"

    # Step 1: Load CSV
    draw_box(ax, 2.1, row1_y, 2.4, box_h, "Load 50_Startups.csv\n(50 records, 5 cols)",
             "#3498DB", fontsize=10)
    # Step 2: EDA
    draw_box(ax, 5.3, row1_y, 2.4, box_h, "EDA\nCorrelation · Distribution\nBoxplot · Scatter",
             "#2ECC71", fontsize=10)
    # Step 3: Feature Encoding
    draw_box(ax, 8.5, row1_y, 2.4, box_h, "Feature Encoding\nOne-Hot (State)\nStandardScaler",
             "#E67E22", fontsize=10)
    # Step 4: ML Models
    draw_box(ax, 11.7, row1_y, 2.4, box_h, "ML Modeling\nLR · RandomForest · XGBoost\n5-Fold CV",
             "#9B59B6", fontsize=10)
    # Step 5: Evaluation
    draw_box(ax, 14.9, row1_y, 2.4, box_h, "Evaluation\nR² / RMSE / MAE\nFeature Importance",
             "#E74C3C", fontsize=10)
    # Step 6: Report
    draw_box(ax, 18.1, row1_y, 2.4, box_h, "Report Generation\nCRISP-DM MD\nFigures · Models",
             "#1ABC9C", fontsize=10)

    for (x1, x2) in [(3.3, 4.1), (6.5, 7.3), (9.7, 10.5), (12.9, 13.7), (16.1, 16.9)]:
        draw_arrow(ax, x1, row1_y, x2, row1_y, arrow_color)

    # ═══════════════════════════════════════════════════
    # Row 2: Expert Discussion (y ≈ 7.0)
    # ═══════════════════════════════════════════════════
    row2_y = 7.0
    ax.text(0.3, row2_y + 0.5, "Expert\nDiscussion", ha="center", va="center",
            fontsize=10, weight="bold", color="#555555", rotation=0)

    # Down arrow from ML Models to Expert Discussion (broad)
    draw_arrow(ax, 11.7, row1_y - box_h/2 - 0.2, 11.7, row2_y + box_h + 1.2,
               "#8E44AD", lw=2.5)

    # Expert boxes row
    expert_colors = {
        "RD": "#C0392B",
        "Marketing": "#2980B9",
        "Administration": "#27AE60",
        "Governor": "#8E44AD",
        "ML_Expert": "#D35400",
    }
    experts = [
        ("RD", 4.0, "R&D Director\n研發總監"),
        ("Marketing", 7.0, "Marketing Dir.\n行銷總監"),
        ("Administration", 10.0, "Admin Director\n行政總監"),
        ("Governor", 13.0, "State Governor\n州長"),
        ("ML_Expert", 16.0, "ML Specialist\n機器學習專家"),
    ]
    for eid, ex, elabel in experts:
        draw_box(ax, ex, row2_y + 1.0, 2.6, 1.0, elabel, expert_colors[eid],
                 fontsize=9)

    # 5 Discussion Rounds
    draw_box(ax, 11.7, row2_y, 12.0, 1.0,
             "5-Round Expert Discussion: Initial Claims → EDA Findings → ML Results → Challenges → Consensus",
             "#F39C12", fontsize=9, fontcolor="white", bold=True)

    # Down arrow from discussion rounds
    draw_arrow(ax, 17.7, row2_y - 0.5, 17.7, row2_y - 2.0, "#F39C12", lw=2.0)

    # ═══════════════════════════════════════════════════
    # Row 3: Outputs (y ≈ 3.5)
    # ═══════════════════════════════════════════════════
    row3_y = 3.5
    ax.text(0.3, row3_y + 0.3, "Outputs", ha="center", va="center",
            fontsize=10, weight="bold", color="#555555")

    draw_box(ax, 4.0, row3_y, 3.0, 1.0, "10 EDA Figures\n(.png)", "#95A5A6",
             fontsize=9)
    draw_box(ax, 8.0, row3_y, 3.0, 1.0, "3 Trained Models\n(.pkl)", "#95A5A6",
             fontsize=9)
    draw_box(ax, 12.0, row3_y, 3.0, 1.0, "Discussion Transcript\n(.txt)", "#95A5A6",
             fontsize=9)
    draw_box(ax, 16.0, row3_y, 4.0, 1.0, "CRISP-DM Report\n(.md, 220+ lines)", "#95A5A6",
             fontsize=9)

    # ═══════════════════════════════════════════════════
    # Legend (bottom-right)
    # ═══════════════════════════════════════════════════
    legend_y = 1.2
    legend_items = [
        ("Data Pipeline", "#3498DB"),
        ("ML Modeling", "#9B59B6"),
        ("Expert Discussion", "#F39C12"),
        ("Outputs", "#95A5A6"),
    ]
    for i, (label, color) in enumerate(legend_items):
        lx = 2.0 + i * 4.0
        rect = FancyBboxPatch((lx - 1.0, legend_y - 0.25), 2.0, 0.5,
                              boxstyle="round,pad=0.05", facecolor=color,
                              edgecolor="#333333", linewidth=1.0, alpha=0.9)
        ax.add_patch(rect)
        ax.text(lx, legend_y, label, ha="center", va="center", fontsize=9,
                color="white", weight="bold")

    # ── Bottom annotation ──
    ax.text(11, 0.5, "Core Finding: R&D Spend is the dominant profit driver (R² = 0.9265 with 1 feature)  ·  Budget: R&D 60% / Marketing 25% / Admin 10% / Reserve 5%",
            ha="center", va="center", fontsize=10, color="#2C3E50",
            style="italic")

    fig.tight_layout(pad=1.0)
    filepath = os.path.join(OUTPUT_DIR, "workflow.png")
    fig.savefig(filepath, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"[OK] Workflow diagram saved: {filepath}")
    return filepath


if __name__ == "__main__":
    main()
