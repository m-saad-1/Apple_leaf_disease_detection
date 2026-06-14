"""
generate_methodology_figures.py
================================
Generates all 5 publication-ready methodology figures for:
  "Quantitative Comparison of Gradient-Based Visual Explanation Methods
   for Apple Leaf Disease Classification"

Figures produced
────────────────
  Fig1_System_Pipeline.png          – End-to-end framework overview
  Fig2_EfficientNetB0_Architecture.png – Model architecture detail
  Fig3_XAI_Methods_Technical.png    – Grad-CAM / Grad-CAM++ / Score-CAM internals
  Fig4_Annotation_Evaluation.png    – Pixel-mask annotation & IoU protocol
  Fig5_Robustness_Protocol.png      – Distortion types & dual-metric evaluation

Run:  python generate_methodology_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib.lines import Line2D
import numpy as np
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "methodology_figures"
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 220   # use 300 for final submission

# ── Global palette ────────────────────────────────────────────────────────────
C = {
    "bg":        "#FFFFFF",
    "deep":      "#1B3A6B",
    "mid":       "#2E6CA4",
    "light":     "#76B7E0",
    "pale":      "#D6E8F7",
    "very_pale": "#EBF4FB",
    "green":     "#2ECC71",
    "dark_green":"#1A8A4A",
    "orange":    "#E67E22",
    "red":       "#C0392B",
    "purple":    "#8E44AD",
    "gray":      "#7F8C8D",
    "light_gray":"#ECF0F1",
    "text":      "#1A1A2E",
    "white":     "#FFFFFF",
}

def save(fig, name, tight=True):
    path = os.path.join(OUT_DIR, name)
    if tight:
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    else:
        fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    import gc
    gc.collect()
    print(f"  Saved -> {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Shared drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel="", fc=C["pale"], ec=C["mid"],
        lw=1.5, fs=9, sfs=7.5, bold=False, radius=0.015, text_color=C["text"]):
    """Draw a rounded rectangle with a title and optional subtitle."""
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(b)
    lbl_lines = label.count('\n') + 1
    sub_lines = sublabel.count('\n') + 1 if sublabel else 0
    if sublabel:
        lbl_y = y + 0.008 * sub_lines + 0.005
        sub_y = y - 0.012 * lbl_lines - 0.002
    else:
        lbl_y = y
        sub_y = y
    ax.text(x, lbl_y, label, ha="center", va="center",
            fontsize=fs, color=text_color, zorder=4,
            fontweight="bold" if bold else "normal",
            wrap=False)
    if sublabel:
        ax.text(x, sub_y, sublabel, ha="center", va="center",
                fontsize=sfs, color=C["gray"], zorder=4, style="italic")

def header_box(ax, x, y, w, h, label, fc=C["deep"], ec=C["deep"],
               fs=10, text_color=C["white"], radius=0.015):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=2, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fs, color=text_color, fontweight="bold", zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C["mid"], lw=1.5,
          style="-|>", mutation=12, shrink=3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=mutation,
                                shrinkA=shrink, shrinkB=shrink),
                zorder=5)

def dim_label(ax, x, y, text, color=C["gray"], fs=7):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color, style="italic", zorder=6)

def section_label(ax, x, y, text, color=C["deep"]):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=8.5, color=color, fontweight="bold",
            bbox=dict(fc=C["very_pale"], ec=C["mid"], lw=1,
                      boxstyle="round,pad=0.25"), zorder=6)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1  –  End-to-End System Pipeline
# ═════════════════════════════════════════════════════════════════════════════
def fig1_pipeline():
    print("Generating Fig 1: System Pipeline...")
    fig, ax = plt.subplots(figsize=(20, 13))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # ── title ──────────────────────────────────────────────────────────────
    ax.text(0.5, 0.97,
            "End-to-End Explainable and Robust Plant Disease Detection Framework",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=C["deep"])
    ax.text(0.5, 0.938,
            "EfficientNetB0  ·  Grad-CAM  /  Grad-CAM++  /  Score-CAM  ·  "
            "Robustness Testing  ·  TFLite Deployment",
            ha="center", va="center", fontsize=9.5, color=C["gray"])

    # thin top rule
    ax.plot([0.04, 0.96], [0.925, 0.925], color=C["mid"], lw=1.2, alpha=0.5)

    # ── Stage boxes — horizontal spine ────────────────────────────────────
    spine_y   = 0.86
    spine_xs  = [0.19, 0.42, 0.65, 0.88]
    spine_w   = 0.16
    spine_h   = 0.08

    stages = [
        ("① Input\nLeaf Image",        "PlantVillage / Field",          C["very_pale"], C["mid"]),
        ("② Preprocessing\nPipeline",  "Resize · Normalise · Augment",  C["very_pale"], C["mid"]),
        ("③ EfficientNetB0\nBackbone",  "Transfer Learning (frozen)",    C["pale"],      C["deep"]),
        ("④ Classification\nOutput",   "Softmax · 4 Classes",           C["pale"],      C["deep"]),
    ]
    for (lbl, sub, fc, ec), x in zip(stages, spine_xs):
        box(ax, x, spine_y, spine_w, spine_h,
            lbl, sub, fc=fc, ec=ec, lw=2, fs=9, sfs=7.5, bold=True, radius=0.012)

    # arrows between spine boxes
    for i in range(len(spine_xs) - 1):
        arrow(ax, spine_xs[i] + spine_w/2 + 0.005, spine_y,
                  spine_xs[i+1] - spine_w/2 - 0.005, spine_y,
              color=C["deep"], lw=2, mutation=14)

    # ── Dataset info box ──────────────────────────────────────────────────
    db_x, db_y = 0.06, 0.86
    db_w, db_h = 0.09, 0.10
    db = FancyBboxPatch((db_x - db_w/2, db_y - db_h/2), db_w, db_h,
                        boxstyle="round,pad=0,rounding_size=0.008",
                        linewidth=1.2, edgecolor=C["mid"],
                        facecolor=C["very_pale"], zorder=3)
    ax.add_patch(db)
    ax.text(db_x, db_y + 0.035, "Dataset", ha="center", fontsize=8,
            fontweight="bold", color=C["deep"], zorder=4)
    for k, line in enumerate(["PlantVillage","4 Classes","Train 8,844",
                               "Val  2,704","Test 2,694"]):
        ax.text(db_x, db_y + 0.010 - k*0.014, line, ha="center",
                fontsize=7, color=C["text"], zorder=4)
    arrow(ax, db_x + db_w/2 + 0.005, db_y, spine_xs[0] - spine_w/2 - 0.005, spine_y,
          color=C["mid"], lw=1.5, mutation=10)

    # ── Preprocessing detail panel ─────────────────────────────────────────
    pre_x = spine_xs[1]
    pre_y = 0.65
    pre_items = [
        "Resize → 224×224 px",
        "EfficientNet preprocess_input()",
        "RandomFlip · RandomRotation (±15°)",
        "RandomZoom (±10%) · RandomTranslation",
        "Contrast · Brightness perturbation",
    ]
    pbox = FancyBboxPatch((pre_x - 0.10, pre_y - 0.08), 0.20, 0.16,
                          boxstyle="round,pad=0,rounding_size=0.008",
                          linewidth=1.2, edgecolor=C["light"], facecolor=C["very_pale"], zorder=3)
    ax.add_patch(pbox)
    ax.text(pre_x, pre_y + 0.06, "Preprocessing Detail", ha="center",
            fontsize=8, fontweight="bold", color=C["mid"], zorder=4)
    for k, itm in enumerate(pre_items):
        ax.text(pre_x - 0.09, pre_y + 0.035 - k*0.024, f"• {itm}", ha="left",
                fontsize=7.2, color=C["text"], zorder=4)
    # dashed guide to spine box
    ax.plot([pre_x, pre_x], [spine_y - spine_h/2, pre_y + 0.08],
            color=C["light"], lw=1, ls="--", alpha=0.7, zorder=2)

    # ── Model architecture detail panel ────────────────────────────────────
    mod_panel_x = spine_xs[2]
    mod_layers = [
        ("Input 224×224×3",       C["very_pale"]),
        ("Augmentation Block",    C["very_pale"]),
        ("EfficientNetB0 (frozen)\n4,049,571 params", C["pale"]),
        ("Global Avg. Pooling",   C["very_pale"]),
        ("Dropout (p = 0.3)",     C["very_pale"]),
        ("Dense 4 · Softmax\n5,124 trainable params", C["pale"]),
    ]
    bw, bh = 0.16, 0.035
    ly_start = 0.72
    for k, (lbl, fc) in enumerate(mod_layers):
        by = ly_start - k * 0.042
        mb = FancyBboxPatch((mod_panel_x - bw/2, by - bh/2), bw, bh,
                            boxstyle="round,pad=0,rounding_size=0.006",
                            linewidth=1, edgecolor=C["mid"], facecolor=fc, zorder=3)
        ax.add_patch(mb)
        ax.text(mod_panel_x, by, lbl, ha="center", va="center",
                fontsize=7, color=C["text"], zorder=4)
        if k < len(mod_layers) - 1:
            arrow(ax, mod_panel_x, by - bh/2 - 0.001,
                      mod_panel_x, ly_start - (k+1)*0.042 + bh/2 + 0.001,
                  color=C["mid"], lw=1, mutation=8, shrink=0)
    ax.plot([mod_panel_x, mod_panel_x], [spine_y - spine_h/2, ly_start - bh/2],
            color=C["light"], lw=1, ls="--", alpha=0.7, zorder=2)

    # ── Prediction Probabilities (moved from top right to flow directly) ──
    pred_y = 0.65
    box(ax, spine_xs[-1], pred_y, 0.14, 0.05, "Prediction\nProbabilities",
        fc=C["pale"], ec=C["mid"], lw=1.5, fs=8, sfs=8, bold=True)
    # arrow from box 4 down to prediction probabilities
    arrow(ax, spine_xs[-1], spine_y - spine_h/2 - 0.005,
          spine_xs[-1], pred_y + 0.025 + 0.005, color=C["deep"], lw=1.5, mutation=10)

    # ── Four lower branches ───────────────────────────────────────────────
    branch_y_top = 0.40

    branches = [
        # (x_center, header_label, items, color)
        (0.19, "Branch A\nXAI Explanation",
         ["Grad-CAM\n(gradient-weighted CAM)",
          "Grad-CAM++\n(second-order weights)",
          "Score-CAM\n(gradient-free scoring)"],
         C["deep"], C["pale"]),
        (0.42, "Branch B\nAnnotation & Evaluation",
         ["LabelMe pixel masks",
          "Lesion IoU computation",
          "Localization Accuracy"],
         C["dark_green"], "#D5F5E3"),
        (0.65, "Branch C\nRobustness Testing",
         ["Gaussian Noise · Blur",
          "Brightness Shift · Occlusion",
          "AUC over severity levels"],
         C["orange"], "#FDEBD0"),
        (0.88, "Branch D\nDeployment",
         ["Keras → TFLite FP32",
          "Dynamic Quantisation",
          "Latency · Size · Accuracy"],
         C["red"], "#FADBD8"),
    ]

    # branch fork line
    # From Prediction Probabilities down to fork line, then split
    fork_y = 0.48
    ax.plot([spine_xs[-1], spine_xs[-1]], [pred_y - 0.025, fork_y], color=C["deep"], lw=1.5, ls="-", zorder=2)
    ax.plot([branches[0][0], branches[-1][0]], [fork_y, fork_y], color=C["deep"], lw=1.2, ls="-", zorder=2, alpha=0.6)
    
    for bx, *_ in branches:
        # Arrow from fork line down to the TOP edge of the header box (which is branch_y_top + 0.040)
        arrow(ax, bx, fork_y, bx, branch_y_top + 0.040 + 0.005,
              color=C["deep"], lw=1.5, mutation=10, shrink=0)

    bw2, bh2 = 0.20, 0.035
    for bx, hdr, items, hdr_c, item_fc in branches:
        # header
        header_box(ax, bx, branch_y_top + 0.020, bw2, 0.040,
                   hdr, fc=hdr_c, fs=8)
        # item boxes
        for k, itm in enumerate(items):
            # added more spacing between header and items
            iy = branch_y_top - 0.025 - k * 0.055
            mb = FancyBboxPatch((bx - bw2/2, iy - bh2/2), bw2, bh2,
                                boxstyle="round,pad=0,rounding_size=0.006",
                                linewidth=1, edgecolor=hdr_c,
                                facecolor=item_fc, zorder=3)
            ax.add_patch(mb)
            ax.text(bx, iy, itm, ha="center", va="center",
                    fontsize=7.5, color=C["text"], zorder=4)
            if k < 2:
                arrow(ax, bx, iy - bh2/2 - 0.001,
                          bx, branch_y_top - 0.025 - (k+1)*0.055 + bh2/2 + 0.001,
                      color=hdr_c, lw=1, mutation=7, shrink=0)

    # ── Bottom output row ──────────────────────────────────────────────────
    out_y = 0.08
    out_items = [
        (0.19, "XAI\nComparison\nTable",       C["deep"],       C["pale"]),
        (0.42, "IoU · Coverage\nLoc. Accuracy\nReport",       C["dark_green"], "#D5F5E3"),
        (0.65, "Robustness\nAUC\nSummary",     C["orange"],     "#FDEBD0"),
        (0.88, "TFLite\nBenchmark\nReport",    C["red"],        "#FADBD8"),
    ]
    for bx, lbl, hdr_c, fc in out_items:
        # arrow from bottom of the last item box to top of output box
        last_iy = branch_y_top - 0.025 - 2*0.055
        arrow(ax, bx, last_iy - bh2/2 - 0.005,
                  bx, out_y + 0.035,
              color=hdr_c, lw=1.2, mutation=9, shrink=0)
        ob = FancyBboxPatch((bx - 0.09, out_y - 0.03), 0.18, 0.06,
                            boxstyle="round,pad=0,rounding_size=0.008",
                            linewidth=1.5, edgecolor=hdr_c, facecolor=fc, zorder=3)
        ax.add_patch(ob)
        ax.text(bx, out_y, lbl, ha="center", va="center",
                fontsize=8, color=C["text"], fontweight="bold", zorder=4)

    # ── Legend ─────────────────────────────────────────────────────────────
    leg_items = [
        (C["deep"],        "Core classification pipeline"),
        (C["dark_green"],  "XAI evaluation branch"),
        (C["orange"],      "Robustness testing branch"),
        (C["red"],         "Deployment branch"),
    ]
    for k, (col, lbl) in enumerate(leg_items):
        ax.plot(0.055 + k*0.22, 0.016, "s", color=col, ms=9, zorder=5)
        ax.text(0.068 + k*0.22, 0.016, lbl, va="center",
                fontsize=8, color=C["text"])

    plt.title("")
    save(fig, "Fig1_System_Pipeline.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2  –  EfficientNetB0 Architecture Detail
# ═════════════════════════════════════════════════════════════════════════════
def fig2_architecture():
    print("Generating Fig 2: EfficientNetB0 Architecture...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 10),
                              gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(C["bg"])
    for a in axes: a.set_facecolor(C["bg"]); a.axis("off")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.08, wspace=0.15)

    # ── Left panel: layer-by-layer stack ────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.97, "Model Architecture", ha="center",
            fontsize=17, fontweight="bold", color=C["deep"])

    layers_info = [
        # (label, sublabel, width_frac, fc, ec, height)
        ("Input Layer",        "224 × 224 × 3  (RGB)",              0.66, C["very_pale"], C["mid"],   0.085),
        ("Augmentation Block", "Flip · Rotate · Zoom · Contrast · Brightness", 0.66, C["very_pale"], C["light"], 0.085),
        ("EfficientNetB0",     "Frozen backbone  –  ImageNet weights", 0.76, C["pale"],      C["deep"],  0.290),
        ("Global Avg. Pool",   "7×7×1280  →  1280",                  0.66, C["very_pale"], C["mid"],   0.085),
        ("Dropout",            "rate = 0.30",                        0.66, C["very_pale"], C["mid"],   0.085),
        ("Dense(4) + Softmax", "4 units  |  5,124 trainable params", 0.66, C["pale"],      C["deep"],  0.085),
    ]

    y_cur = 0.88
    layer_centers = []
    for k, (lbl, sub, w, fc, ec, h) in enumerate(layers_info):
        bx = 0.5
        b = FancyBboxPatch((bx - w/2, y_cur - h), w, h,
                           boxstyle="round,pad=0,rounding_size=0.008",
                           linewidth=2, edgecolor=ec, facecolor=fc, zorder=3)
        ax.add_patch(b)
        
        if lbl == "EfficientNetB0":
            # MBConv detail inside big box - manual layout to prevent overlap
            ax.text(bx, y_cur - 0.035, lbl, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=C["text"], zorder=4)
            ax.text(bx, y_cur - 0.07, sub, ha="center", va="center",
                    fontsize=10.5, color=C["gray"], style="italic", zorder=4)
            
            ax.text(bx, y_cur - 0.12, "MBConv Stages (7 blocks):", ha="center",
                    fontsize=11, color=C["mid"], fontweight="bold", zorder=4)
            
            stages = [("Stage 1–2\n16–24 ch", bx - 0.21),
                      ("Stage 3–4\n40–80 ch", bx),
                      ("Stage 5–7\n112–1280 ch", bx + 0.21)]
            for stxt, sx in stages:
                sb = FancyBboxPatch((sx - 0.09, y_cur - 0.22), 0.18, 0.088,
                                    boxstyle="round,pad=0,rounding_size=0.006",
                                    lw=1.2, edgecolor=C["light"],
                                    facecolor=C["very_pale"], zorder=4)
                ax.add_patch(sb)
                ax.text(sx, y_cur - 0.176, stxt, ha="center", va="center",
                        fontsize=9.5, color=C["text"], fontweight="bold", zorder=5)
            ax.text(bx, y_cur - 0.26, "BatchNorm · SiLU activation · SE modules",
                    ha="center", fontsize=9.5, color=C["gray"], zorder=4)
        else:
            dy = 0.016 if sub else 0
            ax.text(bx, y_cur - h/2 + dy, lbl, ha="center", va="center",
                    fontsize=12.5, fontweight="bold", color=C["text"], zorder=4)
            if sub:
                ax.text(bx, y_cur - h/2 - 0.018, sub, ha="center", va="center",
                        fontsize=10.5, color=C["gray"], style="italic", zorder=4)
                        
        layer_centers.append(y_cur - h/2)
        y_cur -= h + 0.020

    # arrows between layers
    for i in range(len(layers_info)-1):
        y1 = layer_centers[i] - layers_info[i][-1]/2 - 0.005
        y2 = layer_centers[i+1] + layers_info[i+1][-1]/2 + 0.005
        arrow(ax, 0.5, y1, 0.5, y2, color=C["deep"], lw=2.2,
              mutation=13, shrink=0)

    # param counts on right side of boxes
    param_labels = {
        2: "4,049,571 params\n(frozen)",
        5: "5,124 params\n(trainable)"
    }
    for k, txt in param_labels.items():
        yp = layer_centers[k]
        box_right = 0.5 + layers_info[k][2]/2
        ax.plot([box_right, box_right + 0.025], [yp, yp], color=C["light"], lw=1.5, ls="--", zorder=2)
        ax.text(box_right + 0.03, yp, txt, ha="left", va="center",
                fontsize=10.5, color=C["mid"],
                bbox=dict(fc=C["very_pale"], ec=C["light"], lw=1.2,
                          boxstyle="round,pad=0.3"))

    # Total params summary box properly placed below the stack
    ax.text(0.5, y_cur - 0.02, "Total: 4,054,695 params  |  Trainable: 5,124 params",
            ha="center", fontsize=12, color=C["deep"],
            fontweight="bold",
            bbox=dict(fc=C["pale"], ec=C["mid"], lw=1.5,
                      boxstyle="round,pad=0.5"))

    # ── Right panel: MBConv block detail ──────────────────────────────────
    ax2 = axes[1]
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.text(0.5, 0.97, "MBConv Block Detail (EfficientNet building block)",
             ha="center", fontsize=17, fontweight="bold", color=C["deep"])
    ax2.text(0.5, 0.935, "Used in frozen backbone — repeated with increasing channels",
             ha="center", fontsize=11, color=C["gray"])

    mbconv_layers = [
        ("Input Feature Map",             "H × W × C_in",          C["very_pale"], C["mid"]),
        ("Pointwise Conv (Expand)\n1×1",  "C_in → C_in × expansion", C["pale"],    C["mid"]),
        ("Batch Normalisation + SiLU",    "",                        C["very_pale"], C["light"]),
        ("Depthwise Conv\n3×3 or 5×5",    "Spatial filtering",       C["pale"],    C["mid"]),
        ("Batch Normalisation + SiLU",    "",                        C["very_pale"], C["light"]),
        ("Squeeze & Excitation\n(SE)",    "Channel re-weighting",    "#FFF3CD",    C["orange"]),
        ("Pointwise Conv (Project)\n1×1", "C_expanded → C_out",      C["pale"],    C["mid"]),
        ("Batch Normalisation",           "",                        C["very_pale"], C["light"]),
        ("Skip Connection (if same shape)","+ residual add",          "#D5F5E3",   C["dark_green"]),
    ]

    bw3, bh3 = 0.58, 0.076
    y3 = 0.860
    for k, (lbl, sub, fc, ec) in enumerate(mbconv_layers):
        by = y3 - k * 0.085
        mb = FancyBboxPatch((0.5 - bw3/2, by - bh3/2), bw3, bh3,
                            boxstyle="round,pad=0,rounding_size=0.007",
                            linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3)
        ax2.add_patch(mb)
        dy2 = 0.012 if sub else 0
        ax2.text(0.5, by + dy2, lbl, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color=C["text"], zorder=4)
        if sub:
            ax2.text(0.5, by - 0.019, sub, ha="center", va="center",
                     fontsize=9.5, color=C["gray"], style="italic", zorder=4)
        if k < len(mbconv_layers) - 1:
            ay = by - bh3/2 - 0.002
            ay2 = by - 0.085 + bh3/2 + 0.002
            arrow(ax2, 0.5, ay, 0.5, ay2,
                  color=ec, lw=1.5, mutation=10, shrink=0)

    # skip connection clearly drawn as a square bracket
    skip_top = y3
    skip_bot = y3 - (len(mbconv_layers)-1)*0.085
    skip_x_start = 0.5 + bw3/2
    skip_x_out = skip_x_start + 0.08

    ax2.plot([skip_x_start, skip_x_out, skip_x_out, skip_x_start + 0.02],
             [skip_top, skip_top, skip_bot, skip_bot],
             color=C["dark_green"], lw=2, zorder=2)
    arrow(ax2, skip_x_start + 0.03, skip_bot, skip_x_start, skip_bot,
          color=C["dark_green"], lw=2, mutation=12, shrink=0)
    ax2.text(skip_x_out + 0.01, (skip_top + skip_bot)/2, "Skip\nConnection",
             ha="left", va="center", fontsize=11, color=C["dark_green"], fontweight="bold")

    # SiLU formula
    formula_y = y3 - (len(mbconv_layers)-1)*0.085 - 0.090
    ax2.text(0.5, formula_y,
             r"SiLU$(x) = x \cdot \sigma(x)$    "
             r"SE block: $\hat{x} = x \cdot \sigma(W_2 \cdot \delta(W_1 \cdot z))$",
             ha="center", fontsize=12, color=C["deep"],
             bbox=dict(fc=C["very_pale"], ec=C["mid"], lw=1.2,
                       boxstyle="round,pad=0.5"))

    fig.suptitle("EfficientNetB0 Model Architecture",
                 fontsize=18, fontweight="bold", y=0.005, color=C["deep"])
    save(fig, "Fig2_EfficientNetB0_Architecture.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3  –  XAI Methods Technical Diagram
# ═════════════════════════════════════════════════════════════════════════════
def fig3_xai_methods():
    print("Generating Fig 3: XAI Methods Technical Diagram...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 13))
    fig.patch.set_facecolor(C["bg"])
    for a in axes: a.set_facecolor(C["bg"]); a.axis("off")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.08, wspace=0.25)

    fig.suptitle("Technical Comparison of Gradient-Based XAI Methods",
                 fontsize=18, fontweight="bold", color=C["deep"], y=0.97)

    method_configs = [
        (axes[0], "Grad-CAM",
         "Gradient-Weighted Class Activation Mapping\n(Selvaraju et al., ICCV 2017)",
         C["deep"],
         [
             ("Input Image\n224 × 224 × 3",          C["very_pale"], C["mid"],   ""),
             ("Forward Pass\nthrough CNN",            C["very_pale"], C["mid"],   ""),
             ("Feature Maps  Aᵏ\n(final conv layer)", C["pale"],      C["deep"],  "k = 1…1280 maps"),
             ("Compute Gradient\n∂yᶜ / ∂Aᵏ",         C["pale"],      C["deep"],  "class score yᶜ"),
             ("Global Avg. Pool\nαᵏ = (1/Z) Σᵢⱼ ∂yᶜ/∂Aᵏᵢⱼ",
                                                       C["pale"],      C["deep"],  "scalar weight / map"),
             ("Weighted Combination\nL = ReLU(Σₖ αᵏ · Aᵏ)",
                                                       "#D6EAF8",      C["deep"],  "coarse heatmap"),
             ("Upsample & Normalise\nto 224 × 224",   C["very_pale"], C["light"], "bilinear interp."),
             ("Grad-CAM Heatmap\n[0, 1] overlay",     C["pale"],      C["deep"],  "class-discriminative"),
         ]),
        (axes[1], "Grad-CAM++",
         "Improved Gradient-Weighted CAM\n(Chattopadhay et al., WACV 2018)",
         C["purple"],
         [
             ("Input Image\n224 × 224 × 3",          C["very_pale"], C["purple"], ""),
             ("Forward Pass\nthrough CNN",            C["very_pale"], C["purple"], ""),
             ("Feature Maps  Aᵏ\n(final conv layer)", "#EDE7F6",      C["purple"], "same target layer"),
             ("1st-Order Gradients\n∂yᶜ / ∂Aᵏ",       "#EDE7F6",      C["purple"], ""),
             ("2nd & 3rd Gradients\n∂²yᶜ/∂(Aᵏ)²  ∂³yᶜ/∂(Aᵏ)³",
                                                       "#D7BDE2",      C["purple"], "higher-order signals"),
             ("Alpha Coefficients\nαᵏᵢⱼ = (∂²yᶜ/∂Aᵏ²) / (2·∂²yᶜ/∂Aᵏ² + Σ∂³yᶜ/∂Aᵏ³·Aᵏ)",
                                                       "#D7BDE2",      C["purple"], "pixel-wise weights"),
             ("Weighted Sum\nL = ReLU(Σₖ [Σᵢⱼ αᵏᵢⱼ·ReLU(∂yᶜ/∂Aᵏᵢⱼ)] · Aᵏ)",
                                                       "#D7BDE2",      C["purple"], "better small lesions"),
             ("Grad-CAM++ Heatmap\n[0, 1] overlay",   "#EDE7F6",      C["purple"], "fine-grained spatial"),
         ]),
        (axes[2], "Score-CAM",
         "Score-Based Class Activation Mapping\n(Wang et al., CVPR Workshops 2020)",
         C["dark_green"],
         [
             ("Input Image\n224 × 224 × 3",           C["very_pale"], C["dark_green"], ""),
             ("Forward Pass\nthrough CNN",             C["very_pale"], C["dark_green"], ""),
             ("Extract Activation Maps\nAᵏ (top-K channels)",
                                                        "#D5F5E3",      C["dark_green"], "K = 30 (speed)"),
             ("Normalise Each Map\nAᵏₙₒᵣₘ ∈ [0, 1]",  "#D5F5E3",      C["dark_green"], "per-channel norm"),
             ("Mask Input Image\nXₘₐₛₑ𝒹 = X ⊙ ↑Aᵏₙₒᵣₘ", "#D5F5E3",  C["dark_green"], "Hadamard product"),
             ("Score Channel k\nwₖ = Softmax(f(Xₘₐₛₑ𝒹))ᶜ",
                                                        "#A9DFBF",      C["dark_green"], "model confidence"),
             ("Weighted Sum\nL = ReLU(Σₖ wₖ · Aᵏ)",   "#A9DFBF",      C["dark_green"], "gradient-free"),
             ("Score-CAM Heatmap\n[0, 1] overlay",      "#D5F5E3",      C["dark_green"], "stable, clean"),
         ]),
    ]

    bw, bh, gap = 0.72, 0.076, 0.016
    y_start = 0.855

    for (ax, title, subtitle, color, steps) in method_configs:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        # column header
        header_box(ax, 0.5, 0.965, 0.92, 0.040, title, fc=color, fs=14)
        ax.text(0.5, 0.915, subtitle, ha="center", fontsize=9.5,
                color=C["gray"], style="italic")

        y = y_start
        centers_y = []
        for lbl, fc, ec, note in steps:
            mb = FancyBboxPatch((0.5 - bw/2, y - bh/2), bw, bh,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=3)
            ax.add_patch(mb)
            lines = lbl.split("\n")
            dy3 = 0.012 if len(lines) > 1 else 0
            for li, line in enumerate(lines):
                ax.text(0.5, y + dy3 - li*0.024, line,
                        ha="center", va="center",
                        fontsize=9.5 if len(line) < 30 else 8.5,
                        color=C["text"], fontweight="bold" if li==0 else "normal",
                        zorder=4)
            if note:
                ax.text(0.5 + bw/2 + 0.02, y, f"← {note}",
                        ha="left", va="center", fontsize=9.5,
                        color=color, style="italic", fontweight="bold", zorder=4)
            centers_y.append(y)
            y -= bh + gap

        # arrows between boxes
        for i in range(len(centers_y)-1):
            arrow(ax, 0.5, centers_y[i] - bh/2 - 0.002,
                      0.5, centers_y[i+1] + bh/2 + 0.002,
                  color=color, lw=1.8, mutation=11, shrink=0)

        # key approach callout — enlarged, properly structured box
        key_texts = {
            "Grad-CAM":   "KEY APPROACH\nGlobal-average-pooled first-order\ngradient weights per feature map",
            "Grad-CAM++": "KEY APPROACH\nPixel-wise second-order alpha\ncoefficients → better localisation\nof small lesions",
            "Score-CAM":  "KEY APPROACH\nNo gradients used — channel\nimportance = model confidence\non masked input image",
        }
        ax.text(0.5, 0.055, key_texts[title], ha="center", va="center",
                fontsize=10, color=color, fontweight="bold",
                bbox=dict(fc=C["light_gray"], ec=color, lw=2.5,
                          boxstyle="round,pad=0.6"))

    # No column separator lines — removed as requested

    save(fig, "Fig3_XAI_Methods_Technical.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4  –  Annotation & IoU Evaluation Protocol
# ═════════════════════════════════════════════════════════════════════════════
def fig4_annotation():
    print("Generating Fig 4: Annotation & Evaluation Protocol...")
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96,
            "Bounding-Box Annotation Protocol and Quantitative XAI Evaluation",
            ha="center", fontsize=18, fontweight="bold", color=C["deep"])
    ax.text(0.5, 0.915, "From raw leaf image to IoU, Lesion Coverage, and Top-20% Localisation Accuracy metrics",
            ha="center", fontsize=10, color=C["gray"])

    # ── Top row: annotation pipeline ──────────────────────────────────────
    section_label(ax, 0.03, 0.85, "  1. Ground Truth Annotation Pipeline  ", C["deep"])
    top_y   = 0.73
    top_xs  = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    top_bw  = 0.125
    top_bh  = 0.140

    ann_steps = [
        ("Step 1\nOriginal Image", "Leaf photo from\nPlantVillage\ntest set", C["very_pale"], C["mid"]),
        ("Step 2\nLabelImg Tool", "Open image in\nLabelImg\n(bounding box)", C["very_pale"], C["mid"]),
        ("Step 3\nBox Draw", "Draw bounding\nbox around\nvisible lesion", C["very_pale"], C["mid"]),
        ("Step 4\nCSV Export", "Export as CSV\n(x1, y1, x2, y2)\nper annotation", C["pale"], C["deep"]),
        ("Step 5\nMask Convert", "CSV → binary\nmask via\ncv2.rectangle()", C["pale"], C["deep"]),
        ("Step 6\nGround Truth", "Binary H×W array\n1 = lesion box\n0 = background", "#D5F5E3", C["dark_green"]),
    ]

    for (lbl, sub, fc, ec), x in zip(ann_steps, top_xs):
        box(ax, x, top_y, top_bw, top_bh, lbl, sub,
            fc=fc, ec=ec, lw=2.0, fs=10.5, sfs=8.5, bold=True, radius=0.010)

    for i in range(len(top_xs)-1):
        arrow(ax, top_xs[i] + top_bw/2 + 0.003, top_y,
                  top_xs[i+1] - top_bw/2 - 0.003, top_y,
              color=C["deep"], lw=2.5, mutation=15)

    # Callout: Bounding box vs pixel mask (attached to Step 3)
    bbox_w, bbox_h = 0.22, 0.13
    bbox_x, bbox_y = 0.40, 0.53
    bbox_box = FancyBboxPatch((bbox_x - bbox_w/2, bbox_y - bbox_h/2), bbox_w, bbox_h,
                              boxstyle="round,pad=0,rounding_size=0.008",
                              lw=1.8, edgecolor=C["orange"], facecolor="#FDEBD0", zorder=3)
    ax.add_patch(bbox_box)
    ax.text(bbox_x, bbox_y + 0.045, "Why pixel masks > bounding boxes?", ha="center",
            fontsize=10, fontweight="bold", color=C["orange"], zorder=4)
    for k, txt in enumerate([
        "Bounding box includes healthy tissue",
        "Inflates IoU denominator → lowers scores",
        "Pixel mask captures exact lesion area",
        "IoU becomes directly comparable"
    ]):
        ax.text(bbox_x - 0.095, bbox_y + 0.015 - k*0.022, f"• {txt}", ha="left",
                fontsize=8.5, color=C["text"], zorder=4)
    # Arrow from Step 3 to the box
    arrow(ax, 0.40, top_y - top_bh/2 - 0.005, 0.40, bbox_y + bbox_h/2 + 0.005,
          color=C["orange"], lw=1.8, mutation=12)

    # Annotation stats callout (top right)
    stats_x, stats_y = 0.94, 0.835
    ax.text(stats_x, stats_y + 0.015, "Annotation Stats", ha="center", fontsize=9.5,
            fontweight="bold", color=C["deep"],
            bbox=dict(fc=C["very_pale"], ec=C["mid"], lw=1.5, boxstyle="round,pad=0.5"))
    for k, txt in enumerate(["• 542 bounding boxes","• 200 eval images (199 used)",
                               "• 3 disease classes","• LabelImg tool"]):
        ax.text(stats_x, stats_y - 0.025 - k*0.022, txt, ha="center",
                fontsize=8.5, color=C["text"])

    # ── Bottom row: metric computation ────────────────────────────────────
    section_label(ax, 0.03, 0.42, "  2. Quantitative Metric Computation Pipeline  ", C["deep"])

    bot_y = 0.25
    
    # Left side: XAI Pipeline
    box(ax, 0.16, bot_y, 0.15, 0.12, "Grad-CAM /\nGrad-CAM++ /\nScore-CAM", 
        "Generated Heatmap H(x)\nContinuous ∈ [0, 1]", fc=C["pale"], ec=C["deep"], lw=2.0, fs=10.5, sfs=8.5, bold=True)
    
    arrow(ax, 0.235, bot_y, 0.285, bot_y, color=C["mid"], lw=2.5, mutation=15)
    
    box(ax, 0.36, bot_y, 0.15, 0.12, "Threshold\nHeatmap", 
        "H_mask = (H ≥ P₇₀)\nBinary mask of top 30%", fc=C["very_pale"], ec=C["mid"], lw=2.0, fs=10.5, sfs=8.5, bold=True)

    # Combine them into a central Metrics Evaluation Block
    eval_x = 0.62
    eval_w = 0.28
    eval_h = 0.20
    eval_box = FancyBboxPatch((eval_x - eval_w/2, bot_y - eval_h/2), eval_w, eval_h,
                              boxstyle="round,pad=0,rounding_size=0.01",
                              lw=2.5, edgecolor=C["purple"], facecolor="#F4ECF7", zorder=1)
    ax.add_patch(eval_box)
    ax.text(eval_x, bot_y + 0.075, "Evaluation Metrics Output", ha="center", fontsize=12, fontweight="bold", color=C["purple"])
    
    metrics = [
        ("1. Mean IoU", "Overlap of H_mask and GT"),
        ("2. Lesion Coverage", "Fraction of GT within H_mask"),
        ("3. Loc. Accuracy", "Top-20% brightest pixels hit GT")
    ]
    for i, (m_title, m_desc) in enumerate(metrics):
        ax.text(eval_x - 0.12, bot_y + 0.02 - i*0.04, m_title, ha="left", fontsize=10, fontweight="bold", color=C["deep"])
        ax.text(eval_x - 0.12, bot_y - 0.00 - i*0.04, m_desc, ha="left", fontsize=9, color=C["text"])
        
    # Arrows to evaluation block
    arrow(ax, 0.435, bot_y, eval_x - eval_w/2 - 0.005, bot_y, color=C["mid"], lw=2.5, mutation=15)
    
    # GT drops down from Step 6 to bot_y (on the right)
    gt_x = 0.85
    mid_y = bot_y + eval_h/2 + 0.06
    ax.plot([gt_x, gt_x, eval_x], [top_y - top_bh/2 - 0.005, mid_y, mid_y], color=C["dark_green"], lw=3, zorder=2)
    arrow(ax, eval_x, mid_y, eval_x, bot_y + eval_h/2 + 0.005, color=C["dark_green"], lw=3, mutation=18)
    
    # Label the green GT mask arrow
    ax.text(gt_x - 0.015, (top_y - top_bh/2 + mid_y)/2, "Ground Truth\nPixel Mask", ha="right", va="center", fontsize=10, color=C["dark_green"], fontweight="bold", style="italic")

    # Final output arrow
    out_x = 0.88
    arrow(ax, eval_x + eval_w/2 + 0.005, bot_y, out_x - 0.075, bot_y, color=C["purple"], lw=2.5, mutation=15)
    box(ax, out_x, bot_y, 0.15, 0.12, "Final Output", "Method Ranking Table\n(Grad-CAM vs ++ vs Score)", fc="#D7BDE2", ec=C["purple"], lw=2.5, fs=10.5, sfs=8.5, bold=True)

    save(fig, "Fig4_Annotation_Evaluation.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5  –  Robustness Testing Protocol
# ═════════════════════════════════════════════════════════════════════════════
def fig5_robustness():
    print("Generating Fig 5: Robustness Testing Protocol...")
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(C["bg"])

    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 1); ax_main.set_ylim(0, 1)
    ax_main.axis("off")
    ax_main.set_facecolor(C["bg"])

    ax_main.text(0.5, 0.97,
                 "Dual-Metric Robustness Testing Protocol",
                 ha="center", fontsize=19, fontweight="bold", color=C["deep"])
    ax_main.text(0.5, 0.932,
                 "Classification Accuracy AND Explanation Quality (Grad-CAM++ IoU) "
                 "evaluated under 4 distortion types × 4 severity levels (120 test images)",
                 ha="center", fontsize=12, color=C["gray"])
    ax_main.plot([0.03, 0.97], [0.915, 0.915], color=C["mid"], lw=1.5, alpha=0.5)

    # ── 4 distortion columns ───────────────────────────────────────────────
    dist_configs = [
        ("Gaussian\nNoise", C["deep"], C["pale"],
         "σ levels: 0.01 / 0.03 / 0.05 / 0.10", "Simulates camera sensor\nnoise and dust particles",
         ["Level 1 (Mild)\nσ=0.01", "Level 2\nσ=0.03", "Level 3\nσ=0.05", "Level 4 (Severe)\nσ=0.10"]),
        ("Gaussian\nBlur", C["purple"], "#EDE7F6",
         "Kernel σ: 0.5 / 1.0 / 2.0 / 3.0", "Simulates out-of-focus\nimaging and camera shake",
         ["Level 1 (Mild)\nσ=0.5", "Level 2\nσ=1.0", "Level 3\nσ=2.0", "Level 4 (Severe)\nσ=3.0"]),
        ("Brightness\nShift", C["orange"], "#FDEBD0",
         "Factor f: 0.6 / 0.8 / 1.2 / 1.4", "Simulates underexposure,\noverexposure, glare",
         ["Level 1 (Dark)\nf=0.6", "Level 2\nf=0.8", "Level 3\nf=1.2", "Level 4 (Bright)\nf=1.4"]),
        ("Random\nOcclusion", C["dark_green"], "#D5F5E3",
         "Patch area: 10% / 20% / 30% / 40%", "Simulates overlapping\nleaves, partial coverage",
         ["Level 1 (Mild)\n10%", "Level 2\n20%", "Level 3\n30%", "Level 4 (Severe)\n40%"]),
    ]

    col_xs  = [0.13, 0.375, 0.625, 0.87]
    col_w   = 0.22
    box_h   = 0.082
    box_gap = 0.020

    for (name, hdr_c, fc, param_str, effect_str, levels), cx in zip(dist_configs, col_xs):
        # column header
        header_box(ax_main, cx, 0.860, col_w, 0.055, name, fc=hdr_c, fs=14)
        ax_main.text(cx, 0.810, param_str, ha="center", fontsize=10.5, color=hdr_c, fontweight="bold")
        ax_main.text(cx, 0.782, effect_str, ha="center", fontsize=9.5, color=C["gray"], style="italic")

        # severity level boxes
        y0 = 0.715
        for k, lbl in enumerate(levels):
            by = y0 - k*(box_h + box_gap)
            shade = fc if k % 2 == 0 else C["light_gray"]
            mb = FancyBboxPatch((cx - col_w/2 + 0.005, by - box_h/2),
                                col_w - 0.010, box_h,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=2.0, edgecolor=hdr_c, facecolor=shade, zorder=3)
            ax_main.add_patch(mb)
            ax_main.text(cx, by, lbl, ha="center", va="center", fontsize=11, color=C["text"], fontweight="bold", zorder=4)
            if k < 3:
                arrow(ax_main, cx, by - box_h/2 - 0.002, cx, y0 - (k+1)*(box_h + box_gap) + box_h/2 + 0.002,
                      color=hdr_c, lw=2.5, mutation=15, shrink=0)

        # Severity arrow alongside
        ax_main.annotate("",
            xy=(cx + col_w/2 + 0.015, y0 - 3*(box_h+box_gap)),
            xytext=(cx + col_w/2 + 0.015, y0 + box_h/2),
            arrowprops=dict(arrowstyle="-|>", color=hdr_c, lw=2.5, mutation_scale=15))
        ax_main.text(cx + col_w/2 + 0.038, y0 - 1.5*(box_h+box_gap), "Severity", ha="center", fontsize=11, color=hdr_c, fontweight="bold", rotation=90)

    # ── Bottom: dual-metric computation boxes ─────────────────────────────
    # Connection logic:
    bot_box_y = 0.715 - 3*(box_h + box_gap) - box_h/2
    
    # Draw an explicit junction for the flows
    junction_y = 0.355
    ax_main.plot([0.13, 0.87], [junction_y, junction_y], color=C["mid"], lw=2.5, zorder=1)
    
    # Arrows from bottom of columns to junction
    for cx, (_, hdr_c, *_) in zip(col_xs, dist_configs):
        arrow(ax_main, cx, bot_box_y - 0.005, cx, junction_y + 0.005, color=hdr_c, lw=3.0, mutation=15)
        
    # Text centered below the junction line, clear of the wider columns
    ax_main.text(0.5, junction_y - 0.030, "At each severity level — two metrics are computed", 
                 ha="center", va="center", fontsize=12.5, fontweight="bold", color=C["deep"], 
                 bbox=dict(fc=C["bg"], ec="none", pad=2), zorder=2)

    metric_y = 0.210
    metric_boxes = [
        (0.25, "Classification Accuracy", "Correct predictions / Total\nPer class and overall", C["deep"], C["pale"]),
        (0.75, "Explanation IoU\n(Grad-CAM++)", "Grad-CAM++ heatmap mask\nvs. ground-truth lesion mask", C["purple"], "#EDE7F6"),
    ]
    for (mx, lbl, sub, ec, fc) in metric_boxes:
        box(ax_main, mx, metric_y, 0.30, 0.11, lbl, sub, fc=fc, ec=ec, lw=2.5, fs=13, sfs=10.5, bold=True, radius=0.012)
        arrow(ax_main, mx, junction_y - 0.005, mx, metric_y + 0.055 + 0.005, color=ec, lw=3.0, mutation=18)

    # AUC box
    auc_y = 0.095
    box(ax_main, 0.50, auc_y, 0.45, 0.08,
        "Robustness AUC Score",
        "Area Under Curve (AUC) for Accuracy and IoU over the 4 severity levels",
        fc="#D5F5E3", ec=C["dark_green"], lw=2.5, fs=13, sfs=10.5, bold=True, radius=0.012)

    arrow(ax_main, 0.25, metric_y - 0.055 - 0.005, 0.45, auc_y + 0.04 + 0.005, color=C["deep"], lw=2.5, mutation=15)
    arrow(ax_main, 0.75, metric_y - 0.055 - 0.005, 0.55, auc_y + 0.04 + 0.005, color=C["purple"], lw=2.5, mutation=15)

    # ── Key finding callout ────────────────────────────────────────────────
    ax_main.text(0.50, 0.035,
                 "Expected Key Finding: Explanation quality (IoU) degrades faster than classification accuracy under Gaussian blur,\n"
                 "because high-frequency gradient signals required by Grad-CAM++ are destroyed even when the classifier retains enough low-frequency\n"
                 "cues to predict correctly. This is the core discussion point of Section 5.4 in the revised paper.",
                 ha="center", va="center", fontsize=11.5, color=C["deep"], fontweight="bold",
                 bbox=dict(fc=C["very_pale"], ec=C["deep"], lw=2.5, boxstyle="round,pad=0.5"))

    save(fig, "Fig5_Robustness_Protocol.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6  –  Complete Publication Methodology Summary (one-page overview)
# ═════════════════════════════════════════════════════════════════════════════
def fig6_summary():
    print("Generating Fig 6: One-Page Methodology Summary...")
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.97,
            "Complete Methodology Overview — One-Page Summary",
            ha="center", fontsize=17, fontweight="bold", color=C["deep"])
    ax.text(0.5, 0.935,
            "EfficientNetB0  ·  Grad-CAM / Grad-CAM++ / Score-CAM  ·  "
            "Pixel-Level XAI Evaluation  ·  Robustness  ·  TFLite Deployment",
            ha="center", fontsize=9, color=C["gray"])
    ax.plot([0.03, 0.97], [0.918, 0.918], color=C["mid"], lw=1.0, alpha=0.5)

    # ── 5 horizontal phase rows ────────────────────────────────────────────
    phases = [
        # (label, color, items_list)
        ("Phase 1\nData &\nAnnotation",
         C["mid"],
         ["PlantVillage\n4-class apple\n8,844 / 2,704 / 2,694",
          "Preprocessing\n224×224 · norm\naugmentation",
          "LabelImg\nBounding-Box\nAnnotation",
          "542 Annotations\n200 eval images\n3 disease classes"]),
        ("Phase 2\nModel\nTraining",
         C["deep"],
         ["EfficientNetB0\nfrozen backbone\n4,049,571 params",
          "Dense(4) Head\n(trainable)\n5,124 params",
          "Adam lr=1e-3\nbatch=32\n5 epochs",
          "Best weights\nval_accuracy\nModelCheckpoint"]),
        ("Phase 3\nXAI\nComparison",
         C["purple"],
         ["Grad-CAM\n∇yᶜ global\navg. pooled",
          "Grad-CAM++\n2nd-order α\npixel weights",
          "Score-CAM\nMasked-input\nscoring",
          "IoU · Coverage\nLoc. Accuracy\n3-method table"]),
        ("Phase 4\nRobustness\nTesting",
         C["orange"],
         ["Gaussian Noise\nσ: 0.01–0.10\nAUC metric",
          "Gaussian Blur\nσ: 0.5–3.0\nAUC metric",
          "Brightness Shift\nf: 0.6–1.4\nAUC metric",
          "Occlusion\n10%–40%\nDual-metric"]),
        ("Phase 5\nDeployment\n& Paper",
         C["dark_green"],
         ["TFLite FP32\n15.31 MB\n27.18 ms/img",
          "Dynamic Quant.\n4.34 MB\n52.88 ms/img",
          "Revised Paper\nXAI comparison\nas main result",
          "Journal Submit\nApplied Sciences\n(MDPI)"]),
    ]

    ph_xs  = [0.21, 0.355, 0.50, 0.645]   # label column x
    item_col_xs = [0.21, 0.355, 0.50, 0.645] # four item cols
    row_h  = 0.100
    y_top  = 0.795
    row_gap = 0.018

    for row_idx, (ph_lbl, ph_c, items) in enumerate(phases):
        row_y = y_top - row_idx * (row_h + row_gap)

        # phase label strip
        pl = FancyBboxPatch((0.02, row_y - row_h/2), 0.090, row_h,
                            boxstyle="round,pad=0,rounding_size=0.007",
                            lw=2.5, edgecolor=ph_c, facecolor=ph_c, zorder=3)
        ax.add_patch(pl)
        ax.text(0.065, row_y, ph_lbl, ha="center", va="center",
                fontsize=10.5, color=C["white"], fontweight="bold", zorder=4)

        arrow(ax, 0.112, row_y, 0.142, row_y,
              color=ph_c, lw=2.5, mutation=12, shrink=0)

        # item boxes
        for col_idx, (itm, ix) in enumerate(zip(items, item_col_xs)):
            ib = FancyBboxPatch((ix - 0.0625, row_y - row_h/2),
                                0.125, row_h,
                                boxstyle="round,pad=0,rounding_size=0.007",
                                lw=2.0, edgecolor=ph_c,
                                facecolor=C["very_pale"] if row_idx%2==0 else C["pale"],
                                zorder=3)
            ax.add_patch(ib)
            lines = itm.split("\n")
            dy = 0.025 if len(lines) > 2 else 0.012
            for li, line in enumerate(lines):
                ax.text(ix, row_y + dy - li*0.022, line,
                        ha="center", va="center",
                        fontsize=9.5 if li==0 else 8.5,
                        fontweight="bold" if li==0 else "normal",
                        color=C["text"], zorder=4)
            if col_idx < len(items)-1:
                arrow(ax, ix + 0.0625 + 0.002, row_y,
                          item_col_xs[col_idx+1] - 0.0625 - 0.002, row_y,
                      color=ph_c, lw=1.5, mutation=10, shrink=0)

        # row separator
        if row_idx < len(phases)-1:
            sep_y = row_y - row_h/2 - row_gap/2
            ax.plot([0.02, 0.75], [sep_y, sep_y],
                    color=C["light_gray"], lw=1.0, ls="--", zorder=2)

    # ── Outputs column (right) ─────────────────────────────────────────────
    out_x = 0.875
    ax.text(out_x, y_top + 0.060, "Key Outputs", ha="center",
            fontsize=11.5, fontweight="bold", color=C["deep"],
            bbox=dict(fc=C["pale"], ec=C["mid"], lw=1.5,
                      boxstyle="round,pad=0.45"))
    out_items = [
        ("dataset_split.png\nclass_distribution.png",       C["mid"]),
        ("training_curves.png\nclassification_report.csv",  C["deep"]),
        ("xai_4method_comparison.png\nxai_comparison_results.csv", C["purple"]),
        ("robustness_curves.png\nexplanation_robustness_auc.csv",  C["orange"]),
        ("deployment_benchmark.png\npublication_summary.png",      C["dark_green"]),
    ]
    for row_idx, ((txt, col), _) in enumerate(zip(out_items, phases)):
        oy = y_top - row_idx*(row_h + row_gap)
        ob = FancyBboxPatch((out_x - 0.100, oy - row_h/2),
                            0.200, row_h,
                            boxstyle="round,pad=0,rounding_size=0.007",
                            lw=1.8, edgecolor=col,
                            facecolor=C["very_pale"], zorder=3)
        ax.add_patch(ob)
        for li, line in enumerate(txt.split("\n")):
            ax.text(out_x, oy + 0.012 - li*0.022, line,
                    ha="center", va="center",
                    fontsize=9.0, color=C["text"], style="italic", zorder=4)
        
        # Connect Phase 4th item box to the Output box
        arrow(ax, 0.645 + 0.0625 + 0.002, oy, out_x - 0.100 - 0.002, oy,
              color=col, lw=2.0, mutation=12, shrink=0)

    # vertical output column separator
    ax.plot([0.755, 0.755], [0.240, 0.880], color=C["light_gray"],
            lw=1.2, ls="--", zorder=2)

    # ── Novelty claim box ─────────────────────────────────────────────────
    ax.text(0.50, 0.150,
            "Novelty Claim: First quantitative comparison of Grad-CAM, Grad-CAM++, and Score-CAM on apple leaf disease\n"
            "using pixel-level lesion annotations, with explanation robustness analysis under controlled image distortions.",
            ha="center", va="center", fontsize=11.5, color=C["deep"], fontweight="bold",
            bbox=dict(fc=C["very_pale"], ec=C["deep"], lw=2.5,
                      boxstyle="round,pad=0.6"))

    save(fig, "Fig6_Methodology_Summary.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n Generating all methodology figures ...\n")
    fig1_pipeline()
    fig2_architecture()
    fig3_xai_methods()
    fig4_annotation()
    fig5_robustness()
    fig6_summary()
    print(f"\n All 6 figures saved to: {os.path.abspath(OUT_DIR)}/\n")
    print("  Fig1_System_Pipeline.png")
    print("  Fig2_EfficientNetB0_Architecture.png")
    print("  Fig3_XAI_Methods_Technical.png")
    print("  Fig4_Annotation_Evaluation.png")
    print("  Fig5_Robustness_Protocol.png")
    print("  Fig6_Methodology_Summary.png")
    print()
    print("Usage in Google Colab:")
    print("  !python generate_methodology_figures.py")
    print("  from IPython.display import Image")
    print("  Image('methodology_figures/Fig1_System_Pipeline.png')")