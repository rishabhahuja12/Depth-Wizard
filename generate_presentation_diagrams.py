"""
Generate High-Resolution Diagrammatic Assets for DepthWizard SIH 2026 Presentation.
Designed for maximum clarity, professional aesthetic, and effortless explanation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUT_DIR = Path(r"E:\rishabh\sih\DepthWizard\presentation_assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Theme Palette (ISRO Navy & Electric Cyan)
BG_DARK = "#0B132B"
CARD_BG = "#1C2541"
CARD_BORDER = "#3A86FF"
CYAN = "#00E5FF"
SKY = "#38BDF8"
EMERALD = "#10B981"
AMBER = "#F59E0B"
CORAL = "#F43F5E"
WHITE = "#FFFFFF"
MUTED = "#94A3B8"


def create_slide2_flowchart():
    """Slide 2: End-to-End System Solution Flowchart."""
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    stages = [
        {"num": "01", "title": "Optical Ingestion", "desc": "Single-View Optical Sat\n(Cartosat / Maxar / Landsat)\nGeoTIFF / PNG Input", "color": SKY},
        {"num": "02", "title": "Neural Estimation", "desc": "Depth Anything V2 (DPT)\nViT Multi-Scale Feature Fusion\nPer-Sample Norm (0-1)", "color": CYAN},
        {"num": "03", "title": "Dual Geocalibration", "desc": "DTM Affine Polynomial +\nGSD Scale Parameter (m/px)\nAbsolute Metric Z (Meters)", "color": EMERALD},
        {"num": "04", "title": "3D WebGL Flythrough", "desc": "Three.js PlaneGeometry\nCustom Elevation Vertex Shader\n6-DoF Real-Time Camera", "color": AMBER},
        {"num": "05", "title": "Disaster Analytics", "desc": "Dynamic Flood Simulation\nTransect Cross-Section\n32-bit GeoTIFF Export", "color": CORAL},
    ]

    box_w = 2.0
    box_h = 2.8
    spacing = 2.35
    start_x = 0.4
    start_y = 0.7

    for i, s in enumerate(stages):
        x = start_x + i * spacing
        # Background card
        rect = patches.FancyBboxPatch(
            (x, start_y), box_w, box_h,
            boxstyle="round,pad=0.12,rounding_size=0.15",
            facecolor=CARD_BG, edgecolor=s["color"], linewidth=2.0
        )
        ax.add_patch(rect)

        # Stage number badge
        badge = patches.Circle((x + 0.35, start_y + box_h - 0.35), 0.22, facecolor=s["color"])
        ax.add_patch(badge)
        ax.text(x + 0.35, start_y + box_h - 0.35, s["num"], color=BG_DARK, fontsize=11,
                fontweight="bold", ha="center", va="center")

        # Title
        ax.text(x + 0.7, start_y + box_h - 0.35, s["title"], color=WHITE, fontsize=11.5,
                fontweight="bold", va="center")

        # Divider line
        ax.plot([x + 0.15, x + box_w - 0.15], [start_y + box_h - 0.7, start_y + box_h - 0.7],
                color=s["color"], alpha=0.5, lw=1.2)

        # Description
        ax.text(x + 0.15, start_y + box_h - 0.95, s["desc"], color=WHITE, fontsize=8.8,
                va="top", linespacing=1.4)

        # Forward Arrow
        if i < len(stages) - 1:
            arrow_x = x + box_w + 0.05
            ax.annotate("", xy=(arrow_x + 0.25, start_y + box_h / 2),
                        xytext=(arrow_x, start_y + box_h / 2),
                        arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.4",
                                        color=CYAN, lw=2.5))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    out_path = OUT_DIR / "diagram_slide2_flowchart.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def create_slide3_architecture():
    """Slide 3: 5-Tier Technical Architecture Swimlane Diagram."""
    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    tiers = [
        {"name": "TIER 1: INGESTION & GEOSPATIAL PARSER", "tech": "Python / GDAL / Rasterio / NumPy",
         "items": ["GeoTIFF & Optical PNG Ingestion", "EPSG:32617 / UTM Geotransform", "2%-98% Contrast Percentile Stretch", "HDF5 Tile Quadrant Pipeline"], "color": SKY},
        {"name": "TIER 2: NEURAL DEPTH ESTIMATION ENGINE", "tech": "PyTorch / Transformers / DPT / AMP",
         "items": ["Depth Anything V2 (DPT) Backbone", "Multi-Scale ViT Feature Assembly", "SILog Loss + Gradient Matching Loss", "Per-Sample Normalization Guard"], "color": CYAN},
        {"name": "TIER 3: METRIC GEOCALIBRATION PIPELINE", "tech": "SciPy / Affine DTM / MC Dropout",
         "items": ["Polynomial DTM Terrain Baseline", "GSD Ground Sampling (m/px) Scaling", "nDSM Normalized Elevation Offset", "MC Dropout Uncertainty Bounds"], "color": EMERALD},
        {"name": "TIER 4: REAL-TIME 3D WEBGL ENGINE", "tech": "React / Three.js / R3F / GLSL",
         "items": ["256x256 Dynamic PlaneGeometry", "Custom Displacement Vertex Shader", "6-DoF WASD + Pointer-Lock Flight", "Real-time Elevation Legend (HSL)"], "color": AMBER},
        {"name": "TIER 5: DISASTER DECISION SUPPORT & EXPORT", "tech": "FastAPI / Recharts / OGC GeoTIFF",
         "items": ["Dynamic Flood Inundation Simulator", "Metric Cross-Section Profile Transect", "Topographic Contour Line Overlay", "32-bit Metric GeoTIFF Exporter"], "color": CORAL},
    ]

    tier_h = 0.82
    tier_w = 11.2
    start_x = 0.4
    start_y = 4.2

    for idx, t in enumerate(tiers):
        y = start_y - idx * 0.95
        rect = patches.FancyBboxPatch(
            (start_x, y), tier_w, tier_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=CARD_BG, edgecolor=t["color"], linewidth=1.8
        )
        ax.add_patch(rect)

        # Header tag
        tag = patches.FancyBboxPatch(
            (start_x + 0.15, y + tier_h - 0.32), 3.8, 0.26,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=t["color"], edgecolor="none"
        )
        ax.add_patch(tag)
        ax.text(start_x + 0.25, y + tier_h - 0.19, t["name"], color=BG_DARK,
                fontsize=8.5, fontweight="bold", va="center")

        # Tech stack badge
        ax.text(start_x + tier_w - 0.2, y + tier_h - 0.19, f"[{t['tech']}]", color=MUTED,
                fontsize=8.0, fontweight="bold", ha="right", va="center")

        # 4 items in a row
        col_w = (tier_w - 0.5) / 4.0
        for item_idx, item in enumerate(t["items"]):
            item_x = start_x + 0.25 + item_idx * col_w
            item_y = y + 0.22
            # small bullet
            ax.plot([item_x], [item_y], marker="o", markersize=3.5, color=t["color"])
            ax.text(item_x + 0.12, item_y, item, color=WHITE, fontsize=7.8, va="center")

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.0)
    out_path = OUT_DIR / "diagram_slide3_architecture.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def create_slide4_swot_risks():
    """Slide 4: Feasibility SWOT Matrix + Risk-Mitigation Chevrons."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=300, gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.patch.set_facecolor(BG_DARK)

    # Left: SWOT Matrix
    ax1.set_facecolor(BG_DARK)
    ax1.axis("off")
    ax1.set_title("STRATEGIC SWOT ASSESSMENT", color=CYAN, fontsize=12, fontweight="bold", pad=12)

    swot_cards = [
        {"x": 0.05, "y": 2.2, "title": "STRENGTHS", "color": EMERALD,
         "bullets": ["Sub-second inference (<600ms)", "Zero stereo cost (1 satellite pass)", "Direct browser WebGL (No GIS install)", "Full 15,348 GAMUS patch training"]},
        {"x": 2.55, "y": 2.2, "title": "WEAKNESSES", "color": AMBER,
         "bullets": ["Single-view scale ambiguity", "Edge sharpness on complex roofs", "GSD metadata dependence", "High-res tiled stitch overhead"]},
        {"x": 0.05, "y": 0.1, "title": "OPPORTUNITIES", "color": SKY,
         "bullets": ["ISRO Bhuvan / MOSDAC integration", "NDMA national disaster flood relief", "Drone & UAV urban 3D mapping", "Commercial satellite API plug-ins"]},
        {"x": 2.55, "y": 0.1, "title": "THREATS", "color": CORAL,
         "bullets": ["Dense cloud cover & monsoon haze", "Specular reflection over water", "Uncalibrated camera sensor angles", "Large coordinate misprojection"]},
    ]

    card_w = 2.4
    card_h = 1.95
    for c in swot_cards:
        rect = patches.FancyBboxPatch(
            (c["x"], c["y"]), card_w, card_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=CARD_BG, edgecolor=c["color"], linewidth=1.8
        )
        ax1.add_patch(rect)
        ax1.text(c["x"] + 0.15, c["y"] + card_h - 0.25, c["title"], color=c["color"],
                fontsize=9.5, fontweight="bold")
        for b_idx, b in enumerate(c["bullets"]):
            ax1.text(c["x"] + 0.15, c["y"] + card_h - 0.55 - b_idx * 0.35, f"• {b}",
                    color=WHITE, fontsize=7.2, va="top")

    ax1.set_xlim(0, 5.1)
    ax1.set_ylim(0, 4.3)

    # Right: Risk & Mitigation Chevrons
    ax2.set_facecolor(BG_DARK)
    ax2.axis("off")
    ax2.set_title("RISK MITIGATION CHEVRONS", color=EMERALD, fontsize=12, fontweight="bold", pad=12)

    risks = [
        {"risk": "Monocular Scale Ambiguity", "mitigation": "Hybrid GSD Ground Sampling + DTM Baseline Polynomial", "color": SKY},
        {"risk": "High-Res Tiling Artifacts", "mitigation": "Quadrant Splitting with 50% Overlap Cosine Blending", "color": CYAN},
        {"risk": "Client Hardware Limitations", "mitigation": "FP16 Model (<3.2GB VRAM) & WebGL LOD Mesh Decimation", "color": EMERALD},
        {"risk": "Geospatial Projection Drift", "mitigation": "Strict EPSG/UTM Affine Georeferencing via Rasterio", "color": AMBER},
    ]

    box_h2 = 0.8
    for r_idx, r in enumerate(risks):
        y2 = 3.3 - r_idx * 0.95
        rect = patches.FancyBboxPatch(
            (0.1, y2), 4.6, box_h2,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=CARD_BG, edgecolor=r["color"], linewidth=1.6
        )
        ax2.add_patch(rect)
        # Risk header
        ax2.text(0.25, y2 + box_h2 - 0.22, f"RISK: {r['risk']}", color=CORAL,
                fontsize=8.5, fontweight="bold")
        # Arrow & Mitigation
        ax2.text(0.25, y2 + 0.22, f"SOLVED: {r['mitigation']}", color=EMERALD,
                fontsize=7.8, fontweight="bold")

    ax2.set_xlim(0, 4.8)
    ax2.set_ylim(0, 4.3)

    out_path = OUT_DIR / "diagram_slide4_swot_risks.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def create_slide5_impact_ecosystem():
    """Slide 5: 4-Quadrant Impact Ecosystem Hub & Key Metrics."""
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    # Center Hub: DepthWizard Core
    center_circle = patches.Circle((6.0, 2.2), 1.1, facecolor=CARD_BG, edgecolor=CYAN, linewidth=2.8)
    ax.add_patch(center_circle)
    ax.text(6.0, 2.45, "DEPTHWIZARD", color=WHITE, fontsize=11, fontweight="bold", ha="center")
    ax.text(6.0, 2.15, "GEOSPATIAL HUB", color=CYAN, fontsize=9.5, fontweight="bold", ha="center")
    ax.text(6.0, 1.85, "ISRO SAC (SIH26175)", color=MUTED, fontsize=8.0, ha="center")

    quadrants = [
        {"x": 0.5, "y": 2.3, "title": "DISASTER RESPONSE & FLOOD RELIEF", "color": CORAL,
         "bullets": ["Real-time inundation simulation (<600ms)", "Immediate evacuation route elevation profiles", "Rapid assessment without stereo flight delays"], "anchor": (5.0, 2.5)},
        {"x": 7.5, "y": 2.3, "title": "DEFENSE & TACTICAL RECONNAISSANCE", "color": AMBER,
         "bullets": ["Instant 3D digital twin from single sat pass", "Line-of-sight & terrain vantage analysis", "Zero drone / aircraft sensor exposure"], "anchor": (7.0, 2.5)},
        {"x": 0.5, "y": 0.1, "title": "URBAN PLANNING & SMART CITIES", "color": SKY,
         "bullets": ["Building height & rooftop density survey", "Stormwater runoff & drainage slope modeling", "Decentralized browser-accessible spatial GIS"], "anchor": (5.0, 1.9)},
        {"x": 7.5, "y": 0.1, "title": "ENVIRONMENT & ECOLOGY MONITORING", "color": EMERALD,
         "bullets": ["Glacial lake elevation change tracking", "Landslide & slope vulnerability mapping", "Coastal erosion & sediment deposition study"], "anchor": (7.0, 1.9)},
    ]

    card_w = 4.0
    card_h = 1.95
    for q in quadrants:
        rect = patches.FancyBboxPatch(
            (q["x"], q["y"]), card_w, card_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=CARD_BG, edgecolor=q["color"], linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(q["x"] + 0.2, q["y"] + card_h - 0.28, q["title"], color=q["color"],
                fontsize=9.0, fontweight="bold")
        for b_idx, b in enumerate(q["bullets"]):
            ax.text(q["x"] + 0.2, q["y"] + card_h - 0.6 - b_idx * 0.4, f"• {b}",
                    color=WHITE, fontsize=7.8, va="top")

        # Connection line to center
        card_center_x = q["x"] + card_w if q["x"] < 5 else q["x"]
        card_center_y = q["y"] + card_h / 2
        ax.plot([card_center_x, q["anchor"][0]], [card_center_y, q["anchor"][1]],
                color=q["color"], linestyle="--", linewidth=1.5, alpha=0.7)

    # 4 Bottom Metric Badges
    metrics = [
        {"val": "< 600 ms", "label": "INFERENCE LATENCY", "color": CYAN},
        {"val": "90%", "label": "COST REDUCTION", "color": EMERALD},
        {"val": "100%", "label": "WEB-BASED (ZERO INSTALL)", "color": SKY},
        {"val": "15,348", "label": "GAMUS PATCHES TRAINED", "color": AMBER},
    ]

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    out_path = OUT_DIR / "diagram_slide5_impact_ecosystem.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


import textwrap

def create_slide6_research_lineage():
    """Slide 6: Research Citations & Verification Benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    columns = [
        {"title": "FOUNDATIONAL AI LITERATURE", "color": CYAN,
         "items": [
             "Ranftl et al., 'Vision Transformers for Dense Prediction' (DPT), IEEE TPAMI 2021.",
             "Yang et al., 'Depth Anything V2: Metric Depth Estimation', arXiv:2406.09414, 2024.",
             "Eigen et al., 'Predicting Depth & Normals with Multi-Scale CNNs', ICCV 2015 (SILog Loss)."
         ]},
        {"title": "DATASETS & BENCHMARKS", "color": EMERALD,
         "items": [
             "GAMUS Dataset: High-res satellite optical & LiDAR DSM benchmark (ESA / EarthFlow 2024).",
             "ISRO Cartosat-1 / Cartosat-2 Stereo DEM Validation Standards (SAC Bhuvan Portal).",
             "SpaceNet 6 Multi-Sensor All-Weather Mapping Dataset (IEEE GRSS Urban Benchmarks)."
         ]},
        {"title": "STANDARDS & FRAMEWORKS", "color": AMBER,
         "items": [
             "Open Geospatial Consortium (OGC): Cloud-Optimized GeoTIFF Standard (OGC 19-008r4).",
             "FastAPI & Starlette High-Throughput Asynchronous Geospatial REST Architecture.",
             "Three.js WebGL Graphics Library for Browser-Native 60 FPS Terrain Rendering."
         ]}
    ]

    col_w = 3.6
    col_h = 3.9
    start_x = 0.4
    start_y = 0.25

    for c_idx, c in enumerate(columns):
        x = start_x + c_idx * 3.85
        rect = patches.FancyBboxPatch(
            (x, start_y), col_w, col_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=CARD_BG, edgecolor=c["color"], linewidth=1.8
        )
        ax.add_patch(rect)

        # Header tag
        tag = patches.FancyBboxPatch(
            (x + 0.12, start_y + col_h - 0.46), col_w - 0.24, 0.34,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=c["color"], edgecolor="none"
        )
        ax.add_patch(tag)
        ax.text(x + col_w / 2, start_y + col_h - 0.29, c["title"], color=BG_DARK,
                fontsize=8.5, fontweight="bold", ha="center", va="center")

        # Items formatted with clean text wrapping
        current_y = start_y + col_h - 0.72
        for i_idx, itm in enumerate(c["items"]):
            wrapped = textwrap.fill(f"[{i_idx+1}] {itm}", width=34)
            num_lines = len(wrapped.split("\n"))
            ax.text(x + 0.18, current_y, wrapped,
                    color=WHITE, fontsize=7.4, va="top", linespacing=1.3)
            current_y -= (num_lines * 0.22 + 0.32)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    out_path = OUT_DIR / "diagram_slide6_research_lineage.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")



if __name__ == "__main__":
    create_slide2_flowchart()
    create_slide3_architecture()
    create_slide4_swot_risks()
    create_slide5_impact_ecosystem()
    create_slide6_research_lineage()
    print("\nAll presentation diagrammatic assets successfully generated in:", OUT_DIR)
