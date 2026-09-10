"""
Build Official SIH 2026 Presentation for DepthWizard (SIH26175 - ISRO SAC).
Adheres strictly to the official 6-slide limit, official template headers,
professional diagrammatic assets, and effortless explanation structure.
"""
import sys
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

TEMPLATE_PATH = Path(r"E:\rishabh\sih\SIH2026-IDEA-Presentation-Format (1).pptx")
OUTPUT_PATH = Path(r"E:\rishabh\sih\DepthWizard_SIH2026_Final_Presentation.pptx")
ASSETS_DIR = Path(r"E:\rishabh\sih\DepthWizard\presentation_assets")

# Professional Color Palette
NAVY_DEEP = RGBColor(11, 19, 43)
NAVY_CARD = RGBColor(28, 37, 65)
CYAN_ACCENT = RGBColor(0, 229, 255)
SKY_BLUE = RGBColor(56, 189, 248)
EMERALD = RGBColor(16, 185, 129)
AMBER = RGBColor(245, 158, 11)
CORAL = RGBColor(244, 63, 94)
WHITE = RGBColor(255, 255, 255)
GRAY_TEXT = RGBColor(203, 213, 225)
MUTED = RGBColor(148, 163, 184)
BORDER_BLUE = RGBColor(58, 134, 255)


def remove_shape(slide, shape):
    """Safely removes a shape from a slide."""
    sp = shape._element
    sp.getparent().remove(sp)


def find_shape_by_text(slide, query):
    """Finds a shape containing specific text."""
    for s in slide.shapes:
        if s.has_text_frame and query.lower() in s.text.lower():
            return s
    return None


def add_card(slide, left, top, width, height, bg_color=NAVY_CARD, border_color=BORDER_BLUE):
    """Adds a styled rounded rectangular card container."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    return shape


def format_bullet(p, bold_prefix, text, font_size=10, prefix_color=CYAN_ACCENT, text_color=WHITE):
    """Formats a bullet point with a bold colored prefix followed by normal text."""
    p.space_after = Pt(4)
    p.space_before = Pt(2)
    # Bold Prefix
    run_pre = p.add_run()
    run_pre.text = bold_prefix + ": "
    run_pre.font.bold = True
    run_pre.font.size = Pt(font_size)
    run_pre.font.color.rgb = prefix_color
    run_pre.font.name = "Segoe UI"

    # Body text
    run_body = p.add_run()
    run_body.text = text
    run_body.font.bold = False
    run_body.font.size = Pt(font_size)
    run_body.font.color.rgb = text_color
    run_body.font.name = "Segoe UI"


def build_presentation():
    print(f"Loading template from {TEMPLATE_PATH}...")
    prs = Presentation(str(TEMPLATE_PATH))
    
    # Check initial slide count
    print(f"Original slides count: {len(prs.slides)}")

    # =========================================================================
    # SLIDE 1: TITLE PAGE
    # =========================================================================
    print("Formatting Slide 1 (Title Page)...")
    s1 = prs.slides[0]

    # Update Subtitle placeholder
    sub_shape = find_shape_by_text(s1, "TITLE PAGE")
    if sub_shape and sub_shape.has_text_frame:
        sub_shape.text_frame.clear()
        p = sub_shape.text_frame.paragraphs[0]
        p.text = "DEPTHWIZARD: SINGLE-VIEW SATELLITE TO METRIC DSM & 3D FLYTHROUGH"
        p.font.bold = True
        p.font.size = Pt(18)
        p.font.color.rgb = CYAN_ACCENT
        p.font.name = "Segoe UI"

    # Replace TextBox 9 with structured project information card
    tb9 = find_shape_by_text(s1, "Problem Statement ID")
    if tb9:
        remove_shape(s1, tb9)

    # Add structured Details Card on the left
    card1 = add_card(s1, Inches(0.5), Inches(2.2), Inches(6.8), Inches(4.5))
    tf1 = card1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.25)
    tf1.margin_right = Inches(0.25)
    tf1.margin_top = Inches(0.2)
    tf1.margin_bottom = Inches(0.2)

    fields = [
        ("Problem Statement ID", "SIH26175"),
        ("Problem Statement Title", "Digital Surface Model (DSM) Generation from High-Resolution Optical Satellite Images"),
        ("Theme", "Space Technology / Remote Sensing / Disaster Management"),
        ("PS Category", "Software (AI / Computer Vision & Geospatial 3D)"),
        ("Target Organization", "Indian Space Research Organisation (ISRO) — Space Applications Centre (SAC)"),
        ("Core Capability", "Single-View RGB to Metric DSM & 60 FPS WebGL Flythrough (<600ms)"),
        ("Team Name & ID", "Your Team Name  |  SIH 2026 Finalist"),
    ]

    for idx, (label, val) in enumerate(fields):
        p = tf1.paragraphs[0] if idx == 0 else tf1.add_paragraph()
        format_bullet(p, f"• {label}", val, font_size=10.5, prefix_color=CYAN_ACCENT, text_color=WHITE)

    # On the right, add hero live working studio preview image
    hero_img = ASSETS_DIR / "02_sample1_geotiff_studio.png"
    if hero_img.exists():
        # Border card behind image
        add_card(s1, Inches(7.5), Inches(2.2), Inches(5.3), Inches(4.5), bg_color=NAVY_CARD, border_color=EMERALD)
        pic = s1.shapes.add_picture(str(hero_img), Inches(7.6), Inches(2.3), Inches(5.1), Inches(3.8))
        # Caption below picture
        tx_box = s1.shapes.add_textbox(Inches(7.6), Inches(6.15), Inches(5.1), Inches(0.45))
        tf_cap = tx_box.text_frame
        p_cap = tf_cap.paragraphs[0]
        p_cap.text = "PROVEN PROTOTYPE: Real-time 3D Flythrough & Flood Inundation Studio (<600ms)"
        p_cap.font.size = Pt(8.5)
        p_cap.font.bold = True
        p_cap.font.color.rgb = EMERALD
        p_cap.alignment = PP_ALIGN.CENTER

    # Remove template clutter from Slide 1 (Picture 4 and Freeform background)
    for s in list(s1.shapes):
        if s.name in ["Picture 4", "Freeform: Shape 26"]:
            remove_shape(s1, s)

    # =========================================================================
    # SLIDE 2: PROPOSED SOLUTION
    # =========================================================================
    print("Formatting Slide 2 (Proposed Solution)...")
    s2 = prs.slides[1]

    # First: Remove placeholder text box from Slide 2 cleanly
    for s in list(s2.shapes):
        if s.has_text_frame and ("detailed explanation" in s.text.lower() or "describe your idea" in s.text.lower()):
            remove_shape(s2, s)

    # Second: Update Title safely
    for s in s2.shapes:
        if s.has_text_frame and "idea title" in s.text.lower():
            s.text_frame.text = "PROPOSED SOLUTION: DEPTHWIZARD SINGLE-VIEW 3D ELEVATION ENGINE"
            for p in s.text_frame.paragraphs:
                p.font.size = Pt(20)
                p.font.bold = True
                p.font.color.rgb = NAVY_DEEP
            break

    # Left Column: Problem Resolution & UVPs (Width = 4.8 inches)
    card_sol1 = add_card(s2, Inches(0.5), Inches(1.3), Inches(4.8), Inches(2.6))
    tf_sol1 = card_sol1.text_frame
    tf_sol1.word_wrap = True
    tf_sol1.margin_left = Inches(0.2)
    tf_sol1.margin_top = Inches(0.15)
    
    p = tf_sol1.paragraphs[0]
    p.text = "PROBLEM RESOLUTION & CORE INNOVATION"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = CYAN_ACCENT

    format_bullet(tf_sol1.add_paragraph(), "Single-View Paradigm", "Eliminates multi-pass stereo satellites and airborne LiDAR (>90% survey cost reduction).", font_size=8.5)
    format_bullet(tf_sol1.add_paragraph(), "Sub-Second Latency", "512x512 elevation grid computed in <600ms on consumer laptop GPU (RTX 4060).", font_size=8.5)
    format_bullet(tf_sol1.add_paragraph(), "Metric Ground Anchoring", "Separates bare-earth DTM slope from nDSM building heights using physical GSD priors.", font_size=8.5)

    card_uvp = add_card(s2, Inches(0.5), Inches(4.05), Inches(4.8), Inches(2.75))
    tf_uvp = card_uvp.text_frame
    tf_uvp.word_wrap = True
    tf_uvp.margin_left = Inches(0.2)
    tf_uvp.margin_top = Inches(0.15)

    p_uvp = tf_uvp.paragraphs[0]
    p_uvp.text = "4 UNIQUE VALUE PROPOSITIONS (UVPs)"
    p_uvp.font.bold = True
    p_uvp.font.size = Pt(11)
    p_uvp.font.color.rgb = EMERALD

    format_bullet(tf_uvp.add_paragraph(), "UVP 1 — Flood Simulator", "Dynamic water altitude slider with live submerged % calculation and 3D water plane.", font_size=8.5, prefix_color=EMERALD)
    format_bullet(tf_uvp.add_paragraph(), "UVP 2 — Cross-Section", "Center-axis metric transect plotting ground elevation profile (9m to 169m).", font_size=8.5, prefix_color=EMERALD)
    format_bullet(tf_uvp.add_paragraph(), "UVP 3 — Contour Shader", "GPU-accelerated topographic isolines with adjustable contour intervals (1m to 50m).", font_size=8.5, prefix_color=EMERALD)
    format_bullet(tf_uvp.add_paragraph(), "UVP 4 — Uncertainty Map", "Monte Carlo Dropout confidence estimation flagging cloud and shadow ambiguity.", font_size=8.5, prefix_color=EMERALD)

    # Right Column: High-DPI Flowchart Diagram + Live Flood Studio Screenshot
    flow_img = ASSETS_DIR / "diagram_slide2_flowchart.png"
    if flow_img.exists():
        s2.shapes.add_picture(str(flow_img), Inches(5.5), Inches(1.3), Inches(7.3), Inches(2.6))

    flood_img = ASSETS_DIR / "03_sample1_flood_active.png"
    if flood_img.exists():
        add_card(s2, Inches(5.5), Inches(4.05), Inches(7.3), Inches(2.75), bg_color=NAVY_CARD, border_color=CYAN_ACCENT)
        s2.shapes.add_picture(str(flood_img), Inches(5.6), Inches(4.15), Inches(7.1), Inches(2.35))
        tx = s2.shapes.add_textbox(Inches(5.6), Inches(6.52), Inches(7.1), Inches(0.25))
        p = tx.text_frame.paragraphs[0]
        p.text = "LIVE WORKING STUDIO: Real-Time Dynamic Flood Inundation Simulation (99.6% Submerged @ 45.1m)"
        p.font.size = Pt(8.0)
        p.font.bold = True
        p.font.color.rgb = CYAN_ACCENT
        p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 3: TECHNICAL APPROACH
    # =========================================================================
    print("Formatting Slide 3 (Technical Approach)...")
    s3 = prs.slides[2]

    t_shape = find_shape_by_text(s3, "TECHNICAL APPROACH")
    if t_shape and t_shape.has_text_frame:
        t_shape.text_frame.text = "TECHNICAL APPROACH: 5-TIER NEURAL PIPELINE & 3D ENGINE"
        for p in t_shape.text_frame.paragraphs:
            p.font.size = Pt(20)
            p.font.bold = True
            p.font.color.rgb = NAVY_DEEP

    tb_placeholder = find_shape_by_text(s3, "Technologies to be used")
    if tb_placeholder:
        remove_shape(s3, tb_placeholder)

    # Top Section: 5-Tier Architecture Swimlane Diagram
    arch_img = ASSETS_DIR / "diagram_slide3_architecture.png"
    if arch_img.exists():
        s3.shapes.add_picture(str(arch_img), Inches(0.5), Inches(1.25), Inches(12.3), Inches(3.6))

    # Bottom Section: 4 Mathematical & Technological Deep Dive Cards
    cards_tech = [
        {"title": "NEURAL BACKBONE", "color": CYAN_ACCENT,
         "bullets": ["Depth Anything V2 (DPT)", "Multi-scale ViT feature fusion", "Fine-tuned on 15,348 GAMUS patches"]},
        {"title": "LOSS FORMULATION", "color": EMERALD,
         "bullets": ["SILog Loss (lambda=0.5 scale invariant)", "Sobel Gradient Matching Loss", "Per-sample normalization guard"]},
        {"title": "TWO-COMPONENT CALIBRATION", "color": AMBER,
         "bullets": ["Z(u,v) = DTM_base + alpha * nDSM", "Affine polynomial ground baseline", "Physical GSD (m/px) resolution prior"]},
        {"title": "3D WEBGL RENDERING", "color": CORAL,
         "bullets": ["Three.js dynamic displacement plane", "Locked 60 FPS flythrough in Chrome", "Atomic 32-bit GeoTIFF export (EPSG)"]},
    ]

    card_w = 2.92
    for c_idx, ct in enumerate(cards_tech):
        c_left = Inches(0.5 + c_idx * 3.12)
        c_shape = add_card(s3, c_left, Inches(5.0), Inches(card_w), Inches(1.8), border_color=ct["color"])
        tf_c = c_shape.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = Inches(0.15)
        tf_c.margin_top = Inches(0.12)

        p = tf_c.paragraphs[0]
        p.text = ct["title"]
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = ct["color"]

        for b in ct["bullets"]:
            p_b = tf_c.add_paragraph()
            p_b.text = f"• {b}"
            p_b.font.size = Pt(8.0)
            p_b.font.color.rgb = WHITE
            p_b.space_after = Pt(2)

    # =========================================================================
    # SLIDE 4: FEASIBILITY AND VIABILITY
    # =========================================================================
    print("Formatting Slide 4 (Feasibility and Viability)...")
    s4 = prs.slides[3]

    t_shape = find_shape_by_text(s4, "FEASIBILITY AND VIABILITY")
    if t_shape and t_shape.has_text_frame:
        t_shape.text_frame.text = "FEASIBILITY, OPERATIONAL VIABILITY & RISK MITIGATION"
        for p in t_shape.text_frame.paragraphs:
            p.font.size = Pt(20)
            p.font.bold = True
            p.font.color.rgb = NAVY_DEEP

    tb_placeholder = find_shape_by_text(s4, "Analysis of the feasibility")
    if tb_placeholder:
        remove_shape(s4, tb_placeholder)

    # Center: SWOT & Risk Chevrons Diagram
    swot_img = ASSETS_DIR / "diagram_slide4_swot_risks.png"
    if swot_img.exists():
        s4.shapes.add_picture(str(swot_img), Inches(0.5), Inches(1.25), Inches(12.3), Inches(3.7))

    # Bottom Section: 3 Comprehensive Viability Pillars
    viab_cards = [
        {"title": "OPERATIONAL VIABILITY", "color": CYAN_ACCENT,
         "desc": "100% Web-based zero-install client running directly in standard browsers (Chrome/Edge). Field disaster teams need no desktop GIS software or manual tie-point matching."},
        {"title": "TECHNICAL ROBUSTNESS", "color": EMERALD,
         "desc": "Lightweight FP16 model consumes only 3.18 GB VRAM, operating smoothly on consumer laptops (RTX 4060) or CPU servers with automated relative fallback if metadata is absent."},
        {"title": "ECONOMIC SUSTAINABILITY", "color": AMBER,
         "desc": "Replaces $50,000+ airborne LiDAR surveys and multi-day satellite stereo tasks with instant single-pass optical imagery, built entirely on permissive open-source frameworks."},
    ]

    card_w3 = 3.95
    for idx3, vc in enumerate(viab_cards):
        c_left3 = Inches(0.5 + idx3 * 4.18)
        c_shape3 = add_card(s4, c_left3, Inches(5.1), Inches(card_w3), Inches(1.7), border_color=vc["color"])
        tf_vc = c_shape3.text_frame
        tf_vc.word_wrap = True
        tf_vc.margin_left = Inches(0.18)
        tf_vc.margin_top = Inches(0.12)

        p = tf_vc.paragraphs[0]
        p.text = vc["title"]
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = vc["color"]

        p_desc = tf_vc.add_paragraph()
        p_desc.text = vc["desc"]
        p_desc.font.size = Pt(8.2)
        p_desc.font.color.rgb = WHITE
        p_desc.space_before = Pt(3)

    # =========================================================================
    # SLIDE 5: IMPACT AND BENEFITS
    # =========================================================================
    print("Formatting Slide 5 (Impact and Benefits)...")
    s5 = prs.slides[4]

    t_shape = find_shape_by_text(s5, "IMPACT AND BENEFITS")
    if t_shape and t_shape.has_text_frame:
        t_shape.text_frame.text = "DISASTER RESPONSE ACCELERATION & GEOSPATIAL IMPACT ECOSYSTEM"
        for p in t_shape.text_frame.paragraphs:
            p.font.size = Pt(20)
            p.font.bold = True
            p.font.color.rgb = NAVY_DEEP

    tb_placeholder = find_shape_by_text(s5, "Potential impact on the target")
    if tb_placeholder:
        remove_shape(s5, tb_placeholder)

    # Center: 4-Quadrant Impact Ecosystem Hub Diagram
    ecosystem_img = ASSETS_DIR / "diagram_slide5_impact_ecosystem.png"
    if ecosystem_img.exists():
        s5.shapes.add_picture(str(ecosystem_img), Inches(0.5), Inches(1.25), Inches(12.3), Inches(3.7))

    # Bottom Section: 3 Transformative National & Strategic Pillars
    imp_cards = [
        {"title": "ATMANIRBHAR BHARAT (INDIGENOUS IP)", "color": CYAN_ACCENT,
         "desc": "Delivers an indigenous deep learning elevation extraction pipeline customized for ISRO satellite sensors (Cartosat, EOS series), reducing reliance on proprietary foreign photogrammetry suites."},
        {"title": "NDRF & CIVILIAN LIFE PROTECTION", "color": CORAL,
         "desc": "Sub-minute 3D elevation profiling during flash floods, landslides, and coastal cyclones enables disaster response commanders to simulate water levels and plan life-saving evacuation routes."},
        {"title": "SMART CITY & TERRAIN GOVERNANCE", "color": EMERALD,
         "desc": "Empowers municipal planners with automated building height compliance surveys, rooftop solar potential estimation, and stormwater drainage modeling without costly field visits."},
    ]

    for idx5, ic in enumerate(imp_cards):
        c_left5 = Inches(0.5 + idx5 * 4.18)
        c_shape5 = add_card(s5, c_left5, Inches(5.1), Inches(card_w3), Inches(1.7), border_color=ic["color"])
        tf_ic = c_shape5.text_frame
        tf_ic.word_wrap = True
        tf_ic.margin_left = Inches(0.18)
        tf_ic.margin_top = Inches(0.12)

        p = tf_ic.paragraphs[0]
        p.text = ic["title"]
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = ic["color"]

        p_desc = tf_ic.add_paragraph()
        p_desc.text = ic["desc"]
        p_desc.font.size = Pt(8.2)
        p_desc.font.color.rgb = WHITE
        p_desc.space_before = Pt(3)

    # =========================================================================
    # SLIDE 6: RESEARCH AND REFERENCES
    # =========================================================================
    print("Formatting Slide 6 (Research and References)...")
    s6 = prs.slides[5]

    t_shape = find_shape_by_text(s6, "RESEARCH  AND REFERENCES")
    if t_shape and t_shape.has_text_frame:
        t_shape.text_frame.text = "RESEARCH FOUNDATIONS, CITATIONS & GEOSPATIAL STANDARDS"
        for p in t_shape.text_frame.paragraphs:
            p.font.size = Pt(20)
            p.font.bold = True
            p.font.color.rgb = NAVY_DEEP

    tb_placeholder = find_shape_by_text(s6, "Details / Links of the reference")
    if tb_placeholder:
        remove_shape(s6, tb_placeholder)

    # Center: Research Lineage Matrix Diagram
    lineage_img = ASSETS_DIR / "diagram_slide6_research_lineage.png"
    if lineage_img.exists():
        s6.shapes.add_picture(str(lineage_img), Inches(0.5), Inches(1.25), Inches(12.3), Inches(3.7))

    # Bottom Section: Validation & ISRO Standards Card
    card_ref = add_card(s6, Inches(0.5), Inches(5.1), Inches(12.3), Inches(1.7), border_color=CYAN_ACCENT)
    tf_ref = card_ref.text_frame
    tf_ref.word_wrap = True
    tf_ref.margin_left = Inches(0.25)
    tf_ref.margin_top = Inches(0.12)

    p_rt = tf_ref.paragraphs[0]
    p_rt.text = "EMPIRICAL RESEARCH RIGOR & ISRO SAC BENCHMARK COMPLIANCE"
    p_rt.font.bold = True
    p_rt.font.size = Pt(10)
    p_rt.font.color.rgb = CYAN_ACCENT

    format_bullet(tf_ref.add_paragraph(), "Zero-Shot Foundation AI", "Built upon DPT & Depth Anything V2 (Yang et al., 2024), utilizing pre-trained ViT feature hierarchies for robust monocular structural generalization.", font_size=8.2)
    format_bullet(tf_ref.add_paragraph(), "ISRO Cartosat Standards", "Benchmarked against ISRO Space Applications Centre Cartosat DEM guidelines for vertical accuracy tolerances in urban and hilly terrain.", font_size=8.2)
    format_bullet(tf_ref.add_paragraph(), "Open Geospatial Interoperability", "Strict compliance with OGC Cloud-Optimized GeoTIFF standard ensures native drag-and-drop loading into ArcGIS, QGIS, and ISRO Bhuvan geoportal.", font_size=8.2)

    # =========================================================================
    # DELETE SLIDE 7 (INSTRUCTIONS SLIDE)
    # =========================================================================
    if len(prs.slides) > 6:
        print("Removing instruction slide 7 to maintain strict 6-slide SIH limit...")
        rId = prs.slides._sldIdLst[6].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[6]

    # Save final presentation
    print(f"Saving final 6-slide presentation to: {OUTPUT_PATH}")
    prs.save(str(OUTPUT_PATH))
    print("Presentation generation completed successfully!")


if __name__ == "__main__":
    build_presentation()
