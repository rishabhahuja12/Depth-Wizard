import json
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_html, to_json

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text(encoding='utf-8'))
detection  = json.loads(Path('graphify-out/.graphify_detect.json').read_text(encoding='utf-8'))
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text(encoding='utf-8'))

G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
tokens = {'input': extraction.get('input_tokens', 0), 'output': extraction.get('output_tokens', 0)}

labels = {
    0: "Frontend React UI & Geospatial Analysis Tools",
    1: "Backend Elevation Calibration, Geospatial Parser & Validation",
    2: "GAMUS Deep Learning Training & Losses",
    3: "FPV & TPP Drone Flight Simulator, Proximity Ray-Marching & HUD",
    4: "Frontend Build System & Dev Dependencies",
    5: "FastAPI REST Endpoints & GeoTIFF Export Services",
    6: "Inference Engine, Benchmark Validation & Configuration",
    7: "Voxel Terrain Quantization & Instanced Mesh Rendering",
    8: "TypeScript Compiler Configuration",
    9: "Presentation Diagram Generation",
    10: "SIH 2026 Presentation Builder",
    11: "Frontend Runtime Dependencies & 3D Engine",
    12: "CUDA Hardware Verification Suite",
    13: "Synthetic Aerial Benchmark Scene Generator",
    14: "Turbo Colormap Palette Shaders",
    15: "Backend Production Server Launcher",
    16: "Automated Browser Testing Harness",
    17: "API Package Namespace",
    18: "App Package Namespace",
    19: "Services Package Namespace",
    20: "Training Package Namespace",
    21: "Presentation Template Inspector",
    22: "Presentation Output Verification",
}

questions = suggest_questions(G, communities, labels)
report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, 'DepthWizard', suggested_questions=questions)

Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding='utf-8')
Path('graphify-out/.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}, indent=2), encoding='utf-8')
to_json(G, communities, 'graphify-out/graph.json')

# Export interactive standalone HTML visualization
to_html(G, communities, 'graphify-out/graph.html', community_labels=labels or None)
print('SUCCESS: GRAPH_REPORT.md, graph.json, and graph.html generated!')
