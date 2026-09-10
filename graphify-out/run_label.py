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
    0: "Frontend React UI & Flythrough Shell",
    1: "GAMUS Training & Fine-Tuning Pipeline",
    2: "Frontend Project Configuration",
    3: "FastAPI Routes & GeoTIFF Export",
    4: "TypeScript Compiler Options",
    5: "Heightfield Mesh Builder & GPU Displacement",
    6: "Build Tools & Platform Binaries",
    7: "Geospatial Reader & Metadata Parser",
    8: "Two-Component Elevation Calibration",
    9: "UI Libraries & Geospatial Dependencies",
    10: "Depth Anything V2 Inference Engine",
    11: "CUDA Hardware Verification Suite",
    12: "Synthetic Aerial Benchmark Suite",
    13: "Turbo Colormap Palette Shaders",
    14: "API Package Namespace",
    15: "Services Package Namespace",
    16: "Training Package Namespace",
    17: "App Package Namespace",
}

questions = suggest_questions(G, communities, labels)
report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, 'DepthWizard', suggested_questions=questions)

Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding='utf-8')
Path('graphify-out/.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}, indent=2), encoding='utf-8')
to_json(G, communities, 'graphify-out/graph.json')

# Export interactive standalone HTML visualization
to_html(G, communities, 'graphify-out/graph.html', community_labels=labels or None)
print('SUCCESS: GRAPH_REPORT.md, graph.json, and graph.html generated!')
