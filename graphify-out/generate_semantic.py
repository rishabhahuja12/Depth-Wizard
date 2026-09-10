import json
from pathlib import Path

semantic_nodes = [
    {
        "id": "concept_gamus_dataset",
        "label": "earthflow/GAMUS Dataset (DFC2019)",
        "file_type": "document",
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "concept_silog_loss",
        "label": "Scale-Invariant Logarithmic (SILog) Loss",
        "file_type": "document",
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "uvp_contour_lines",
        "label": "UVP: Topographic Contour Isolines",
        "file_type": "document",
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "uvp_flood_simulator",
        "label": "UVP: Flood Inundation Simulator",
        "file_type": "document",
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "uvp_cross_section",
        "label": "UVP: Elevation Cross-Section Profiler",
        "file_type": "document",
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "uvp_geotiff_export",
        "label": "UVP: OGC Cloud-Optimized GeoTIFF Export",
        "file_type": "document",
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "uvp_fpv_tpp_drone_flight",
        "label": "UVP: FPV & TPP Drone Flight Simulator with Proximity Ray-Marching & Telemetry",
        "file_type": "document",
        "source_file": "reports/drone_and_navigation_mechanics_report.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "uvp_voxel_terrain_mode",
        "label": "UVP: Voxelized Digital Surface Model (DSM) Quantization & Instanced Rendering",
        "file_type": "document",
        "source_file": "reports/mesh_construction_and_project_architecture.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "uvp_terrain_quick_panel",
        "label": "UVP: Real-Time In-Flight Terrain & Voxel Shading Quick Panel",
        "file_type": "code",
        "source_file": "frontend/src/components/TerrainQuickPanel.tsx",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "doc_readme_depthwizard",
        "label": "DepthWizard System Overview README.md",
        "file_type": "document",
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "doc_walkthrough_report",
        "label": "SIH26175 Verification Walkthrough",
        "file_type": "document",
        "source_file": "WALKTHROUGH.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "doc_drone_mechanics_report",
        "label": "Drone & Navigation Flight Mechanics Report",
        "file_type": "document",
        "source_file": "reports/drone_and_navigation_mechanics_report.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "doc_mesh_architecture_report",
        "label": "3D Mesh Construction & Project Architecture Report",
        "file_type": "document",
        "source_file": "reports/mesh_construction_and_project_architecture.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "doc_project_brief",
        "label": "DepthWizard Multi-Modal Geospatial Brief",
        "file_type": "document",
        "source_file": "reports/project_brief.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "id": "img_sample_aerial",
        "label": "Sample Aerial Overhead Imagery",
        "file_type": "image",
        "source_file": "benchmark/sample_aerial.png",
        "source_location": None,
        "weight": 0.85
    }
]

semantic_edges = [
    {
        "source": "concept_gamus_dataset",
        "target": "backend_training_dataset_gamus",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "concept_silog_loss",
        "target": "backend_training_losses",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_contour_lines",
        "target": "frontend_src_components_contouroverlay",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_cross_section",
        "target": "frontend_src_components_crosssection",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_flood_simulator",
        "target": "frontend_src_components_floodsimulator",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_geotiff_export",
        "target": "backend_app_api_routes",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_fpv_tpp_drone_flight",
        "target": "frontend_src_components_drone_droneflightcontroller",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/drone_and_navigation_mechanics_report.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_fpv_tpp_drone_flight",
        "target": "frontend_src_components_drone_dronecanvas",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/drone_and_navigation_mechanics_report.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_fpv_tpp_drone_flight",
        "target": "frontend_src_components_drone_dronehud",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/drone_and_navigation_mechanics_report.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_fpv_tpp_drone_flight",
        "target": "frontend_src_components_drone_proximitysensors",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/drone_and_navigation_mechanics_report.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_fpv_tpp_drone_flight",
        "target": "frontend_src_components_drone_dsmsampling",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/drone_and_navigation_mechanics_report.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_voxel_terrain_mode",
        "target": "frontend_src_components_voxelterrain",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/mesh_construction_and_project_architecture.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_voxel_terrain_mode",
        "target": "frontend_src_lib_voxelize",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/mesh_construction_and_project_architecture.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_terrain_quick_panel",
        "target": "frontend_src_components_terrainquickpanel",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "frontend/src/components/TerrainQuickPanel.tsx",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "uvp_terrain_quick_panel",
        "target": "frontend_src_hooks_useterrainsettings",
        "relation": "implements",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "frontend/src/components/TerrainQuickPanel.tsx",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_readme_depthwizard",
        "target": "concept_gamus_dataset",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_readme_depthwizard",
        "target": "concept_silog_loss",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_readme_depthwizard",
        "target": "uvp_contour_lines",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_readme_depthwizard",
        "target": "uvp_cross_section",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_readme_depthwizard",
        "target": "uvp_flood_simulator",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_readme_depthwizard",
        "target": "uvp_geotiff_export",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_readme_depthwizard",
        "target": "uvp_fpv_tpp_drone_flight",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_readme_depthwizard",
        "target": "uvp_voxel_terrain_mode",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "README.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_drone_mechanics_report",
        "target": "uvp_fpv_tpp_drone_flight",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/drone_and_navigation_mechanics_report.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_mesh_architecture_report",
        "target": "uvp_voxel_terrain_mode",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/mesh_construction_and_project_architecture.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_project_brief",
        "target": "doc_readme_depthwizard",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "reports/project_brief.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "doc_walkthrough_report",
        "target": "doc_readme_depthwizard",
        "relation": "references",
        "confidence": "EXTRACTED",
        "confidence_score": 1.0,
        "source_file": "WALKTHROUGH.md",
        "source_location": None,
        "weight": 1.0
    },
    {
        "source": "img_sample_aerial",
        "target": "backend_app_api_routes",
        "relation": "conceptually_related_to",
        "confidence": "INFERRED",
        "confidence_score": 0.85,
        "source_file": "benchmark/sample_aerial.png",
        "source_location": None,
        "weight": 0.85
    },
    {
        "source": "frontend_src_components_terraincanvas",
        "target": "uvp_flood_simulator",
        "relation": "shares_data_with",
        "confidence": "INFERRED",
        "confidence_score": 0.9,
        "source_file": "frontend/src/components/TerrainCanvas.tsx",
        "source_location": None,
        "weight": 0.9
    }
]

hyperedges = [
    {
        "id": "disaster_management_uvp_suite",
        "label": "Disaster Management & Topography UVP Suite",
        "nodes": [
            "uvp_flood_simulator",
            "uvp_cross_section",
            "uvp_contour_lines",
            "uvp_geotiff_export"
        ],
        "relation": "form",
        "confidence": "INFERRED",
        "confidence_score": 0.95,
        "source_file": "README.md"
    },
    {
        "id": "drone_simulation_and_voxel_suite",
        "label": "FPV/TPP Drone Flight Simulation & Voxel Terrain Suite",
        "nodes": [
            "uvp_fpv_tpp_drone_flight",
            "uvp_voxel_terrain_mode",
            "uvp_terrain_quick_panel"
        ],
        "relation": "form",
        "confidence": "INFERRED",
        "confidence_score": 0.95,
        "source_file": "reports/drone_and_navigation_mechanics_report.md"
    }
]

sem_data = {
    "nodes": semantic_nodes,
    "edges": semantic_edges,
    "hyperedges": hyperedges,
    "input_tokens": 15000,
    "output_tokens": 3500
}

Path('graphify-out/.graphify_semantic.json').write_text(json.dumps(sem_data, indent=2))
print(f"Generated semantic data: {len(semantic_nodes)} nodes, {len(semantic_edges)} edges, {len(hyperedges)} hyperedges")
