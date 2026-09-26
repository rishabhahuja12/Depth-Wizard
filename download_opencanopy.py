"""
Download Open-Canopy dataset slice into .\\Open-Canopy.

HF Repo: AI4Forest/Open-Canopy
Note: The full Open-Canopy dataset is ~360 GB.
This script downloads a manageable slice (e.g. 2021) or full train data.
Expected layout for Depth-Wizard:
  .\\Open-Canopy\\images\\*.tif
  .\\Open-Canopy\\canopy_height\\*.tif
"""
from huggingface_hub import snapshot_download

print("Starting Open-Canopy download...")
print("Destination: .\\Open-Canopy")
print("Repository: AI4Forest/Open-Canopy")
print()

# You can adjust allow_patterns to limit download size if needed (e.g. specific years)
snapshot_download(
    repo_id="AI4Forest/Open-Canopy",
    repo_type="dataset",
    local_dir="./Open-Canopy",
    max_workers=4
)

print()
print("===================================")
print("OPEN-CANOPY DOWNLOAD COMPLETE")
print("Location: .\\Open-Canopy")
print("===================================")
