from huggingface_hub import snapshot_download

print("Starting GAMUS TRAIN download (RGB images)...")
print("Destination: .\\GAMUS")
print("(Heights are already 100% complete at heights\\train)")
print()

snapshot_download(
    repo_id="earthflow/GAMUS",
    repo_type="dataset",
    local_dir="./GAMUS",
    allow_patterns=[
        "images/train/**",
    ],
    max_workers=4
)

print()
print("===================================")
print("GAMUS TRAIN RGB DOWNLOAD COMPLETE")
print("Location: .\\GAMUS")
print("===================================")