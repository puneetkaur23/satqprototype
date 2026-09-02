# download_samples.py
import os
import urllib.request

os.makedirs("samples", exist_ok=True)

# Direct links to free Sentinel-2 RGB composites from ESA
samples = {
    "optical_1.jpg": "https://dataspace.copernicus.eu/exodata/S2MSI2A/2023/01/01/S2A_T31TFJ_20230101T104431_L2A_TCI_10m.jpg",
    "optical_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Sentinel-2_L1C_True_color_image_of_the_Nile_Delta.jpg/800px-Sentinel-2_L1C_True_color_image_of_the_Nile_Delta.jpg",
    "sar_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Sentinel-1_SAR_image_of_the_Netherlands.jpg/800px-Sentinel-1_SAR_image_of_the_Netherlands.jpg"
}

print("📡 Downloading sample satellite imagery...")
for filename, url in samples.items():
    filepath = os.path.join("samples", filename)
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  ✅ {filename}")
    except Exception as e:
        print(f"  ❌ {filename}: {e}")

print("\n🎯 Done! Images saved in ./samples/")
print("💡 Pro tip: If downloads fail, just save any JPG from Google Images search 'Sentinel-2 RGB composite' into the samples/ folder.")
