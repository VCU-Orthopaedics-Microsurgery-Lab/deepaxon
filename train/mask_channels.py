import numpy as np, cv2, os

mask_dir = r"T:\Orthopaedics\Lab Imaging Data\mmazur\DA_Model_Training\100X\100X Model Training\Selected Images\trainingv3a\masks\cropped\patches"

if not os.path.exists(mask_dir):
    print("❌ Path not found:", mask_dir)
else:
    files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    print(f"✅ Found {len(files)} PNG files in {mask_dir}")

    vals = set()
    for f in files:
        vals |= set(np.unique(cv2.imread(os.path.join(mask_dir, f), 0)))
    print("🧩 Unique pixel values:", sorted(vals))