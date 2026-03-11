import pandas as pd
import glob
import os
import re

folder_path = r"T:\Orthopaedics\Lab Imaging Data\mmazur\NT_Validation_Study4\Rb45\Rb45_Left_A\40X_Morphometrics"

# Grab files, ignore temporary Excel files
files = [
    f for f in glob.glob(os.path.join(folder_path, "*.xlsx"))
    if not os.path.basename(f).startswith("~$")
]

# Function to extract number from filename for numeric sorting
def extract_number(f):
    # Matches the first sequence of digits after "40X_"
    match = re.search(r'40X_(\d+)_seg', os.path.basename(f))
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # Put files without a match at the end

# Sort files numerically
files = sorted(files, key=extract_number)

results = []

for file in files:
    df = pd.read_excel(file, usecols="N")
    df.columns = ["g_ratio"]
    df["g_ratio"] = pd.to_numeric(df["g_ratio"], errors="coerce")
    df = df.dropna()

    axon_count = len(df)
    mean_g = df["g_ratio"].mean()

    results.append({
        "Image_File": os.path.basename(file),
        "Axon_Count": axon_count,
        "Mean_g_ratio": mean_g
    })

summary_df = pd.DataFrame(results)
output_path = os.path.join(folder_path, "compiled_summary.xlsx")
summary_df.to_excel(output_path, index=False)

print("Summary file saved to:")
print(output_path)