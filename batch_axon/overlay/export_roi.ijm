setBatchMode(true);
print("--- Fiji Macro Started in Batch Mode ---");

arg = getArgument();
args = split(arg, ",");
inputImagePath = args[0];
outputRoiPath = args[1];
print("  - Input Image: " + inputImagePath);
print("  - Output ROI File: " + outputRoiPath);

print("Opening image and converting overlay...");
open(inputImagePath);
if (nImages() == 0) { exit("FATAL: Image failed to open."); }

width = getWidth();
height = getHeight();

newWidth = floor(width / 256) * 256;
newHeight = floor(height / 256) * 256;

xStart = (width - newWidth) / 2;
yStart = (height - newHeight) / 2;

makeRectangle(xStart, yStart, newWidth, newHeight);
run("Crop");

run("To ROI Manager");
count = roiManager("count");
print("Found " + count + " ROIs.");

if (count == 0) { exit("FATAL: No outline was found."); }

// print("Selecting first ROI and saving...");
// roiManager("select", 0);
// roiManager("save", outputRoiPath);

if (count > 1) {
    print("Combining all ROIs into a single ROI...");
    indices = newArray(count);
    for (i = 0; i < count; i++) {
        indices[i] = i;
    }
    roiManager("select", indices);
    roiManager("Combine");
    roiManager("reset");
    roiManager("add"); // add combined ROI
} else {
    print("Only one ROI found, skipping combine...");
    // leave it as-is in ROI Manager
}

print("Saving combined ROI...");
roiManager("select", 0);
roiManager("save", outputRoiPath);

if (File.exists(outputRoiPath)) {
    print("SUCCESS: Single .roi file created.");
} else {
    print("FATAL: Failed to save .roi file.");
}
print("--- Fiji Macro Finished ---");