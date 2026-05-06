# 🧠 Logo Recognition using SIFT

A Python computer vision tool that detects whether a reference logo
appears inside a query image using **SIFT (Scale-Invariant Feature
Transform)** and **FLANN matching**.
It works by extracting SIFT keypoints from both images and matching their descriptors using a FLANN-based (KD-Tree) approach, applying Lowe’s ratio test to filter out poor matches, and computing a similarity score (0–100%). The system is robust to scale, rotation, and perspective changes, and can optionally validate geometric consistency using homography with RANSAC. It also provides visualization of matched features and is implemented as a CLI-based tool for easy automation..

## Requirements

Install dependencies:

``` bash
pip install requirements.txt
```
OR
``` bash
pip install opencv-python numpy
```

------------------------------------------------------------------------

## Usage

### Basic detection

``` bash
python logo_recognition.py --reference logo.png --query image.png
```

### Adjust sensitivity

``` bash
python logo_recognition.py --reference logo.png --query image.png --threshold 20
```

### Change matching strictness

``` bash
python logo_recognition.py --reference logo.png --query image.png --ratio 0.75
```

### Generate visualization

``` bash
python logo_recognition.py --reference logo.png --query image.png --visualize
```

------------------------------------------------------------------------

## Arguments

  Argument        Description
  --------------- -------------------------------------
  `--reference`   Reference logo image
  `--query`       Image to search in
  `--threshold`   Minimum good matches to detect logo
  `--ratio`       Lowe's ratio test threshold
  `--visualize`   Save match visualization
  `--output`      Output file for visualization

------------------------------------------------------------------------

## Output Example

    Logo FOUND — high confidence (78.4% similarity)

    Details:
    • Good matches: 42
    • Ref keypoints: 120
    • Geometry verified: True
    • Similarity score: 78.40%

------------------------------------------------------------------------

## Use cases

-   Brand/logo detection in images
-   Screenshot analysis
-   Content moderation
-   Visual search prototypes
