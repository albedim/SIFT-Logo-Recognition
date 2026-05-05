# 🧠 Logo Recognition using SIFT

A Python computer vision tool that detects whether a reference logo
appears inside a query image using **SIFT (Scale-Invariant Feature
Transform)** and **FLANN matching**.

------------------------------------------------------------------------

## 🚀 Features

-   Detects logos even with scale, rotation, and perspective changes
-   Uses SIFT keypoints + FLANN matcher
-   Computes a **similarity score (0--100%)**
-   Optional geometric verification using homography (RANSAC)
-   Visualization of matched features
-   CLI-based tool for easy automation

------------------------------------------------------------------------

## 📦 Requirements

Install dependencies:

``` bash
pip install requirements.txt or pip install opencv-python numpy
```

------------------------------------------------------------------------

## ▶️ Usage

### Basic detection

``` bash
python logo_recognition.py --reference logo.png --query image.png
```

------------------------------------------------------------------------

### Adjust sensitivity

``` bash
python logo_recognition.py --reference logo.png --query image.png --threshold 20
```

------------------------------------------------------------------------

### Change matching strictness

``` bash
python logo_recognition.py --reference logo.png --query image.png --ratio 0.75
```

------------------------------------------------------------------------

### Generate visualization

``` bash
python logo_recognition.py --reference logo.png --query image.png --visualize
```

Output:

    matches.png

------------------------------------------------------------------------

## ⚙️ Arguments

  Argument        Description
  --------------- -------------------------------------
  `--reference`   Reference logo image
  `--query`       Image to search in
  `--threshold`   Minimum good matches to detect logo
  `--ratio`       Lowe's ratio test threshold
  `--visualize`   Save match visualization
  `--output`      Output file for visualization

------------------------------------------------------------------------

## 🧠 How it works

1.  Extract SIFT keypoints from both images\
2.  Match descriptors using FLANN (KD-Tree)\
3.  Apply Lowe's ratio test to filter bad matches\
4.  Estimate similarity score\
5.  Optionally validate geometry using homography (RANSAC)

------------------------------------------------------------------------

## 📊 Output Example

    Logo FOUND — high confidence (78.4% similarity)

    Details:
    • Good matches: 42
    • Ref keypoints: 120
    • Geometry verified: True
    • Similarity score: 78.40%

------------------------------------------------------------------------

## 📁 Output files

-   `matches.png` → visual match preview (optional)

------------------------------------------------------------------------

## ⚠️ Notes

-   Works best with clean, high-resolution reference logos
-   Performance depends on image quality and lighting
-   SIFT is CPU-intensive but very robust

------------------------------------------------------------------------

## 🧩 Use cases

-   Brand/logo detection in images
-   Screenshot analysis
-   Content moderation
-   Visual search prototypes
