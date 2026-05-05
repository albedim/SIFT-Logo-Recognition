"""
Logo Recognition using SIFT (Scale-Invariant Feature Transform)
Usage:
    python logo_recognition.py --reference logo.png --query image.png
    python logo_recognition.py --reference logo.png --query image.png --threshold 20
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: '{path}'")
    return img


def recognize_logo(
    reference_path: str,
    query_path: str,
    min_match_count: int = 10,
    ratio_thresh: float = 0.75,
    detection_threshold: int = 20,
) -> dict:
    """
    Compare a reference logo against a query image using SIFT + FLANN.

    Parameters
    ----------
    reference_path      : path to the clean logo (template)
    query_path          : path to the image to inspect
    min_match_count     : minimum good matches to attempt homography
    ratio_thresh        : Lowe's ratio-test threshold (0.6–0.8 typical)
    detection_threshold : minimum good matches to declare the logo present

    Returns
    -------
    dict with keys:
        logo_found  (bool)
        similarity  (float, 0–100)
        good_matches (int)
        total_keypoints_ref (int)
        message     (str)
    """

    ref = load_image(reference_path)
    query = load_image(query_path)

    sift = cv2.SIFT_create()

    kp_ref, des_ref = sift.detectAndCompute(ref, None)
    kp_qry, des_qry = sift.detectAndCompute(query, None)

    if des_ref is None or len(kp_ref) == 0:
        return {
            "logo_found": False,
            "similarity": 0.0,
            "good_matches": 0,
            "total_keypoints_ref": 0,
            "message": "❌ Reference logo has no detectable keypoints.",
        }

    if des_qry is None or len(kp_qry) == 0:
        return {
            "logo_found": False,
            "similarity": 0.0,
            "good_matches": 0,
            "total_keypoints_ref": len(kp_ref),
            "message": "❌ Query image has no detectable keypoints.",
        }

    index_params = dict(algorithm=1, trees=5)   # KD-Tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_ref, des_qry, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    total_ref_kp = len(kp_ref)
    good_count   = len(good)

    raw_ratio  = good_count / total_ref_kp if total_ref_kp > 0 else 0.0
    base_score = np.sqrt(raw_ratio)

    geometry_ok  = False
    inlier_bonus = 0.0
    if good_count >= min_match_count:
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_qry[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers      = int(mask.ravel().sum())
            inlier_ratio = inliers / good_count if good_count > 0 else 0.0
            inlier_bonus = inlier_ratio
            geometry_ok  = inliers >= min_match_count

    combined = base_score * 0.4 + inlier_bonus * 0.6

    similarity = min(100.0, round(combined * 130, 2))

    logo_found = good_count >= detection_threshold

    if logo_found:
        if similarity >= 60:
            verdict = f"✅ Logo FOUND — high confidence ({similarity:.1f}% similarity)"
        elif similarity >= 35:
            verdict = f"⚠️  Logo FOUND — moderate confidence ({similarity:.1f}% similarity)"
        else:
            verdict = f"⚠️  Logo possibly present — low confidence ({similarity:.1f}% similarity)"
    else:
        similarity = min(similarity, 15.0)
        verdict = (
            f"❌ Logo NOT found in the query image "
            f"(only {good_count} good matches, need ≥ {detection_threshold}; "
            f"similarity ≈ {similarity:.1f}%)"
        )

    return {
        "logo_found":           logo_found,
        "similarity":           similarity,
        "good_matches":         good_count,
        "total_keypoints_ref":  total_ref_kp,
        "geometry_verified":    geometry_ok,
        "message":              verdict,
    }


def visualize_matches(reference_path: str, query_path: str, output_path: str = "matches.png"):
    ref   = cv2.imread(reference_path)
    query = cv2.imread(query_path)
    gray_ref   = cv2.cvtColor(ref,   cv2.COLOR_BGR2GRAY)
    gray_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_ref,  des_ref  = sift.detectAndCompute(gray_ref,   None)
    kp_qry,  des_qry  = sift.detectAndCompute(gray_query, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des_ref, des_qry, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    vis = cv2.drawMatches(ref, kp_ref, query, kp_qry, good[:50], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(output_path, vis)
    print(f"Match visualisation saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SIFT-based logo recognition — returns similarity %"
    )
    parser.add_argument("--reference",  required=True, help="Path to reference logo image")
    parser.add_argument("--query",      required=True, help="Path to query/scene image")
    parser.add_argument("--threshold",  type=int, default=20,
                        help="Min good matches to declare logo present (default: 20)")
    parser.add_argument("--ratio",      type=float, default=0.75,
                        help="Lowe's ratio-test threshold (default: 0.75)")
    parser.add_argument("--visualize",  action="store_true",
                        help="Save a match visualisation image")
    parser.add_argument("--output",     default="matches.png",
                        help="Output path for visualisation (default: matches.png)")
    args = parser.parse_args()

    # Validate paths
    for label, path in [("Reference", args.reference), ("Query", args.query)]:
        if not Path(path).exists():
            print(f"[ERROR] {label} file not found: '{path}'")
            sys.exit(1)

    print(f"\n{'─'*55}")
    print(f"  Reference logo : {args.reference}")
    print(f"  Query image    : {args.query}")
    print(f"  Threshold      : {args.threshold} matches")
    print(f"{'─'*55}")

    result = recognize_logo(
        reference_path=args.reference,
        query_path=args.query,
        detection_threshold=args.threshold,
        ratio_thresh=args.ratio,
    )

    print(f"\n  {result['message']}")
    print(f"\n  Details:")
    print(f"    • Good matches          : {result['good_matches']}")
    print(f"    • Ref keypoints         : {result['total_keypoints_ref']}")
    print(f"    • Geometry verified     : {result.get('geometry_verified', 'N/A')}")
    print(f"    • Similarity score      : {result['similarity']:.2f}%")
    print(f"{'─'*55}\n")

    if args.visualize:
        visualize_matches(args.reference, args.query, args.output)

    return 0 if result["logo_found"] else 1


if __name__ == "__main__":
    sys.exit(main())
