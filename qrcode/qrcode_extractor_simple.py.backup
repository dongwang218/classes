#!/usr/bin/env python3
"""
Simplified QR Code Corner Extractor

Detects QR code corners using finder pattern detection only.
Uses OpenCV and NumPy - no complex dependencies.

Usage:
    python qrcode_extractor_simple.py <image_path>
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys


def find_finder_patterns(gray: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Find QR code finder patterns (the three large nested squares at corners).
    Returns list of (center_x, center_y, size) for each detected pattern.
    """
    h, w = gray.shape

    # Apply adaptive threshold to handle varying lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 51, 10)

    # Find contours with hierarchy (needed to detect nested structures)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    candidates = []

    # Look for nested squares (characteristic of finder patterns)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 100 or area > (h * w * 0.15):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # Check squareness
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        # Check for nested structure (finder pattern has 3 nested squares)
        child = hierarchy[i][2]
        depth = 0
        idx = i
        while hierarchy[idx][3] >= 0:
            depth += 1
            idx = hierarchy[idx][3]

        # Middle ring should have depth >= 1 and have children
        if depth >= 1 and child >= 0:
            child_area = cv2.contourArea(contours[child])
            if child_area > 0:
                ratio = area / child_area
                if 1.5 < ratio < 10:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Use bounding rect for accurate size
                        x, y, bw, bh = cv2.boundingRect(contour)
                        size = max(bw, bh)
                        candidates.append((cx, cy, size, area))

    if len(candidates) < 2:
        return [(c[0], c[1], c[2]) for c in candidates]

    # Cluster nearby candidates (multiple detections of same pattern)
    candidates.sort(key=lambda x: x[3], reverse=True)
    clusters = []
    used = set()

    for i, (x1, y1, s1, a1) in enumerate(candidates):
        if i in used:
            continue
        cluster = [(x1, y1, s1, a1)]
        used.add(i)

        for j, (x2, y2, s2, a2) in enumerate(candidates):
            if j in used:
                continue
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < max(s1, s2) * 1.5:
                cluster.append((x2, y2, s2, a2))
                used.add(j)

        largest = max(cluster, key=lambda x: x[3])
        clusters.append((largest[0], largest[1], largest[2], largest[3]))

    if len(clusters) < 3:
        return [(c[0], c[1], c[2]) for c in clusters]

    # Select best 3 patterns that form a right-angle triangle at corners
    from itertools import combinations

    all_pts = np.array([(c[0], c[1]) for c in clusters], dtype=np.float32)
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    bb_corners = np.array([[min_x, min_y], [max_x, min_y],
                           [min_x, max_y], [max_x, max_y]])

    # Distance to nearest bounding box corner for each cluster
    corner_distances = []
    for c in clusters:
        pt = np.array([c[0], c[1]])
        min_dist = min(np.linalg.norm(pt - corner) for corner in bb_corners)
        corner_distances.append(min_dist)

    best_triplet = None
    best_score = float('inf')

    for triplet in combinations(range(min(10, len(clusters))), 3):
        pts = np.array([(clusters[i][0], clusters[i][1]) for i in triplet], dtype=np.float32)
        sizes = [clusters[i][2] for i in triplet]

        # Check that patterns are at 3 different corners
        closest_corners = []
        for i in triplet:
            pt = np.array([clusters[i][0], clusters[i][1]])
            dists = [np.linalg.norm(pt - corner) for corner in bb_corners]
            closest_corners.append(np.argmin(dists))

        if len(set(closest_corners)) < 3:
            continue

        # Check right-angle triangle geometry
        d01 = np.linalg.norm(pts[0] - pts[1])
        d02 = np.linalg.norm(pts[0] - pts[2])
        d12 = np.linalg.norm(pts[1] - pts[2])
        dists = sorted([d01, d02, d12])

        if dists[0] < 10:
            continue

        ratio1 = dists[1] / dists[0]
        ratio2 = dists[2] / dists[0]
        geometry_score = abs(ratio1 - 1.0) + abs(ratio2 - 1.414)

        # Penalize patterns far from corners
        corner_penalty = sum(corner_distances[i] for i in triplet) / (max(w, h) + 1) * 0.5

        # Penalize small patterns (likely alignment patterns, not finders)
        avg_size = np.mean(sizes)
        min_size = min(sizes)
        max_size = clusters[0][2]
        size_penalty = (max_size - avg_size) / max_size * 0.5
        if min_size < max_size * 0.6:
            size_penalty += 0.5

        total_score = geometry_score + corner_penalty + size_penalty

        if total_score < best_score:
            best_score = total_score
            best_triplet = tuple(clusters[i] for i in triplet)

    if best_triplet and best_score < 3.0:
        return [(t[0], t[1], t[2]) for t in best_triplet]

    return [(c[0], c[1], c[2]) for c in clusters[:3]]


def compute_corners_from_finders(finders: List[Tuple[int, int, int]],
                                  img_shape: Tuple[int, int],
                                  gray: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Compute QR code corners from finder pattern positions.
    Works with 2 or 3 finder patterns.
    Returns corners ordered as: top-left, top-right, bottom-right, bottom-left
    """
    if len(finders) < 2:
        return None

    h, w = img_shape
    pts = np.array([(f[0], f[1]) for f in finders], dtype=np.float32)
    sizes = [f[2] for f in finders]
    avg_size = np.mean(sizes)

    if len(finders) == 2:
        # Infer corners assuming 2 finders are adjacent (TL-TR or TL-BL)
        p1, p2 = pts[0], pts[1]
        v = p2 - p1

        if abs(v[0]) > abs(v[1]):
            # Horizontal - TL and TR
            if p1[0] < p2[0]:
                tl, tr = p1, p2
            else:
                tl, tr = p2, p1
            v_down = np.array([-v[1], v[0]]) if v[0] > 0 else np.array([v[1], -v[0]])
            if v_down[1] < 0:
                v_down = -v_down
            bl = tl + v_down
            br = tr + v_down
        else:
            # Vertical - TL and BL
            if p1[1] < p2[1]:
                tl, bl = p1, p2
            else:
                tl, bl = p2, p1
            v_right = np.array([v[1], -v[0]]) if v[1] > 0 else np.array([-v[1], v[0]])
            if v_right[0] < 0:
                v_right = -v_right
            tr = tl + v_right
            br = bl + v_right

        all_corners = np.array([tl, tr, br, bl], dtype=np.float32)
        fourth_corner_local_idx = 2  # br is at index 2

    else:
        # 3 finders - find which one is at the right-angle corner
        best_corner_idx = 0
        best_angle_diff = float('inf')

        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angle_diff = abs(angle - np.pi/2)

            if angle_diff < best_angle_diff:
                best_angle_diff = angle_diff
                best_corner_idx = i

        corner_pt = pts[best_corner_idx]
        pt1 = pts[(best_corner_idx + 1) % 3]
        pt2 = pts[(best_corner_idx + 2) % 3]

        # 4th corner via parallelogram
        fourth_corner = pt1 + pt2 - corner_pt

        all_corners = np.array([corner_pt, pt1, pt2, fourth_corner], dtype=np.float32)
        fourth_corner_local_idx = 3  # Track which corner has no finder pattern

    # Expand corners outward to include full finder patterns
    expand_size = avg_size * 0.6
    center = np.mean(all_corners, axis=0)
    for i in range(4):
        direction = all_corners[i] - center
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        # For the 4th corner (no finder pattern), expand less
        if len(finders) >= 3 and i == fourth_corner_local_idx:
            # Reduce expansion for the corner without finder pattern
            all_corners[i] = all_corners[i] + direction * expand_size * 0.3
        else:
            all_corners[i] = all_corners[i] + direction * expand_size

    # Clamp to image boundaries
    all_corners[:, 0] = np.clip(all_corners[:, 0], 2, w - 2)
    all_corners[:, 1] = np.clip(all_corners[:, 1], 2, h - 2)

    # Order corners: top-left, top-right, bottom-right, bottom-left
    center = np.mean(all_corners, axis=0)
    angles = np.arctan2(all_corners[:, 1] - center[1], all_corners[:, 0] - center[0])
    order = np.argsort(angles)
    sorted_corners = all_corners[order]

    # Find top-left (smallest x+y sum)
    sums = sorted_corners[:, 0] + sorted_corners[:, 1]
    start_idx = np.argmin(sums)
    sorted_corners = np.roll(sorted_corners, -start_idx, axis=0)

    return sorted_corners


def perspective_transform(img: np.ndarray, corners: np.ndarray, size: int = 400) -> np.ndarray:
    """Apply perspective transformation to extract rectified QR code."""
    dst = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (size, size))


def extract_qrcode(img: np.ndarray, output_size: int = 400,
                   min_detection_size: int = 450) -> Tuple[Optional[np.ndarray], dict]:
    """
    Extract QR code from image.

    For small images, upscales for detection but extracts from original resolution.

    Args:
        img: Input BGR image
        output_size: Size of output extracted QR code
        min_detection_size: Minimum size for reliable detection. Images smaller
                           than this will be upscaled for detection.

    Returns (extracted_image, debug_info) or (None, debug_info) if detection fails.
    """
    h, w = img.shape[:2]

    # Calculate scale factor needed for reliable detection
    min_dim = min(h, w)
    scale = 1.0
    if min_dim < min_detection_size:
        scale = min_detection_size / min_dim
        # Round up to ensure we meet minimum size
        scale = max(scale, 1.0)

    # Upscale for detection if needed
    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        detection_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        detection_img = img

    gray = cv2.cvtColor(detection_img, cv2.COLOR_BGR2GRAY)
    det_h, det_w = gray.shape

    finders = find_finder_patterns(gray)
    debug_info = {
        'finders': finders,
        'method': None,
        'corners': None,
        'scale_used': scale
    }

    if len(finders) < 2:
        debug_info['method'] = 'failed'
        return None, debug_info

    corners = compute_corners_from_finders(finders, (det_h, det_w), gray=gray)
    if corners is None:
        debug_info['method'] = 'failed'
        return None, debug_info

    # Scale corners back to original image coordinates
    if scale > 1.0:
        corners = corners / scale
        # Scale finder positions back for debug info
        debug_info['finders'] = [(int(x/scale), int(y/scale), int(s/scale))
                                  for (x, y, s) in finders]

    # Clamp corners to original image boundaries
    corners[:, 0] = np.clip(corners[:, 0], 2, w - 2)
    corners[:, 1] = np.clip(corners[:, 1], 2, h - 2)

    debug_info['method'] = f'finder({len(finders)})' + (f'@{scale:.1f}x' if scale > 1 else '')
    debug_info['corners'] = corners

    # Extract from ORIGINAL image (not upscaled)
    extracted = perspective_transform(img, corners, output_size)
    return extracted, debug_info


def draw_debug(img: np.ndarray, debug_info: dict) -> np.ndarray:
    """Draw debug visualization."""
    debug_img = img.copy()

    # Draw finder patterns
    for (x, y, size) in debug_info.get('finders', []):
        cv2.circle(debug_img, (x, y), size // 2 + 5, (255, 0, 0), 2)

    # Draw corners and quadrilateral
    corners = debug_info.get('corners')
    if corners is not None:
        pts = corners.astype(np.int32)
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i), (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return debug_img


def main():
    import os

    if len(sys.argv) < 3:
        print("Usage: python qrcode_extractor_simple.py output_dir <image_path>")
        sys.exit(1)

    out_dir = sys.argv[1]
    for img_path in sys.argv[2:]:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load {img_path}")
            sys.exit(1)

        # Generate output filenames based on input name
        basename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(basename)[0]
        debug_output = os.path.join(out_dir, f"output_{name_without_ext}_debug.png")
        extracted_output = os.path.join(out_dir, f"output_{name_without_ext}_extracted.png")

        print(f"Processing: {img_path}")
        print(f"Image size: {img.shape}")

        extracted, debug_info = extract_qrcode(img)

        print(f"Method: {debug_info['method']}")
        if debug_info.get('scale_used', 1.0) > 1.0:
            print(f"Detection scale: {debug_info['scale_used']:.1f}x")
        if debug_info['corners'] is not None:
            print("Corners:")
            for i, c in enumerate(debug_info['corners']):
                print(f"  {i}: ({c[0]:.1f}, {c[1]:.1f})")

        # Save outputs
        debug_img = draw_debug(img, debug_info)
        cv2.imwrite(debug_output, debug_img)

        if extracted is not None:
            cv2.imwrite(extracted_output, extracted)
            print(f"\nSaved: {debug_output}, {extracted_output}")
        else:
            print(f"\nDetection failed - saved: {debug_output}")


if __name__ == '__main__':
    main()
