#!/usr/bin/env python3
"""
QR Code Extractor with Perspective Correction

Detects QR code boundaries and applies perspective transformation.
Uses multiple strategies to handle various backgrounds.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


def find_finder_patterns(gray: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Find QR code finder patterns (the three large squares at corners).
    Returns list of (x, y, size) for each detected pattern.
    Uses ratio-based detection for more accuracy.

    Key insight: The 3 finder patterns are the LARGEST nested square patterns
    and they are located at the corners of the QR code (not in the middle).
    """
    h, w = gray.shape

    # Minimum finder pattern size should be at least 3% of image dimension
    min_finder_size = int(min(h, w) * 0.03)
    min_area = min_finder_size * min_finder_size

    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 51, 10)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    finder_candidates = []

    # Look for nested squares (characteristic of finder patterns)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < max(100, min_area) or area > (h * w * 0.15):  # Size constraints
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # Check squareness using circularity approximation
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:  # Not square-ish enough
            continue

        # Check for nested contours (finder pattern has 3 nested squares)
        parent = hierarchy[i][3]
        child = hierarchy[i][2]

        # Count nesting depth
        depth = 0
        idx = i
        while hierarchy[idx][3] >= 0:
            depth += 1
            idx = hierarchy[idx][3]

        # Middle ring of finder pattern should have depth >= 1 and have children
        if depth >= 1 and child >= 0:
            # Verify the child is also roughly square
            child_area = cv2.contourArea(contours[child])
            if child_area > 0:
                ratio = area / child_area
                # The ratio between outer and middle rings should be reasonable
                if 1.5 < ratio < 10:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Use bounding rect for more accurate size
                        x, y, bw, bh = cv2.boundingRect(contour)
                        size = max(bw, bh)  # Use max dimension as size
                        finder_candidates.append((cx, cy, size, area))

    if len(finder_candidates) < 3:
        return [(c[0], c[1], c[2]) for c in finder_candidates]

    # Cluster nearby candidates first (before filtering by size)
    clusters = []
    used = set()

    # Sort by area (larger first) - this is key to getting the main finder patterns
    finder_candidates.sort(key=lambda x: x[3], reverse=True)

    for i, (x1, y1, s1, a1) in enumerate(finder_candidates):
        if i in used:
            continue
        cluster = [(x1, y1, s1, a1)]
        used.add(i)

        for j, (x2, y2, s2, a2) in enumerate(finder_candidates):
            if j in used:
                continue
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < max(s1, s2) * 1.5:
                cluster.append((x2, y2, s2, a2))
                used.add(j)

        # Use the largest candidate in the cluster
        largest = max(cluster, key=lambda x: x[3])
        clusters.append((largest[0], largest[1], largest[2], largest[3]))

    if len(clusters) < 3:
        return [(c[0], c[1], c[2]) for c in clusters]

    # Now we have clustered candidates sorted by size (largest first)
    # The 3 finder patterns should be among the largest ones AND at the corners
    # Key insight: Finder patterns are at EXTREME corners, not in the middle

    # Strategy: Find patterns that are closest to the convex hull corners
    # of all detected patterns. Alignment patterns will be inside.

    from itertools import combinations

    # Get all pattern positions
    all_pts = np.array([(c[0], c[1]) for c in clusters], dtype=np.float32)

    # Find the bounding box corners of all patterns
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)

    # The 4 theoretical corners of the QR region
    corners = np.array([
        [min_x, min_y],  # top-left
        [max_x, min_y],  # top-right
        [min_x, max_y],  # bottom-left
        [max_x, max_y],  # bottom-right
    ])

    # For each cluster, calculate how close it is to any corner
    # Lower distance = more likely to be a finder pattern
    corner_distances = []
    for c in clusters:
        pt = np.array([c[0], c[1]])
        min_dist = min(np.linalg.norm(pt - corner) for corner in corners)
        corner_distances.append(min_dist)

    # Find the best triplet that:
    # 1. Has patterns close to corners (low corner distance)
    # 2. Forms a right-angle triangle
    # 3. Uses large patterns
    # 4. The three patterns should be at 3 DIFFERENT corners

    best_triplet = None
    best_score = float('inf')

    num_to_consider = min(10, len(clusters))

    for triplet in combinations(range(num_to_consider), 3):
        pts = np.array([(clusters[i][0], clusters[i][1]) for i in triplet], dtype=np.float32)
        sizes = [clusters[i][2] for i in triplet]

        # Check which corners each pattern is closest to
        closest_corners = []
        for i in triplet:
            pt = np.array([clusters[i][0], clusters[i][1]])
            dists_to_corners = [np.linalg.norm(pt - corner) for corner in corners]
            closest_corners.append(np.argmin(dists_to_corners))

        # The 3 finder patterns should be at 3 DIFFERENT corners
        if len(set(closest_corners)) < 3:
            continue  # Two patterns closest to same corner - likely not all finders

        # Calculate distances between all pairs
        d01 = np.linalg.norm(pts[0] - pts[1])
        d02 = np.linalg.norm(pts[0] - pts[2])
        d12 = np.linalg.norm(pts[1] - pts[2])

        dists = sorted([d01, d02, d12])

        if dists[0] < 10:  # Too close together
            continue

        # Right-angle triangle check: ratio1 ≈ 1, ratio2 ≈ sqrt(2)
        ratio1 = dists[1] / dists[0]
        ratio2 = dists[2] / dists[0]
        geometry_score = abs(ratio1 - 1.0) + abs(ratio2 - 1.414)

        # Penalize patterns far from corners (alignment patterns are inside)
        corner_penalty = sum(corner_distances[i] for i in triplet) / (max(w, h) + 1) * 0.5

        # Penalize smaller patterns (alignment patterns are usually smaller than finders)
        avg_size = np.mean(sizes)
        min_size = min(sizes)
        max_size_in_clusters = clusters[0][2]
        # Heavily penalize if any pattern is much smaller than the largest
        size_penalty = (max_size_in_clusters - avg_size) / max_size_in_clusters * 0.5
        # Extra penalty if minimum size is less than 60% of max (likely not a finder)
        if min_size < max_size_in_clusters * 0.6:
            size_penalty += 0.5

        # Coverage check
        triangle_area = 0.5 * abs((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) -
                                   (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]))
        img_area = w * h
        coverage = triangle_area / img_area
        coverage_penalty = max(0, 0.1 - coverage) * 2

        total_score = geometry_score + corner_penalty + size_penalty + coverage_penalty

        if total_score < best_score:
            best_score = total_score
            best_triplet = tuple(clusters[i] for i in triplet)

    if best_triplet and best_score < 3.0:
        return [(t[0], t[1], t[2]) for t in best_triplet]

    # Fallback: return the 3 largest clusters
    return [(c[0], c[1], c[2]) for c in clusters[:3]]


def estimate_qr_bounds_from_finders(finders: List[Tuple[int, int, int]],
                                     img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Estimate QR code bounds from finder pattern locations."""
    if len(finders) < 3:
        return None

    h, w = img_shape
    xs = [f[0] for f in finders]
    ys = [f[1] for f in finders]
    sizes = [f[2] for f in finders]
    avg_size = np.mean(sizes)

    # QR code extends about half a finder pattern beyond the finder centers
    margin = int(avg_size * 1.5)

    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(w, max(xs) + margin)
    y2 = min(h, max(ys) + margin)

    return x1, y1, x2, y2


def compute_corners_from_finders(finders: List[Tuple[int, int, int]],
                                  img_shape: Tuple[int, int] = None) -> Optional[np.ndarray]:
    """
    Compute QR code corners directly from finder pattern positions.
    QR codes have 3 finder patterns at corners, the 4th corner is computed.
    Returns corners ordered as: top-left, top-right, bottom-right, bottom-left
    """
    if len(finders) < 2:
        return None

    # Convert to numpy array
    pts = np.array([(f[0], f[1]) for f in finders], dtype=np.float32)
    sizes = [f[2] for f in finders]
    avg_size = np.mean(sizes)

    if len(finders) == 2:
        # Special case: only 2 finders detected
        # Assume they are adjacent corners (TL-TR or TL-BL) and infer the third
        p1, p2 = pts[0], pts[1]
        v = p2 - p1
        v_len = np.linalg.norm(v)

        # Rotate 90 degrees to get perpendicular direction
        v_perp = np.array([-v[1], v[0]])

        # Determine which configuration: are these TL-TR or TL-BL?
        # TL-TR: horizontal-ish (|dx| > |dy|)
        # TL-BL: vertical-ish (|dy| > |dx|)
        if abs(v[0]) > abs(v[1]):
            # Horizontal - these are TL and TR
            # BL is below TL
            if p1[0] < p2[0]:
                tl, tr = p1, p2
            else:
                tl, tr = p2, p1
            # BL = TL + perpendicular (downward)
            v_down = np.array([-v[1], v[0]]) if v[0] > 0 else np.array([v[1], -v[0]])
            if v_down[1] < 0:
                v_down = -v_down
            bl = tl + v_down
            br = tr + v_down
        else:
            # Vertical - these are TL and BL
            if p1[1] < p2[1]:
                tl, bl = p1, p2
            else:
                tl, bl = p2, p1
            # TR = TL + perpendicular (rightward)
            v_right = np.array([v[1], -v[0]]) if v[1] > 0 else np.array([-v[1], v[0]])
            if v_right[0] < 0:
                v_right = -v_right
            tr = tl + v_right
            br = bl + v_right

        all_corners = np.array([tl, tr, br, bl], dtype=np.float32)

    else:
        # Normal case: 3 finders
        # Find which finder is at the "corner" (the one where other two form ~90 degrees)
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

        corner_idx = best_corner_idx
        other_indices = [(corner_idx + 1) % 3, (corner_idx + 2) % 3]

        corner_pt = pts[corner_idx]
        pt1 = pts[other_indices[0]]
        pt2 = pts[other_indices[1]]

        # The 4th corner is: pt1 + pt2 - corner_pt (parallelogram)
        fourth_corner = pt1 + pt2 - corner_pt

        # All four corners
        all_corners = np.array([corner_pt, pt1, pt2, fourth_corner], dtype=np.float32)

    # Adjust corners outward to include the full finder patterns
    # We detect the middle ring (white square) of the finder pattern
    # The full finder pattern extends further due to outer black ring
    # Finder structure is 1:1:3:1:1, so outer edge is ~7/5 of middle ring
    # Use 0.6x size to reach the outer edge of finder pattern
    expand_size = avg_size * 0.6
    center = np.mean(all_corners, axis=0)
    for i in range(4):
        direction = all_corners[i] - center
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        all_corners[i] = all_corners[i] + direction * expand_size

    # Clamp to image boundaries if provided
    if img_shape is not None:
        h, w = img_shape
        all_corners[:, 0] = np.clip(all_corners[:, 0], 2, w - 2)
        all_corners[:, 1] = np.clip(all_corners[:, 1], 2, h - 2)

    # Now order corners as: top-left, top-right, bottom-right, bottom-left
    center = np.mean(all_corners, axis=0)  # Recalculate after clamping
    angles = np.arctan2(all_corners[:, 1] - center[1], all_corners[:, 0] - center[0])
    order = np.argsort(angles)
    sorted_corners = all_corners[order]

    # Find corner with smallest x+y sum (top-left)
    sums = sorted_corners[:, 0] + sorted_corners[:, 1]
    start_idx = np.argmin(sums)
    sorted_corners = np.roll(sorted_corners, -start_idx, axis=0)

    return sorted_corners


def has_qr_pattern(patch: np.ndarray) -> bool:
    """Check if patch contains QR-like pattern."""
    if patch.size == 0:
        return False
    black = np.mean(patch < 100)
    white = np.mean(patch > 160)
    std = np.std(patch)
    return black > 0.08 and white > 0.08 and std > 35


def find_qr_edge_from_outside(gray: np.ndarray, start_x: int, start_y: int,
                               dx: int, dy: int, max_dist: int = 500) -> Tuple[int, int]:
    """Scan from outside inward to find QR code edge."""
    h, w = gray.shape
    window = 12
    half_w = window // 2
    x, y = start_x, start_y

    # Skip until we find high contrast area
    for step in range(max_dist):
        x += dx
        y += dy

        if x < half_w or x >= w - half_w or y < half_w or y >= h - half_w:
            continue

        patch = gray[y-half_w:y+half_w, x-half_w:x+half_w]
        if patch.size == 0:
            continue

        std = np.std(patch)
        black_ratio = np.mean(patch < 100)

        # Found high contrast area with dark pixels
        if std > 40 and black_ratio > 0.1:
            return (x, y)

        # Found solid dark (finder pattern edge)
        if black_ratio > 0.3:
            return (x, y)

    return (start_x, start_y)


def find_edge_points(gray: np.ndarray, num_samples: int = 50) -> dict:
    """Find edge points by scanning from outside inward."""
    h, w = gray.shape
    edges = {'top': [], 'bottom': [], 'left': [], 'right': []}

    # Top edge
    for i in range(num_samples):
        x = int(w * (i + 0.5) / num_samples)
        pt = find_qr_edge_from_outside(gray, x, 0, 0, 1)
        edges['top'].append(pt)

    # Bottom edge
    for i in range(num_samples):
        x = int(w * (i + 0.5) / num_samples)
        pt = find_qr_edge_from_outside(gray, x, h - 1, 0, -1)
        edges['bottom'].append(pt)

    # Left edge
    for i in range(num_samples):
        y = int(h * (i + 0.5) / num_samples)
        pt = find_qr_edge_from_outside(gray, 0, y, 1, 0)
        edges['left'].append(pt)

    # Right edge
    for i in range(num_samples):
        y = int(h * (i + 0.5) / num_samples)
        pt = find_qr_edge_from_outside(gray, w - 1, y, -1, 0)
        edges['right'].append(pt)

    return edges


def filter_edge_points(edges: dict, img_shape: Tuple[int, int],
                       expected_bounds: Optional[Tuple[int, int, int, int]] = None) -> dict:
    """Filter edge points to remove outliers."""
    h, w = img_shape
    margin = 5
    filtered = {}

    for edge_name, points in edges.items():
        valid = []
        for pt in points:
            x, y = pt
            # Skip points at image boundaries
            if x <= margin or x >= w - margin or y <= margin or y >= h - margin:
                continue

            # If we have expected bounds, filter points too far from them
            if expected_bounds:
                x1, y1, x2, y2 = expected_bounds
                bounds_margin = max(w, h) * 0.2
                if edge_name == 'top' and y > y1 + bounds_margin:
                    continue
                if edge_name == 'bottom' and y < y2 - bounds_margin:
                    continue
                if edge_name == 'left' and x > x1 + bounds_margin:
                    continue
                if edge_name == 'right' and x < x2 - bounds_margin:
                    continue

            valid.append(pt)

        filtered[edge_name] = valid if len(valid) >= 3 else points

    return filtered


def fit_line_robust(points: List[Tuple[int, int]], edge_type: str) -> Optional[Tuple[float, float, float]]:
    """Fit a line with robust outlier rejection."""
    if len(points) < 2:
        return None

    points = np.array(points, dtype=np.float32)
    primary_axis = 1 if edge_type == 'horizontal' else 0

    # Multiple passes of outlier removal
    for _ in range(3):
        if len(points) < 3:
            break

        axis = primary_axis
        q1, q3 = np.percentile(points[:, axis], [25, 75])
        iqr = q3 - q1
        if iqr < 1:
            iqr = max(np.std(points[:, axis]), 1)
        lower = q1 - 1.0 * iqr
        upper = q3 + 1.0 * iqr
        mask = (points[:, axis] >= lower) & (points[:, axis] <= upper)
        if np.sum(mask) >= 2:
            points = points[mask]

    if len(points) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    a, b = -vy, vx
    c = -(a * x0 + b * y0)
    return a, b, c


def line_intersection(line1, line2) -> Optional[Tuple[float, float]]:
    """Find intersection of two lines."""
    if line1 is None or line2 is None:
        return None
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return x, y


def find_corners(edges: dict, img_shape: Tuple[int, int],
                 expected_bounds: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Find four corners from edge points."""
    h, w = img_shape

    edges = filter_edge_points(edges, img_shape, expected_bounds)

    lines = {
        'top': fit_line_robust(edges.get('top', []), 'horizontal'),
        'bottom': fit_line_robust(edges.get('bottom', []), 'horizontal'),
        'left': fit_line_robust(edges.get('left', []), 'vertical'),
        'right': fit_line_robust(edges.get('right', []), 'vertical'),
    }

    corners = []
    for (e1, e2) in [('top', 'left'), ('top', 'right'), ('bottom', 'right'), ('bottom', 'left')]:
        pt = line_intersection(lines.get(e1), lines.get(e2))
        if pt:
            corners.append(pt)

    if len(corners) != 4:
        raise ValueError(f"Could not find 4 corners, found {len(corners)}")

    corners = np.array(corners, dtype=np.float32)
    corners[:, 0] = np.clip(corners[:, 0], 2, w - 2)
    corners[:, 1] = np.clip(corners[:, 1], 2, h - 2)

    return corners


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order as: top-left, top-right, bottom-right, bottom-left"""
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    order = np.argsort(angles)
    sorted_corners = corners[order]

    sums = sorted_corners[:, 0] + sorted_corners[:, 1]
    start_idx = np.argmin(sums)
    sorted_corners = np.roll(sorted_corners, -start_idx, axis=0)

    return sorted_corners


def validate_quadrilateral(corners: np.ndarray, img_shape: Tuple[int, int]) -> bool:
    """Check if the quadrilateral is valid."""
    h, w = img_shape

    n = len(corners)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i, 0] * corners[j, 1]
        area -= corners[j, 0] * corners[i, 1]
    area = abs(area) / 2

    img_area = h * w
    if area < 0.03 * img_area or area > 0.98 * img_area:
        return False

    widths = [np.linalg.norm(corners[1] - corners[0]),
              np.linalg.norm(corners[2] - corners[3])]
    heights = [np.linalg.norm(corners[3] - corners[0]),
               np.linalg.norm(corners[2] - corners[1])]

    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    aspect = max(avg_width, avg_height) / (min(avg_width, avg_height) + 1e-6)
    if aspect > 2.5:
        return False

    return True


def perspective_transform(img: np.ndarray, corners: np.ndarray, size: int = 400) -> np.ndarray:
    """Apply perspective transformation."""
    dst = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (size, size))


def extract_qrcode(img: np.ndarray, output_size: int = 400) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Extract QR code with perspective correction."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Try to find finder patterns
    finders = find_finder_patterns(gray)
    corners = None
    use_finder_corners = False

    if len(finders) >= 2:
        # Try to compute corners from finder patterns (works with 2 or 3 finders)
        finder_corners = compute_corners_from_finders(finders, (h, w))
        if finder_corners is not None:
            # Validate the finder-based corners
            if validate_quadrilateral(finder_corners, (h, w)):
                corners = finder_corners
                use_finder_corners = True
                print(f"Using corners from {len(finders)} finder patterns")

    if corners is None:
        # Fall back to edge-based detection
        expected_bounds = None
        if len(finders) >= 3:
            expected_bounds = estimate_qr_bounds_from_finders(finders, (h, w))
            print(f"Found {len(finders)} finder patterns, using for bounds")

        edges = find_edge_points(gray)
        corners = find_corners(edges, (h, w), expected_bounds)
        corners = order_corners(corners)
    else:
        edges = {'top': [], 'bottom': [], 'left': [], 'right': []}

    if not validate_quadrilateral(corners, (h, w)):
        print("Warning: Detected quadrilateral may not be valid")

    result = perspective_transform(img, corners, output_size)

    return result, corners, {'edges': edges, 'corners': corners, 'finders': finders,
                              'use_finder_corners': use_finder_corners}


def draw_debug(img: np.ndarray, corners: np.ndarray, debug_info: dict) -> np.ndarray:
    """Draw debug visualization."""
    out = img.copy()

    colors = {'top': (0, 255, 0), 'bottom': (0, 255, 255),
              'left': (255, 0, 255), 'right': (255, 255, 0)}

    for name, pts in debug_info['edges'].items():
        for pt in pts:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 2, colors[name], -1)

    # Draw finder patterns
    for (fx, fy, fs) in debug_info.get('finders', []):
        cv2.circle(out, (fx, fy), fs // 2, (255, 128, 0), 2)

    for i, c in enumerate(corners):
        cv2.circle(out, (int(c[0]), int(c[1])), 8, (0, 0, 255), -1)
        cv2.putText(out, str(i), (int(c[0])+10, int(c[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], True, (0, 255, 0), 2)

    return out


def main():
    import sys

    path = "qrcode_test.png" if len(sys.argv) < 2 else sys.argv[1]
    print(f"Loading: {path}")

    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load {path}")
        return

    print(f"Image size: {img.shape}")

    try:
        result, corners, debug = extract_qrcode(img)

        print("Found corners:")
        for i, c in enumerate(corners):
            print(f"  {i}: ({c[0]:.1f}, {c[1]:.1f})")

        cv2.imwrite("qrcode_debug.png", draw_debug(img, corners, debug))
        cv2.imwrite("qrcode_extracted.png", result)
        print("\nSaved: qrcode_debug.png, qrcode_extracted.png")

        det = cv2.QRCodeDetector()
        data, _, _ = det.detectAndDecode(result)
        print(f"\nDecoded: {data}" if data else "\nCould not decode")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
