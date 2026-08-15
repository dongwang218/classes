#!/usr/bin/env python3
"""
Test qrcode_extractor_simple.py with resized images.
Compares detection results between original and upscaled versions.
"""

import cv2
import numpy as np
import os
from qrcode_extractor_simple import extract_qrcode, draw_debug

def test_with_resize(img_path, scale_factors=[2, 3, 4]):
    """Test QR code extraction with different scale factors."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load {img_path}")
        return

    basename = os.path.basename(img_path)
    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"Testing: {basename}")
    print(f"Original size: {w}x{h}")

    # Test original
    extracted, debug_info = extract_qrcode(img)
    finders_found = len(debug_info.get('finders', []))
    success = extracted is not None
    print(f"  Original: finders={finders_found}, success={success}")

    results = [('original', 1, success, finders_found)]

    # Test with different scale factors
    for scale in scale_factors:
        new_w, new_h = w * scale, h * scale
        # Use INTER_CUBIC for upscaling (better quality than INTER_LINEAR)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        extracted_resized, debug_resized = extract_qrcode(resized)
        finders_resized = len(debug_resized.get('finders', []))
        success_resized = extracted_resized is not None
        print(f"  {scale}x ({new_w}x{new_h}): finders={finders_resized}, success={success_resized}")

        results.append((f'{scale}x', scale, success_resized, finders_resized))

        # Save debug output for best result
        if success_resized and not success:
            debug_img = draw_debug(resized, debug_resized)
            out_name = f"resized_debug_{basename.replace('.png', '')}_{scale}x.png"
            cv2.imwrite(out_name, debug_img)
            if extracted_resized is not None:
                cv2.imwrite(f"resized_extracted_{basename.replace('.png', '')}_{scale}x.png", extracted_resized)

    return results


def main():
    test_dir = "test_set_crop"
    all_results = []

    # Get all PNG files
    files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])

    print("Testing QR code extraction with resized images")
    print("="*60)

    for f in files:
        img_path = os.path.join(test_dir, f)
        results = test_with_resize(img_path)
        if results:
            all_results.append((f, results))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'File':<45} {'Orig':>6} {'2x':>6} {'3x':>6} {'4x':>6}")
    print("-"*60)

    orig_success = 0
    best_success = 0

    for filename, results in all_results:
        row = [filename[:44]]
        orig_ok = False
        any_ok = False
        for name, scale, success, finders in results:
            symbol = '✓' if success else f'({finders})'
            row.append(symbol)
            if scale == 1 and success:
                orig_ok = True
            if success:
                any_ok = True
        print(f"{row[0]:<45} {row[1]:>6} {row[2]:>6} {row[3]:>6} {row[4]:>6}")

        if orig_ok:
            orig_success += 1
        if any_ok:
            best_success += 1

    print("-"*60)
    print(f"Original success: {orig_success}/{len(all_results)}")
    print(f"Best (with resize): {best_success}/{len(all_results)}")


if __name__ == '__main__':
    main()
