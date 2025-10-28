"""
Stitching function for sliding windowed predictions (UNPADDED DATA VERSION).

Refactored for the new setup where:
  - Data is NO LONGER PRE-PADDED (removed _padded_data)
  - Patches are extracted from unpadded data (dset._data)
  - Boundary patches handled on-demand during cropping
  - Index manager uses unpadded data shape

For your datasets:
  - 3D case: patch_spatial_dims = (9, 64, 64)  → (Z, Y, X)
  - 2D case: patch_spatial_dims = (64, 64)     → (Y, X)

Key Changes from Previous Version:
  1. Canvas initialized with dset._data.shape (unpadded)
  2. Removed dset._padded_data references (doesn't exist anymore)
  3. Removed dset.pad_width_spatial references (doesn't exist)
  4. Removed final cropping step (data already unpadded)
  5. Handles patches properly when they're extracted from unpadded space
"""

import numpy as np
from tqdm import tqdm
from typing import Iterator, Tuple, Union, List, Optional


def stitch_predictions_windowed(
    generator: Iterator[np.ndarray],
    dset,
    num_patches: int,
    inner_fraction: Union[float, List[float]] = 0.5,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stitch tile predictions in a sliding windowed manner with per-axis control.
    
    Works with UNPADDED data (refactored WindowedLCDLoader/WindowedTilingDloader).
    
    Correctly handles:
      - 2D datasets: patch_spatial_dims = (H, W) = (64, 64)
      - 3D datasets: patch_spatial_dims = (Z, H, W) = (9, 64, 64)
    
    Supports different inner crop fractions for each spatial dimension.
    Only the inner centered portion of each tile is stitched into the output canvas.
    Overlapping regions are averaged together.
    
    Args:
        generator: Generator or iterator that yields prediction arrays.
                  Each prediction should be:
                  - 2D: (H, W, C) or (C, H, W)
                  - 3D: (Z, H, W, C) or (C, Z, H, W)
        
        dset: Dataset object (WindowedLCDLoader or WindowedTilingDloader) with:
              - _data.shape: Original unpadded data (N, H, W, C) for 2D or (N, Z, H, W, C) for 3D
              - idx_manager: WindowedTilingGridIndexManager with:
                  - patch_spatial_dims: Tuple of spatial patch dimensions
                    * 2D: (H_size, W_size) e.g., (64, 64)
                    * 3D: (Z_size, H_size, W_size) e.g., (9, 64, 64)
                  - get_patch_location_from_dataset_idx(i) → (N_idx, *spatial_coords)
              - _5Ddata: Boolean flag for 3D vs 2D (optional, inferred from patch_spatial_dims)
        
        num_patches: Total number of patches to process.
        
        inner_fraction: Fraction of tile to use as inner region per axis.
                       Can be:
                       - float: Same fraction for all spatial dims (default 0.5)
                       - List[float]: Per-axis fractions in spatial order
                         * 2D: [fy, fx]
                         * 3D: [fz, fy, fx]
        
        batch_size: Process this many predictions at a time (for future optimization).
        
        debug: If True, print debug information.
    
    Returns:
        final_image: Stitched image with unpadded shape (no cropping needed!)
        coverage_mask: Pixel coverage count array (same shape as final_image)
    
    Examples:
        # 2D: uniform 50% center crop
        stitched, coverage = stitch_predictions_windowed(
            gen, dset, len(dset), inner_fraction=0.5
        )
        
        # 2D: asymmetric per-axis
        stitched, coverage = stitch_predictions_windowed(
            gen, dset, len(dset), inner_fraction=[0.5, 0.25]  # 50% Y, 25% X
        )
        
        # 3D: full Z, center 50% XY (RECOMMENDED FOR 3D)
        stitched, coverage = stitch_predictions_windowed(
            gen, dset, len(dset), inner_fraction=[1.0, 0.5, 0.5]
        )
    """
    
    # ========================================================================
    # Get dimensions from dataset (UNPADDED)
    # ========================================================================
    original_shape = dset._data.shape
    idx_manager = dset.idx_manager
    
    # Determine if 2D or 3D from patch_spatial_dims
    patch_spatial_dims = idx_manager.patch_spatial_dims
    num_spatial_dims = len(patch_spatial_dims)
    
    is_3d = num_spatial_dims == 3
    
    if debug:
        print(f"[DEBUG] Original unpadded data shape: {original_shape}")
        print(f"[DEBUG] patch_spatial_dims: {patch_spatial_dims}")
        print(f"[DEBUG] Data is {'3D' if is_3d else '2D'}")
        print(f"[DEBUG] Number of patches to process: {num_patches}")
    
    # ========================================================================
    # Initialize output canvas (UNPADDED SIZE - NO CROPPING NEEDED AT END!)
    # ========================================================================
    stitched = np.zeros(original_shape, dtype=np.float32)
    counts = np.zeros_like(stitched)
    
    if debug:
        print(f"[DEBUG] Initialized canvas with shape: {stitched.shape}")
    
    # ========================================================================
    # Parse inner_fraction into per-axis list
    # ========================================================================
    if isinstance(inner_fraction, (int, float)):
        inner_fractions = [inner_fraction] * num_spatial_dims
    elif isinstance(inner_fraction, (list, tuple)):
        inner_fractions = list(inner_fraction)
        if len(inner_fractions) != num_spatial_dims:
            raise ValueError(
                f"inner_fraction list length ({len(inner_fractions)}) must match "
                f"number of spatial dimensions ({num_spatial_dims}). "
                f"Expected {num_spatial_dims} values for {'3D (Z,H,W)' if is_3d else '2D (H,W)'}"
            )
    else:
        raise TypeError(f"inner_fraction must be float or list, got {type(inner_fraction)}")
    
    # ========================================================================
    # Compute inner crop parameters per axis
    # ========================================================================
    inner_tile_sizes = []
    start_inners = []
    end_inners = []
    
    for axis_idx, (full_size, frac) in enumerate(zip(patch_spatial_dims, inner_fractions)):
        inner_size = int(full_size * frac)
        start = (full_size - inner_size) // 2
        end = start + inner_size
        
        inner_tile_sizes.append(inner_size)
        start_inners.append(start)
        end_inners.append(end)
    
    if debug:
        print(f"[DEBUG] Full tile sizes: {patch_spatial_dims}")
        print(f"[DEBUG] Inner fractions: {inner_fractions}")
        print(f"[DEBUG] Inner tile sizes: {inner_tile_sizes}")
        print(f"[DEBUG] Start crop indices: {start_inners}")
        print(f"[DEBUG] End crop indices: {end_inners}")
    
    # ========================================================================
    # Process predictions from generator
    # ========================================================================
    patch_idx = 0
    for pred_batch in tqdm(generator, total=num_patches, desc="Stitching predictions"):
        # Handle case where generator yields batches
        if isinstance(pred_batch, (list, tuple)):
            pred_list = pred_batch if isinstance(pred_batch, list) else [pred_batch]
        else:
            pred_list = [pred_batch]
        
        for pred in pred_list:
            if patch_idx >= num_patches:
                break
            
            # ================================================================
            # Ensure prediction is in spatial-last format
            # ================================================================
            if is_3d:
                # Could be (C, Z, H, W) - check if C is smallest
                if pred.ndim == 4 and pred.shape[0] < min(pred.shape[1:]):
                    pred = np.transpose(pred, (1, 2, 3, 0))  # → (Z, H, W, C)
            else:
                # 2D: Could be (C, H, W)
                if pred.ndim == 3 and pred.shape[0] < min(pred.shape[1:]):
                    pred = np.transpose(pred, (1, 2, 0))  # → (H, W, C)
            
            # ================================================================
            # Extract inner crop from prediction using per-axis fractions
            # ================================================================
            if is_3d:
                # For 3D: (Z, H, W, C)
                inner_pred = pred[
                    start_inners[0]:end_inners[0],
                    start_inners[1]:end_inners[1],
                    start_inners[2]:end_inners[2],
                    :
                ]
            else:
                # For 2D: (H, W, C)
                inner_pred = pred[
                    start_inners[0]:end_inners[0],
                    start_inners[1]:end_inners[1],
                    :
                ]
            
            # ================================================================
            # Get patch location in UNPADDED data space
            # ================================================================
            loc = idx_manager.get_patch_location_from_dataset_idx(patch_idx)
            batch_idx = loc[0]  # N index
            
            if is_3d:
                # loc = (N, Z, H, W)
                z_start, h_start, w_start = loc[1], loc[2], loc[3]
                
                # Compute inner crop positions in unpadded data space
                z_start_inner = z_start + start_inners[0]
                h_start_inner = h_start + start_inners[1]
                w_start_inner = w_start + start_inners[2]
                
                z_end_inner = z_start_inner + inner_tile_sizes[0]
                h_end_inner = h_start_inner + inner_tile_sizes[1]
                w_end_inner = w_start_inner + inner_tile_sizes[2]
                
                # Bounds check against UNPADDED data shape
                # Clips to actual image boundaries
                z_end_inner = min(z_end_inner, original_shape[1])
                h_end_inner = min(h_end_inner, original_shape[2])
                w_end_inner = min(w_end_inner, original_shape[3])
                
                # Clip inner_pred if it exceeds bounds
                z_inner_crop = max(0, z_end_inner - z_start_inner)
                h_inner_crop = max(0, h_end_inner - h_start_inner)
                w_inner_crop = max(0, w_end_inner - w_start_inner)
                
                # Sanity check - skip if dimensions become invalid
                if z_inner_crop <= 0 or h_inner_crop <= 0 or w_inner_crop <= 0:
                    if debug:
                        print(f"[WARNING] Patch {patch_idx} has invalid inner_cropped dimensions: "
                              f"Z:{z_inner_crop}, H:{h_inner_crop}, W:{w_inner_crop}")
                    patch_idx += 1
                    continue
                
                inner_pred_inner_cropped = inner_pred[:z_inner_crop, :h_inner_crop, :w_inner_crop, :]
                
                if debug and patch_idx < 5:  # Debug first few patches
                    print(
                        f"[DEBUG] Patch {patch_idx}: batch={batch_idx}, "
                        f"Z:[{z_start_inner},{z_end_inner}), "
                        f"H:[{h_start_inner},{h_end_inner}), "
                        f"W:[{w_start_inner},{w_end_inner}), "
                        f"pred_shape={pred.shape}, inner_cropped_shape={inner_pred_inner_cropped.shape}"
                    )
                
                # Add to canvas
                stitched[batch_idx, z_start_inner:z_end_inner, h_start_inner:h_end_inner, w_start_inner:w_end_inner, :] += inner_pred_inner_cropped
                counts[batch_idx, z_start_inner:z_end_inner, h_start_inner:h_end_inner, w_start_inner:w_end_inner, :] += 1
                
            else:
                # loc = (N, H, W)
                h_start, w_start = loc[1], loc[2]
                
                # Compute inner crop positions in unpadded data space
                h_start_inner = h_start + start_inners[0]
                w_start_inner = w_start + start_inners[1]
                
                h_end_inner = h_start_inner + inner_tile_sizes[0]
                w_end_inner = w_start_inner + inner_tile_sizes[1]
                
                # Bounds check against UNPADDED data shape
                h_end_inner = min(h_end_inner, original_shape[1])
                w_end_inner = min(w_end_inner, original_shape[2])
                
                # Clip inner_pred if it exceeds bounds
                h_inner_crop = max(0, h_end_inner - h_start_inner)
                w_inner_crop = max(0, w_end_inner - w_start_inner)
                
                # Sanity check
                if h_inner_crop <= 0 or w_inner_crop <= 0:
                    if debug:
                        print(f"[WARNING] Patch {patch_idx} has invalid inner_cropped dimensions: "
                              f"H:{h_inner_crop}, W:{w_inner_crop}")
                    patch_idx += 1
                    continue
                
                inner_pred_inner_cropped = inner_pred[:h_inner_crop, :w_inner_crop, :]
                
                if debug and patch_idx < 5:  # Debug first few patches
                    print(
                        f"[DEBUG] Patch {patch_idx}: batch={batch_idx}, "
                        f"H:[{h_start_inner},{h_end_inner}), "
                        f"W:[{w_start_inner},{w_end_inner}), "
                        f"pred_shape={pred.shape}, inner_cropped_shape={inner_pred_inner_cropped.shape}"
                    )
                
                # Add to canvas
                stitched[batch_idx, h_start_inner:h_end_inner, w_start_inner:w_end_inner, :] += inner_pred_inner_cropped
                counts[batch_idx, h_start_inner:h_end_inner, w_start_inner:w_end_inner, :] += 1
            
            patch_idx += 1
    
    if patch_idx < num_patches:
        print(f"[WARNING] Only processed {patch_idx}/{num_patches} patches from generator")
    
    # ========================================================================
    # Average overlapping regions
    # ========================================================================
    counts[counts == 0] = 1  # Avoid division by zero
    stitched /= counts
    
    # ========================================================================
    # NO CROPPING NEEDED - data already in unpadded shape!
    # ========================================================================
    # The canvas was initialized with original_shape (unpadded)
    # so final_image is already the correct unpadded size
    
    if debug:
        print(f"[DEBUG] Final stitched shape: {stitched.shape}")
        print(f"[DEBUG] Original shape was: {original_shape}")
        print(f"[DEBUG] Shapes match: {stitched.shape == original_shape}")
    
    # Return coverage mask with same shape as final image
    # (useful for understanding coverage per-pixel)
    coverage_mask = counts
    
    return stitched, coverage_mask


# ============================================================================
# Utility Functions for Coverage Analysis
# ============================================================================

def analyze_coverage(coverage_mask: np.ndarray, debug: bool = True) -> dict:
    """
    Analyze the coverage of stitched predictions.
    
    Args:
        coverage_mask: Output from stitch_predictions_windowed
        debug: Print analysis
    
    Returns:
        Dictionary with coverage statistics
    """
    # Find non-zero coverage
    non_zero = coverage_mask > 0
    
    stats = {
        "total_pixels": coverage_mask.size,
        "covered_pixels": np.sum(non_zero),
        "uncovered_pixels": np.sum(~non_zero),
        "coverage_percentage": 100 * np.sum(non_zero) / coverage_mask.size,
        "min_coverage": np.min(coverage_mask[non_zero]) if np.any(non_zero) else 0,
        "max_coverage": np.max(coverage_mask),
        "mean_coverage": np.mean(coverage_mask[non_zero]) if np.any(non_zero) else 0,
    }
    
    if debug:
        print("\n" + "=" * 60)
        print("Coverage Analysis")
        print("=" * 60)
        print(f"Total pixels: {stats['total_pixels']:,}")
        print(f"Covered pixels: {stats['covered_pixels']:,} ({stats['coverage_percentage']:.1f}%)")
        print(f"Uncovered pixels: {stats['uncovered_pixels']:,}")
        print(f"Coverage range: {stats['min_coverage']:.1f} - {stats['max_coverage']:.1f}")
        print(f"Mean coverage: {stats['mean_coverage']:.2f}")
        print("=" * 60 + "\n")
    
    return stats


def find_uncovered_regions(coverage_mask: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    Find regions with insufficient coverage.
    
    Args:
        coverage_mask: Output from stitch_predictions_windowed
        threshold: Pixels with coverage < threshold are considered uncovered
    
    Returns:
        Boolean mask of uncovered regions
    """
    return coverage_mask < threshold


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage with refactored windowed datasets.
    """
    
    # Example 1: 2D with patch_spatial_dims = (64, 64)
    # from your_dataset_module import WindowedLCDLoader
    # dset = WindowedLCDLoader(config, data_path, load_fn)
    # stitched_img, coverage = stitch_predictions_windowed(
    #     gen, dset, len(dset),
    #     inner_fraction=0.5,  # 50% center both dims
    #     debug=True
    # )
    # stats = analyze_coverage(coverage)
    
    # Example 2: 3D with patch_spatial_dims = (9, 64, 64)
    # dset = WindowedLCDLoader(config, data_path, load_fn)
    # stitched_img, coverage = stitch_predictions_windowed(
    #     gen, dset, len(dset),
    #     inner_fraction=[1.0, 0.5, 0.5],  # Full Z, center 50% XY
    #     debug=True
    # )
    # stats = analyze_coverage(coverage)
    # uncovered = find_uncovered_regions(coverage)
    
    # Example 3: Asymmetric inner fractions
    # stitched_img, coverage = stitch_predictions_windowed(
    #     gen, dset, len(dset),
    #     inner_fraction=[0.8, 0.6, 0.4],  # Different per-axis
    #     debug=True
    # )
    
    pass
