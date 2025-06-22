#!/usr/bin/env python3
"""
Test script for comparing all four replication patterns:

1. Row-wise replication between warps + Row-wise replication between threads
   - Warps in same row share data
   - Threads in same row within a warp share data

2. Column-wise replication between warps + Row-wise replication between threads
   - Warps in same column share data
   - Threads in same row within a warp share data

3. Row-wise replication between warps + Column-wise replication between threads
   - Warps in same row share data
   - Threads in same column within a warp share data

4. Column-wise replication between warps + Column-wise replication between threads
   - Warps in same column share data
   - Threads in same column within a warp share data
"""

import matplotlib.pyplot as plt
import numpy as np
from visualizer import visualize_hierarchical_tiles, calculate_data_ids
import os

def create_test_structure():
    """Create a common structure for testing replication patterns"""
    # Define dimensions to match the encoding
    warp_per_block = [2, 2]   # 2×2 warp grid
    thread_per_warp = [4, 2]  # 4×2 threads per warp
    
    structure = {
        'TileName': 'Replication Pattern Test',
        'WarpPerBlock': warp_per_block,
        'ThreadPerWarp': thread_per_warp,
        'VectorDimensions': [2],  # Vector_N = 2
        'BlockSize': [warp_per_block[0] * thread_per_warp[0], 
                     warp_per_block[1] * thread_per_warp[1]],
        'ThreadBlocks': {}
    }
    
    # Create thread blocks (warps)
    for warp_idx in range(warp_per_block[0] * warp_per_block[1]):
        warp_key = f'Warp{warp_idx}'
        structure['ThreadBlocks'][warp_key] = {}
        
        # Fill threads in this warp
        for thread_row in range(thread_per_warp[0]):
            for thread_col in range(thread_per_warp[1]):
                thread_idx = thread_row * thread_per_warp[1] + thread_col
                global_thread_id = warp_idx * (thread_per_warp[0] * thread_per_warp[1]) + thread_idx
                
                structure['ThreadBlocks'][warp_key][f'Thread{thread_idx}'] = {
                    'global_id': global_thread_id,
                    'position': [thread_row, thread_col]
                }
    
    return structure

def test_warp_row_thread_row():
    """
    Test pattern 1: Row-wise warp replication + Row-wise thread replication
    - P0[0] maps to R[0]=WarpPerBlock_N causing row-wise replication between warps
      (warps in the same row share data: Warp0 and Warp1 have same data, Warp2 and Warp3 have same data)
    - P1[0] maps to R[1]=ThreadPerWarp_N causing row-wise replication between threads
      (threads in the same row share data within each warp)
      
    This matches a tile_distribution_encoding like:
    tile_distribution_encoding<
        sequence<S::WarpPerBlock_N, S::ThreadPerWarp_N>,  // R dimensions
        tuple<sequence<S::Repeat_N, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_N>>,
        tuple<sequence<0, 1>, sequence<0, 1>>,  // Ps2RHssMajor
        tuple<sequence<1, 0>, sequence<2, 1>>,  // Ps2RHssMinor (critical for row-wise pattern)
        sequence<1, 1>,  // Ys2RHsMajor
        sequence<0, 3>>{}  // Ys2RHsMinor
    """
    print("=== Testing Pattern 1: Row-Warp + Row-Thread Replication ===")
    
    # Create structure
    structure = create_test_structure()
    
    # Create encoding for row-wise replication at both levels
    encoding = {
        'Ps2RHssMajor': [[0, 1], [0, 1]],   # P0 -> R (Major=0), P1 -> R (Major=0)
        'Ps2RHssMinor': [[1, 0], [2, 1]],   # P0 -> R[0]=WarpPerBlock_N, P1 -> R[1]=ThreadPerWarp_N
    }
    
    # Set RsLengths to [2, 2] -> 2 rows and 2 columns for both warp and thread levels
    rs_lengths = [2, 2]
    
    # Store encoding and rs_lengths in the structure for reference
    structure['encoding'] = encoding
    structure['rs_lengths'] = rs_lengths
    
    # Calculate data IDs
    data_ids = calculate_data_ids(structure, encoding, rs_lengths)
    
    # Verify replication pattern
    verify_replication_pattern(data_ids, "Row-Warp + Row-Thread")
    
    # Return the data for visualization
    return data_ids, "Pattern 1: Row-Warp + Row-Thread"

def test_warp_col_thread_row():
    """
    Test pattern 2: Column-wise warp replication + Row-wise thread replication
    - P0[0] maps to R[0]=WarpPerBlock_M causing column-wise replication between warps
      (warps in the same column share data: Warp0 and Warp2 have same data, Warp1 and Warp3 have same data)
    - P1[0] maps to R[1]=ThreadPerWarp_N causing row-wise replication between threads
      (threads in the same row share data within each warp)

    This matches a tile_distribution_encoding like:
    tile_distribution_encoding<
        sequence<S::WarpPerBlock_M, S::ThreadPerWarp_N>,  // R dimensions
        tuple<sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_M, S::Vector_N>>,
        tuple<sequence<0, 1>, sequence<0, 1>>,
        tuple<sequence<0, 0>, sequence<2, 1>>,  // P0[0]=minor 0 for column-wise, P1[0]=minor 2 for row-wise
        sequence<1, 1>,
        sequence<0, 3>>{}
    """
    print("\n=== Testing Pattern 2: Column-Warp + Row-Thread Replication ===")
    
    # Create structure
    structure = create_test_structure()
    
    # Create encoding for column-wise warp + row-wise thread replication
    encoding = {
        'Ps2RHssMajor': [[0, 1], [0, 1]],  # Major indices
        'Ps2RHssMinor': [[0, 0], [2, 1]],  # P0[0]=minor 0 for column-wise, P1[0]=minor 2 for row-wise
    }
    
    # Set RsLengths to [2, 2] -> 2 rows and 2 columns for both warp and thread levels
    rs_lengths = [2, 2]
    
    # Store encoding and rs_lengths in the structure for reference
    structure['encoding'] = encoding
    structure['rs_lengths'] = rs_lengths
    
    # Calculate data IDs
    data_ids = calculate_data_ids(structure, encoding, rs_lengths)
    
    # Verify replication pattern
    verify_replication_pattern(data_ids, "Col-Warp + Row-Thread")
    
    # Return the data for visualization
    return data_ids, "Column-Warp + Row-Thread"

def test_warp_row_thread_col():
    """
    Test pattern 3: Row-wise warp replication + Column-wise thread replication
    - P0[0] maps to R[0]=WarpPerBlock_N causing row-wise replication between warps
      (warps in the same row share data: Warp0 and Warp1 have same data, Warp2 and Warp3 have same data)
    - P1[0] maps to R[1]=ThreadPerWarp_M causing column-wise replication between threads
      (threads in the same column within a warp share data)
      
    This matches a tile_distribution_encoding like:
    tile_distribution_encoding<
        sequence<S::WarpPerBlock_N, S::ThreadPerWarp_M>,  // R dimensions
        tuple<sequence<S::Repeat_N, S::WarpPerBlock_M, S::ThreadPerWarp_N, S::Vector_N>>,
        tuple<sequence<0, 1>, sequence<0, 1>>,
        tuple<sequence<1, 0>, sequence<1, 1>>,  // P0[0]=minor 1 for row-wise, P1[0]=minor 1 for column-wise
        sequence<1, 1>,
        sequence<0, 3>>{}
    """
    print("\n=== Testing Pattern 3: Row-Warp + Column-Thread Replication ===")
    
    # Create structure
    structure = create_test_structure()
    
    # Create encoding for row-wise warp + column-wise thread replication
    encoding = {
        'Ps2RHssMajor': [[0, 1], [0, 1]],  # Major indices
        'Ps2RHssMinor': [[1, 0], [1, 1]],  # P0[0]=minor 1 for row-wise, P1[0]=minor 1 for column-wise
    }
    
    # Set RsLengths to [2, 2] -> 2 rows and 2 columns for both warp and thread levels
    rs_lengths = [2, 2]
    
    # Store encoding and rs_lengths in the structure for reference
    structure['encoding'] = encoding
    structure['rs_lengths'] = rs_lengths
    
    # Calculate data IDs
    data_ids = calculate_data_ids(structure, encoding, rs_lengths)
    
    # Verify replication pattern
    verify_replication_pattern(data_ids, "Row-Warp + Column-Thread")
    
    # Return the data for visualization
    return data_ids, "Row-Warp + Column-Thread"

def test_warp_col_thread_col():
    """
    Test pattern 4: Column-wise warp replication + Column-wise thread replication
    - P0[0] maps to R[0]=WarpPerBlock_M causing column-wise replication between warps
      (warps in the same column share data: Warp0 and Warp2 have same data, Warp1 and Warp3 have same data)
    - P1[0] maps to R[1]=ThreadPerWarp_M causing column-wise replication between threads
      (threads in the same column within a warp share data)
      
    This matches a tile_distribution_encoding like:
    tile_distribution_encoding<
        sequence<S::WarpPerBlock_M, S::ThreadPerWarp_M>,  // R dimensions
        tuple<sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
        tuple<sequence<0, 1>, sequence<0, 1>>,
        tuple<sequence<0, 0>, sequence<1, 1>>,  // P0[0]=minor 0 for column-wise, P1[0]=minor 1 for column-wise
        sequence<1, 1>,
        sequence<0, 3>>{}
    """
    print("\n=== Testing Pattern 4: Column-Warp + Column-Thread Replication ===")
    
    # Create structure
    structure = create_test_structure()
    
    # Create encoding for column-wise warp + column-wise thread replication
    encoding = {
        'Ps2RHssMajor': [[0, 1], [0, 1]],  # Major indices
        'Ps2RHssMinor': [[0, 0], [1, 1]],  # P0[0]=minor 0 for column-wise, P1[0]=minor 1 for column-wise
    }
    
    # Set RsLengths to [2, 2] -> 2 rows and 2 columns for both warp and thread levels
    rs_lengths = [2, 2]
    
    # Store encoding and rs_lengths in the structure for reference
    structure['encoding'] = encoding
    structure['rs_lengths'] = rs_lengths
    
    # Calculate data IDs
    data_ids = calculate_data_ids(structure, encoding, rs_lengths)
    
    # Verify replication pattern
    verify_replication_pattern(data_ids, "Col-Warp + Col-Thread")
    
    # Return the data for visualization
    return data_ids, "Column-Warp + Column-Thread"

def verify_replication_pattern(data_ids, pattern_type):
    """
    Verify that the replication pattern works correctly.
    
    For row-wise warp replication (WarpPerBlock_N):
    - Warps in the same row should share data (Warp0 and Warp1, Warp2 and Warp3)
    - This means warps with the same row coordinate share data
    
    For column-wise warp replication (WarpPerBlock_M):
    - Warps in the same column should share data (Warp0 and Warp2, Warp1 and Warp3)
    - This means warps with the same column coordinate share data
    
    For row-wise thread replication (ThreadPerWarp_N):
    - Threads in the same row within a warp should share data
    
    For column-wise thread replication (ThreadPerWarp_M):
    - Threads in the same column within a warp should share data
    """
    print(f"\n=== Verification for {pattern_type} ===")
    
    # Create a grid representation of the thread IDs and data IDs for visualization
    thread_per_warp = 8  # 4×2 threads per warp
    
    # Define warps in our 2×2 grid
    # Warp0 (row 0, col 0), Warp1 (row 0, col 1)
    # Warp2 (row 1, col 0), Warp3 (row 1, col 1)
    warp0_threads = [i for i in range(0, thread_per_warp)]
    warp1_threads = [i for i in range(thread_per_warp, 2*thread_per_warp)]
    warp2_threads = [i for i in range(2*thread_per_warp, 3*thread_per_warp)]
    warp3_threads = [i for i in range(3*thread_per_warp, 4*thread_per_warp)]
    
    # --- WARP LEVEL REPLICATION CHECKS ---
    if "Row-Warp" in pattern_type:
        # For row-wise replication (WarpPerBlock_N), warps in the same row should share data
        # Warp0 and Warp1 are in row 0, Warp2 and Warp3 are in row 1
        warp_row0_same = all(data_ids.get(warp0_threads[i]) == data_ids.get(warp1_threads[i]) 
                             for i in range(thread_per_warp))
        warp_row1_same = all(data_ids.get(warp2_threads[i]) == data_ids.get(warp3_threads[i])
                             for i in range(thread_per_warp))
        
        # Different rows should have different data
        rows_different = all(data_ids.get(warp0_threads[i]) != data_ids.get(warp2_threads[i])
                            for i in range(thread_per_warp))
        
        warp_level_correct = warp_row0_same and warp_row1_same and rows_different
        print(f"Warp-level row replication test passed: {warp_level_correct}")
        if not warp_row0_same:
            print(f"  - Warp0 and Warp1 (row 0) have different data IDs")
        if not warp_row1_same:
            print(f"  - Warp2 and Warp3 (row 1) have different data IDs")
        if not rows_different:
            print(f"  - Row 0 and Row 1 warps share some data IDs but shouldn't")
    else:  # Column-wise warp replication
        # For column-wise replication (WarpPerBlock_M), warps in the same column should share data
        # Warp0 and Warp2 are in column 0, Warp1 and Warp3 are in column 1
        warp_col0_same = all(data_ids.get(warp0_threads[i]) == data_ids.get(warp2_threads[i]) 
                             for i in range(thread_per_warp))
        warp_col1_same = all(data_ids.get(warp1_threads[i]) == data_ids.get(warp3_threads[i])
                             for i in range(thread_per_warp))
        
        # Different columns should have different data
        cols_different = all(data_ids.get(warp0_threads[i]) != data_ids.get(warp1_threads[i])
                            for i in range(thread_per_warp))
        
        warp_level_correct = warp_col0_same and warp_col1_same and cols_different
        print(f"Warp-level column replication test passed: {warp_level_correct}")
        if not warp_col0_same:
            print(f"  - Warp0 and Warp2 (column 0) have different data IDs")
        if not warp_col1_same:
            print(f"  - Warp1 and Warp3 (column 1) have different data IDs")
        if not cols_different:
            print(f"  - Column 0 and Column 1 warps share some data IDs but shouldn't")
            
    # --- THREAD LEVEL REPLICATION CHECKS ---
    if "Row-Thread" in pattern_type:
        # For row-wise thread replication, threads in the same row should share data
        # Each row has 2 threads (col 0 and col 1)
        thread_level_correct = True
        # Check each warp
        for warp_threads in [warp0_threads, warp1_threads, warp2_threads, warp3_threads]:
            for row in range(4):  # 4 rows per warp
                base_idx = row * 2  # Start of this row
                thread1 = warp_threads[base_idx]
                thread2 = warp_threads[base_idx + 1]
                if data_ids.get(thread1) != data_ids.get(thread2):
                    thread_level_correct = False
                    print(f"Thread level row replication failed: threads {thread1} and {thread2} should share data")
                    
        print(f"Thread-level row replication test passed: {thread_level_correct}")
    else:  # Column-wise thread replication
        # For column-wise thread replication, threads in the same column should share data
        # Each column has 4 threads (rows 0-3)
        thread_level_correct = True
        # Check each warp
        for warp_threads in [warp0_threads, warp1_threads, warp2_threads, warp3_threads]:
            for col in range(2):  # 2 columns per warp
                # Compare each thread with the one in the next row (same column)
                for row in range(3):  # For rows 0-2, check with the next row
                    thread1 = warp_threads[row * 2 + col]
                    thread2 = warp_threads[(row + 1) * 2 + col]
                    if data_ids.get(thread1) != data_ids.get(thread2):
                        thread_level_correct = False
                        print(f"Thread level column replication failed: threads {thread1} and {thread2} should share data")
        
        print(f"Thread-level column replication test passed: {thread_level_correct}")
    
    # Print a visual representation of the data ID pattern
    print("\nData ID Pattern (4×2 threads per warp):")
    print("Warp0 (row 0, col 0)           Warp1 (row 0, col 1)")
    
    # Get data IDs for each warp and format in a grid
    warp0_data = [data_ids.get(i, -1) for i in warp0_threads]
    warp1_data = [data_ids.get(i, -1) for i in warp1_threads]
    warp2_data = [data_ids.get(i, -1) for i in warp2_threads]
    warp3_data = [data_ids.get(i, -1) for i in warp3_threads]
    
    # Print 4 rows of 2 columns for each warp (4×2 thread layout)
    for row in range(4):
        col_indices = [row*2, row*2+1]  # Indices for this row
        print(f"{warp0_data[col_indices[0]]:2d} {warp0_data[col_indices[1]]:2d}    ", end="")
        print(f"{warp1_data[col_indices[0]]:2d} {warp1_data[col_indices[1]]:2d}")

    print("\nWarp2 (row 1, col 0)           Warp3 (row 1, col 1)")
    for row in range(4):
        col_indices = [row*2, row*2+1]
        print(f"{warp2_data[col_indices[0]]:2d} {warp2_data[col_indices[1]]:2d}    ", end="")
        print(f"{warp3_data[col_indices[0]]:2d} {warp3_data[col_indices[1]]:2d}")
    
    if warp_level_correct and thread_level_correct:
        print(f"\nAll tests PASSED! {pattern_type} replication is working correctly.")
    else:
        print(f"\nSome tests FAILED. Check the implementation.")
    
    return warp_level_correct and thread_level_correct

def create_comparison_visualization(patterns):
    """Create a visual comparison of all replication patterns."""
    num_patterns = len(patterns)
    
    # Determine grid layout
    if num_patterns <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle('Replication Pattern Comparison', fontsize=16)
    
    # Flatten axs for easier indexing
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]  # Single subplot case
    
    # Create a visualization for each pattern
    for i, (data_ids, pattern_type) in enumerate(patterns):
        if i < len(axs):
            display_data_id_pattern(axs[i], data_ids, pattern_type)
    
    # Hide empty subplots
    for i in range(len(patterns), len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("all_replication_patterns.png")
    print("\nCreated visual comparison of all replication patterns: all_replication_patterns.png")

def update_visualizer():
    """Fix any issues in the visualizer code if needed"""
    # This function would contain any fixes needed for the visualizer.py file
    # Based on test results
    print("\nChecking if visualizer.py needs updates...")
    # Here we would add any fixes needed

def display_data_id_pattern(ax, data_ids, pattern_name):
    """
    Display the data ID pattern on a given matplotlib axis.
    
    Args:
        ax: The matplotlib axis to draw on
        data_ids: Dict mapping thread_id to data_id
        pattern_name: Name of the pattern for the title
    """
    thread_per_warp = 8  # 4×2 threads per warp
    
    # Define warps
    warp0_threads = [i for i in range(0, thread_per_warp)]
    warp1_threads = [i for i in range(thread_per_warp, 2*thread_per_warp)]
    warp2_threads = [i for i in range(2*thread_per_warp, 3*thread_per_warp)]
    warp3_threads = [i for i in range(3*thread_per_warp, 4*thread_per_warp)]
    
    # Get data IDs for each warp
    warp_data = [
        [data_ids.get(i, -1) for i in warp0_threads],
        [data_ids.get(i, -1) for i in warp1_threads],
        [data_ids.get(i, -1) for i in warp2_threads],
        [data_ids.get(i, -1) for i in warp3_threads]
    ]
    
    # Setup the axis
    ax.set_title(pattern_name, fontsize=12)
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Positions for the warps in a 2×2 grid
    positions = [
        (0, 4),  # Warp0 (top left)
        (4, 4),  # Warp1 (top right)
        (0, 0),  # Warp2 (bottom left)
        (4, 0)   # Warp3 (bottom right)
    ]
    
    # Draw each warp
    for w, (warp_x, warp_y) in enumerate(positions):
        # Draw warp outline
        rect = plt.Rectangle((warp_x, warp_y), 4, 4, 
                            linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(warp_x + 2, warp_y + 4.5, f"Warp{w}", 
                ha='center', fontsize=10, fontweight='bold')
        
        # Plot thread data IDs
        for row in range(4):
            for col in range(2):
                idx = row * 2 + col
                if idx < len(warp_data[w]):
                    # Data ID
                    data_id = warp_data[w][idx]
                    ax.text(warp_x + col + 0.5, warp_y + row + 0.5, 
                            str(data_id), ha='center', va='center', 
                            fontsize=12, fontweight='bold')
                    
                    # Thread position label
                    thread_pos = f"({row},{col})"
                    ax.text(warp_x + col + 0.5, warp_y + row + 0.3, 
                            thread_pos, ha='center', va='top', fontsize=6, alpha=0.5)
    
    # Add row and column indicators
    ax.text(-0.5, 6, "Row 0", fontsize=10, ha='center', rotation=90)
    ax.text(-0.5, 2, "Row 1", fontsize=10, ha='center', rotation=90)
    ax.text(2, 9, "Column 0", fontsize=10, ha='center')
    ax.text(6, 9, "Column 1", fontsize=10, ha='center')
    
    # Add replication indicators based on pattern type
    if "Pattern 5" in pattern_name:
        # No replication pattern - don't add arrows, each thread has unique data
        pass
    elif "Row-Warp" in pattern_name:
        # Row-wise warp replication
        ax.annotate('', xy=(4, 6), xytext=(0, 6), 
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.annotate('', xy=(4, 2), xytext=(0, 2), 
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(2, 6.5, "Same data", color='red', ha='center', fontsize=10)
        ax.text(2, 2.5, "Same data", color='red', ha='center', fontsize=10)
    else:  # Column-wise warp replication
        # Column-wise warp replication
        ax.annotate('', xy=(2, 4), xytext=(2, 0), 
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.annotate('', xy=(6, 4), xytext=(6, 0), 
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(1.5, 2, "Same\ndata", color='red', ha='center', fontsize=10)
        ax.text(6.5, 2, "Same\ndata", color='red', ha='center', fontsize=10)
    
    # Add thread-level replication indicators (except for Pattern 5)
    if "Pattern 5" not in pattern_name:
        for warp_idx, (warp_x, warp_y) in enumerate(positions):
            if "Row-Thread" in pattern_name:
                # Row-wise thread replication (within each row)
                for row in range(4):
                    ax.annotate('', xy=(warp_x + 1, warp_y + row + 0.5), 
                            xytext=(warp_x, warp_y + row + 0.5),
                            arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
            else:  # Column-wise thread replication
                # Column-wise thread replication (within each column)
                for col in range(2):
                    # Connect rows 0-1, 1-2, 2-3 within each column
                    for row in range(3):
                        ax.annotate('', xy=(warp_x + col + 0.5, warp_y + row + 1.5),
                                xytext=(warp_x + col + 0.5, warp_y + row + 0.5),
                                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    
    return ax

if __name__ == '__main__':
    # Test all four original replication patterns
    pattern1_data = test_warp_row_thread_row()
    pattern2_data = test_warp_col_thread_row()
    pattern3_data = test_warp_row_thread_col()
    pattern4_data = test_warp_col_thread_col()

    # Create Pattern 5: No replication
    pattern5_structure = create_test_structure()
    
    # Provide minimal encoding with empty RsLengths to test no replication
    pattern5_structure['encoding'] = {
        'Ps2RHssMajor': [[0], [0]],  # Pointing to R dimensions but with no replication
        'Ps2RHssMinor': [[0], [1]]
    }
    pattern5_structure['rs_lengths'] = []
    
    # Calculate data IDs for pattern 5
    pattern5_data_ids = calculate_data_ids(pattern5_structure, pattern5_structure['encoding'], pattern5_structure['rs_lengths'])
    pattern5_data = (pattern5_data_ids, "Pattern 5: No Replication")
    
    # Store all results
    all_patterns = [pattern1_data, pattern2_data, pattern3_data, pattern4_data, pattern5_data]
    pattern_results = []
    
    # Verify each pattern
    for i, (data_ids, pattern_type) in enumerate(all_patterns):
        if i < 4:  # Original patterns 1-4
            success = verify_replication_pattern(data_ids, pattern_type)
            pattern_results.append(success)
        else:  # Pattern 5 (no replication)
            # Verify all data IDs are unique
            unique_ids = set(data_ids.values())
            all_unique = len(unique_ids) == len(data_ids)
            print(f"\n=== Testing {pattern_type} ===")
            print(f"All threads have unique data IDs: {all_unique}")
            pattern_results.append(all_unique)
    
    # Create combined visualization with all 5 patterns
    create_comparison_visualization(all_patterns)
    
    # Update visualizer if needed based on test results
    if not all(pattern_results):
        update_visualizer()
    
    # Print summary
    print("\n=== Summary ===")
    for i, (_, pattern_type) in enumerate(all_patterns):
        print(f"{pattern_type} test: {'PASSED' if pattern_results[i] else 'FAILED'}")
    
    # Open the image if on a compatible system
    try:
        os.system("display all_replication_patterns.png &")
    except:
        pass 