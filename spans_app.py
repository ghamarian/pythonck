#!/usr/bin/env python3
"""
Streamlit application for visualizing spans in Composable Kernels tile distributions.

This application allows users to:
1. Load examples from examples.py
2. Edit tile_distribution_encoding code
3. Parse and visualize span calculations
4. See detailed span information
"""

import streamlit as st
import sys
import os
import traceback
import json
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.append('.')

try:
    from parser import TileDistributionParser, debug_indexing_relationships
    from examples import get_examples, get_default_variables
    from tiler_pedantic import TileDistributionPedantic
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

def generate_step_by_step_explanation(tile_dist: TileDistributionPedantic) -> str:
    """
    Generate a detailed step-by-step explanation of how spans are calculated,
    similar to the manual analysis provided in the examples.
    """
    explanation = []
    detail = tile_dist.DstrEncode.detail
    
    explanation.append("=== STEP-BY-STEP SPANS ANALYSIS ===\n")
    
    # Step 1: Understand Y mappings
    explanation.append("STEP 1: Understand Y mappings")
    explanation.append("Each Y dimension maps to a specific position in R or H sequences:\n")
    
    for i in range(tile_dist.NDimYs):
        rh_major = tile_dist.DstrEncode.Ys2RHsMajor[i]
        rh_minor = tile_dist.DstrEncode.Ys2RHsMinor[i]
        
        if rh_major == 0:
            source = f'R[{rh_minor}]'
            value = tile_dist.DstrEncode.RsLengths[rh_minor] if rh_minor < len(tile_dist.DstrEncode.RsLengths) else 'N/A'
        else:
            h_idx = rh_major - 1
            source = f'H{h_idx}[{rh_minor}]'
            value = tile_dist.DstrEncode.HsLengthss[h_idx][rh_minor] if rh_minor < len(tile_dist.DstrEncode.HsLengthss[h_idx]) else 'N/A'
        
        explanation.append(f'Y{i}: rh_major={rh_major}, rh_minor={rh_minor} → {source} = {value}')
    
    explanation.append("")
    
    # Step 2: Group Y dimensions by H sequence
    explanation.append("STEP 2: Group Y dimensions by H sequence")
    
    # Group Ys by H sequence
    h_groups = {}
    r_ys = []
    
    for i in range(tile_dist.NDimYs):
        rh_major = tile_dist.DstrEncode.Ys2RHsMajor[i]
        rh_minor = tile_dist.DstrEncode.Ys2RHsMinor[i]
        
        if rh_major == 0:  # R sequence
            r_ys.append((i, rh_minor))
        else:  # H sequence
            h_idx = rh_major - 1
            if h_idx not in h_groups:
                h_groups[h_idx] = []
            h_groups[h_idx].append((i, rh_minor))
    
    if r_ys:
        explanation.append(f'R (rh_major=0): {[f"Y{y_idx}→R[{rh_minor}]" for y_idx, rh_minor in r_ys]}')
    
    for h_idx in sorted(h_groups.keys()):
        ys_in_h = h_groups[h_idx]
        explanation.append(f'H{h_idx} (rh_major={h_idx+1}): {[f"Y{y_idx}→H{h_idx}[{rh_minor}]" for y_idx, rh_minor in ys_in_h]}')
    
    explanation.append("")
    
    # Step 3: Key insight about span_minor assignment
    explanation.append("STEP 3: Assign span_minor in ORDER OF Y DIMENSION INDEX")
    explanation.append("🔑 KEY INSIGHT: span_minor is assigned based on Y dimension order, NOT rh_minor position!")
    explanation.append("")
    
    # Step 4: Show span_minor assignment process
    explanation.append("STEP 4: Assign span_minor sequentially within each H sequence")
    explanation.append("")
    
    for h_idx in sorted(h_groups.keys()):
        ys_in_h = sorted(h_groups[h_idx])  # Sort by Y index
        explanation.append(f'For H{h_idx}:')
        for span_minor, (y_idx, rh_minor) in enumerate(ys_in_h):
            explanation.append(f'  Y{y_idx} (at H{h_idx}[{rh_minor}]) gets span_minor = {span_minor}')
        explanation.append("")
    
    # Step 5: Build rhs_major_minor_to_span_minor_ table
    explanation.append("STEP 5: Build rhs_major_minor_to_span_minor_ table")
    explanation.append("This table maps: (rh_major, rh_minor) → span_minor")
    explanation.append("")
    
    rhs_major_minor_to_span_minor = detail['rhs_major_minor_to_span_minor_']
    rhs_major_minor_to_ys = detail['rhs_major_minor_to_ys_']
    
    explanation.append("Manual construction:")
    for h_idx in sorted(h_groups.keys()):
        ys_in_h = sorted(h_groups[h_idx])
        explanation.append(f'H{h_idx} (rh_major={h_idx+1}):')
        for span_minor, (y_idx, rh_minor) in enumerate(ys_in_h):
            explanation.append(f'  - Y{y_idx} is at H{h_idx}[{rh_minor}] and gets span_minor={span_minor} → table[{h_idx+1}][{rh_minor}] = {span_minor}')
        explanation.append("")
    
    explanation.append("Actual table:")
    for rh_major in range(len(rhs_major_minor_to_span_minor)):
        section_name = 'R' if rh_major == 0 else f'H{rh_major-1}'
        explanation.append(f'{section_name} (rh_major={rh_major}): {rhs_major_minor_to_span_minor[rh_major]}')
        
        for rh_minor in range(len(rhs_major_minor_to_span_minor[rh_major])):
            span_minor = rhs_major_minor_to_span_minor[rh_major][rh_minor]
            y_idx = rhs_major_minor_to_ys[rh_major][rh_minor]
            
            if span_minor >= 0:
                explanation.append(f'  Position [{rh_major}][{rh_minor}]: span_minor={span_minor} (Y{y_idx})')
    
    explanation.append("")
    
    # Step 6: Verify with ys_to_span_minor_
    explanation.append("STEP 6: Verify with ys_to_span_minor_")
    ys_to_span_minor = detail['ys_to_span_minor_']
    explanation.append(f'ys_to_span_minor_: {ys_to_span_minor}')
    for i in range(tile_dist.NDimYs):
        span_minor = ys_to_span_minor[i]
        explanation.append(f'Y{i}: span_minor = {span_minor}')
    
    explanation.append("")
    
    # Step 7: NEW - Building distributed_spans_lengthss_
    explanation.append("STEP 7: Building distributed_spans_lengthss_")
    explanation.append("Now we create the final spans structure using the span_minor mappings:")
    explanation.append("")
    
    max_span_minor = detail['max_ndim_span_minor_']
    explanation.append(f"max_ndim_span_minor_ = {max_span_minor}")
    explanation.append("")
    
    explanation.append("ALGORITHM (from tiler_pedantic.py lines 219-235):")
    explanation.append("```python")
    explanation.append("# Initialize 2D array with -1")
    explanation.append(f"distributed_spans_lengthss_tmp = [[-1] * {max_span_minor} for _ in range({tile_dist.NDimX})]")
    explanation.append("")
    explanation.append("# For each Y dimension:")
    explanation.append(f"for i in range({tile_dist.NDimYs}):  # Y0, Y1, Y2, ...")
    explanation.append("    rh_major = Ys2RHsMajor[i]")
    explanation.append("    rh_minor = Ys2RHsMinor[i]")
    explanation.append("    h_sequence_idx = rh_major - 1  # Convert to 0-based H index")
    explanation.append("")
    explanation.append("    if 0 <= h_sequence_idx < NDimX:  # Only H-space (skip R-space)")
    explanation.append("        h_length = HsLengthss[h_sequence_idx][rh_minor]")
    explanation.append("        span_major = h_sequence_idx")
    explanation.append("        span_minor = rhs_major_minor_to_span_minor_[rh_major][rh_minor]  # KEY LOOKUP!")
    explanation.append("")
    explanation.append("        if span_minor != -1:  # Valid span position")
    explanation.append("            distributed_spans_lengthss_tmp[span_major][span_minor] = h_length")
    explanation.append("```")
    explanation.append("")
    
    explanation.append("Step-by-step execution:")
    distributed_spans = detail['distributed_spans_lengthss_']
    rhs_major_minor_to_span_minor = detail['rhs_major_minor_to_span_minor_']
    
    for i in range(tile_dist.NDimYs):
        rh_major = tile_dist.DstrEncode.Ys2RHsMajor[i]
        rh_minor = tile_dist.DstrEncode.Ys2RHsMinor[i]
        
        explanation.append(f"")
        explanation.append(f"Y{i}: rh_major={rh_major}, rh_minor={rh_minor}")
        
        if rh_major == 0:
            explanation.append(f"  → R-space (rh_major=0), SKIP (not included in spans)")
            continue
            
        h_sequence_idx = rh_major - 1
        if 0 <= h_sequence_idx < tile_dist.NDimX:
            h_length = tile_dist.DstrEncode.HsLengthss[h_sequence_idx][rh_minor]
            span_major = h_sequence_idx
            span_minor = rhs_major_minor_to_span_minor[rh_major][rh_minor]
            
            explanation.append(f"  → H{h_sequence_idx}[{rh_minor}], length={h_length}")
            explanation.append(f"  → span_major = {span_major} (which H sequence)")
            explanation.append(f"  → span_minor = rhs_major_minor_to_span_minor_[{rh_major}][{rh_minor}] = {span_minor}")
            
            if span_minor != -1:
                explanation.append(f"  → distributed_spans_lengthss_[{span_major}][{span_minor}] = {h_length}")
            else:
                explanation.append(f"  → span_minor = -1, SKIP (invalid position)")
        else:
            explanation.append(f"  → Invalid h_sequence_idx = {h_sequence_idx}, SKIP")
    
    explanation.append("")
    explanation.append("Final result:")
    for h_idx in range(len(distributed_spans)):
        explanation.append(f"  H{h_idx}: {distributed_spans[h_idx]}")
    
    explanation.append("")
    explanation.append("KEY ALGORITHM INSIGHTS:")
    explanation.append("1. **Uses rhs_major_minor_to_span_minor_ for lookup**: Not ys_to_span_minor_!")
    explanation.append("2. **Direct position mapping**: (rh_major, rh_minor) → span_minor")
    explanation.append("3. **Skips R-space**: Only H-space Y dimensions contribute")
    explanation.append("4. **Handles gaps**: -1 span_minor values are skipped")
    explanation.append("5. **Preserves compaction**: Uses compacted span_minor indices")
    explanation.append("")
    
    if h_groups:
        explanation.append("Example from this encoding:")
        for h_idx in sorted(h_groups.keys()):
            ys_in_h = sorted(h_groups[h_idx])
            for span_minor, (y_idx, rh_minor) in enumerate(ys_in_h):
                explanation.append(f'- Y{y_idx} maps to H{h_idx}[{rh_minor}] → Y{y_idx} gets span_minor={span_minor} (#{span_minor+1} Y in H{h_idx})')
    
    explanation.append("")
    explanation.append("=== THREAD USAGE ===")
    explanation.append("How threads use this structure:")
    explanation.append("")
    explanation.append("constexpr auto spans = tile.get_distributed_spans();")
    explanation.append("")
    explanation.append("sweep_tile_span(spans[0], [&](auto idx0) {      // H0 iteration")
    explanation.append("    sweep_tile_span(spans[1], [&](auto idx1) {  // H1 iteration")
    explanation.append("        // Process element at (idx0, idx1)")
    explanation.append("        y_tile(idx0, idx1) = x_tile[idx0, idx1] * 2 + (idx0 + idx1);")
    explanation.append("    });")
    explanation.append("});")
    explanation.append("")
    
    explanation.append("Key points:")
    explanation.append("- Only H-space Y dimensions contribute (R-space Ys are excluded)")
    explanation.append("- Order is determined by H sequence grouping and Y dimension order")
    explanation.append("- Lower H index within same H sequence gets lower span_minor")
    explanation.append("- -1 values indicate unused span positions (skipped during iteration)")
    
    # Step 8: Final distributed_spans_lengthss_ (original step 7)
    explanation.append("STEP 8: Final distributed_spans_lengthss_")
    explanation.append("distributed_spans_lengthss_:")
    for span_major in range(len(distributed_spans)):
        explanation.append(f'H{span_major}: {distributed_spans[span_major]}')
        for span_minor in range(len(distributed_spans[span_major])):
            length = distributed_spans[span_major][span_minor]
            if length >= 0:
                # Find which Y maps to this span position
                y_idx = -1
                for i in range(tile_dist.NDimYs):
                    if detail['ys_to_span_major_'][i] == span_major and detail['ys_to_span_minor_'][i] == span_minor:
                        y_idx = i
                        break
                explanation.append(f'  [H{span_major}][span_minor={span_minor}] = {length} (Y{y_idx})')
    
    explanation.append("")
    
    # Final summary
    explanation.append("=== KEY INSIGHTS SUMMARY ===")
    explanation.append("1. span_minor is NOT assigned by rh_minor position!")
    explanation.append("2. span_minor is assigned by Y DIMENSION ORDER within each H sequence!")
    explanation.append("3. The rhs_major_minor_to_span_minor_ table maps physical positions to logical spans")
    explanation.append("4. distributed_spans_lengthss_ is built by:")
    explanation.append("   - Grouping by H sequence (H0, H1, ...)")
    explanation.append("   - Using span_minor as the index within each H")
    explanation.append("   - Filling with lengths from HsLengthss")
    explanation.append("5. **ALGORITHM**: Uses rhs_major_minor_to_span_minor_ for efficient lookup")
    explanation.append("")
    
    return "\n".join(explanation)

def main():
    st.set_page_config(
        page_title="Composable Kernels Spans Visualizer",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Composable Kernels Spans Visualizer")
    st.markdown("Interactive tool for understanding spans in tile distribution encodings")
    
    # Sidebar for example selection and variables
    st.sidebar.header("Configuration")
    
    # Load examples
    examples = get_examples()
    example_names = list(examples.keys())
    
    selected_example = st.sidebar.selectbox(
        "Select Example",
        example_names,
        index=0
    )
    
    # Load default variables for the selected example
    default_vars = get_default_variables(selected_example)
    
    # Variable editor
    st.sidebar.subheader("Variables")
    variables = {}
    
    if default_vars:
        for var_name, default_value in default_vars.items():
            variables[var_name] = st.sidebar.number_input(
                f"{var_name}",
                value=default_value,
                min_value=1,
                step=1
            )
    else:
        st.sidebar.info("No variables needed for this example")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Code Editor")
        
        # Load the selected example code
        initial_code = examples[selected_example]
        
        # Code editor
        edited_code = st.text_area(
            "tile_distribution_encoding",
            value=initial_code,
            height=300,
            help="Edit the tile_distribution_encoding C++ code"
        )
        
        # Parse button
        if st.button("🔄 Parse & Analyze", type="primary"):
            try:
                # Parse the code
                parser = TileDistributionParser()
                parsed_encoding = parser.parse_tile_distribution_encoding(edited_code)
                
                # Create tile distribution
                tile_dist = TileDistributionPedantic(parsed_encoding, variables)
                tile_dist.set_source_code(edited_code)
                
                # Store in session state
                st.session_state.tile_dist = tile_dist
                st.session_state.parsed_encoding = parsed_encoding
                st.session_state.current_code = edited_code
                
                st.success("✅ Parsing successful!")
                
            except Exception as e:
                st.error(f"❌ Parsing failed: {str(e)}")
                st.code(traceback.format_exc())
    
    with col2:
        st.subheader("📊 Basic Information")
        
        if hasattr(st.session_state, 'tile_dist'):
            tile_dist = st.session_state.tile_dist
            
            # Basic dimensions
            col2a, col2b, col2c, col2d = st.columns(4)
            with col2a:
                st.metric("NDimX", tile_dist.NDimX)
            with col2b:
                st.metric("NDimP", tile_dist.NDimPs)
            with col2c:
                st.metric("NDimY", tile_dist.NDimYs)
            with col2d:
                st.metric("NDimR", tile_dist.NDimRs)
            
            # Resolved sequences
            st.subheader("🔗 Resolved Sequences")
            
            col3a, col3b = st.columns(2)
            with col3a:
                st.write("**RsLengths:**", tile_dist.DstrEncode.RsLengths)
            with col3b:
                st.write("**HsLengthss:**")
                for i, h_seq in enumerate(tile_dist.DstrEncode.HsLengthss):
                    st.write(f"  H{i}: {h_seq}")
            
            # Y dimension mappings
            st.subheader("🎯 Y Dimension Mappings")
            
            mapping_data = []
            for i in range(tile_dist.NDimYs):
                rh_major = tile_dist.DstrEncode.Ys2RHsMajor[i]
                rh_minor = tile_dist.DstrEncode.Ys2RHsMinor[i]
                
                if rh_major == 0:
                    source = f"R[{rh_minor}]"
                    length = tile_dist.DstrEncode.RsLengths[rh_minor] if rh_minor < len(tile_dist.DstrEncode.RsLengths) else "N/A"
                else:
                    h_idx = rh_major - 1
                    source = f"H{h_idx}[{rh_minor}]"
                    length = tile_dist.DstrEncode.HsLengthss[h_idx][rh_minor] if rh_minor < len(tile_dist.DstrEncode.HsLengthss[h_idx]) else "N/A"
                
                mapping_data.append({
                    "Y Dimension": f"Y{i}",
                    "RH Major": rh_major,
                    "RH Minor": rh_minor,
                    "Source": source,
                    "Length": length
                })
            
            st.table(mapping_data)
        else:
            st.info("👆 Parse an example above to see the analysis")
    
    # Detailed analysis tabs
    if hasattr(st.session_state, 'tile_dist'):
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Step-by-Step Explanation", 
            "🔍 Spans Calculations", 
            "📊 Distributed Spans", 
            "🧵 P Dimension Analysis",
            "🗺️ PsYs→Xs Adaptor",
            "🔧 Encoding Details", 
            "🔍 Index Mapping"
        ])
        
        with tab1:
            st.subheader("📋 Step-by-Step Spans Explanation")
            st.markdown("This section provides a detailed walkthrough of how spans are calculated:")
            
            explanation = generate_step_by_step_explanation(st.session_state.tile_dist)
            st.code(explanation, language="text")
        
        with tab2:
            st.subheader("🔍 Spans Calculations")
            
            detail = st.session_state.tile_dist.DstrEncode.detail
            
            # Spans major/minor mappings
            col4a, col4b = st.columns(2)
            
            with col4a:
                st.write("**ys_to_span_major_:**", detail['ys_to_span_major_'])
                st.write("**ys_to_span_minor_:**", detail['ys_to_span_minor_'])
                st.write("**ndims_span_minor_:**", detail['ndims_span_minor_'])
                st.write("**max_ndim_span_minor_:**", detail['max_ndim_span_minor_'])
            
            with col4b:
                st.write("**ndims_distributed_spans_minor_:**", detail['ndims_distributed_spans_minor_'])
                st.write("**is_ys_from_r_span_:**", detail['is_ys_from_r_span_'])
            
            # RHS mappings tables
            st.subheader("🗂️ RHS Mapping Tables")
            
            col5a, col5b = st.columns(2)
            
            with col5a:
                st.write("**rhs_major_minor_to_ys_:**")
                rhs_to_ys = detail['rhs_major_minor_to_ys_']
                for i, row in enumerate(rhs_to_ys):
                    section_name = 'R' if i == 0 else f'H{i-1}'
                    st.write(f"  {section_name} (Row {i}): {row}")
            
            with col5b:
                st.write("**rhs_major_minor_to_span_minor_:**")
                rhs_to_span = detail['rhs_major_minor_to_span_minor_']
                for i, row in enumerate(rhs_to_span):
                    section_name = 'R' if i == 0 else f'H{i-1}'
                    st.write(f"  {section_name} (Row {i}): {row}")
        
        with tab3:
            st.subheader("📊 Distributed Spans Structure")
            
            distributed_spans = detail['distributed_spans_lengthss_']
            
            st.write("**distributed_spans_lengthss_:**")
            for span_major in range(len(distributed_spans)):
                st.write(f"  H{span_major} (Row {span_major}): {distributed_spans[span_major]}")
                
                # Show which Y maps to each position
                for span_minor in range(len(distributed_spans[span_major])):
                    length = distributed_spans[span_major][span_minor]
                    if length >= 0:
                        # Find which Y maps to this span position
                        y_idx = -1
                        for i in range(st.session_state.tile_dist.NDimYs):
                            if (detail['ys_to_span_major_'][i] == span_major and 
                                detail['ys_to_span_minor_'][i] == span_minor):
                                y_idx = i
                                break
                        st.write(f"    [H{span_major}][span_minor={span_minor}] = {length} (Y{y_idx})")
        
        with tab4:
            st.subheader("🧵 P Dimension Analysis")
            st.markdown("**P dimensions handle thread/warp distribution and replication, while Y dimensions handle data organization within tiles.**")
            
            # Add helpful note about derivatives
            st.info("💡 **Tip**: To see P→R derivatives in action, try the **'R Sequence with Variables'** example which has P dimensions that map directly to R dimensions!")
            
            detail = st.session_state.tile_dist.DstrEncode.detail
            
            if st.session_state.tile_dist.NDimPs > 0:
                # P vs Y Comparison
                st.subheader("🔄 P vs Y Dimensions Comparison")
                
                col_p, col_y = st.columns(2)
                
                with col_p:
                    st.write("**🧵 P Dimensions (Parallelization)**")
                    st.write("- Handle thread/warp distribution")
                    st.write("- Control which thread gets which data")
                    st.write("- Manage replication across threads")
                    st.write("- Used for memory access patterns")
                    st.write(f"- Count: {st.session_state.tile_dist.NDimPs}")
                
                with col_y:
                    st.write("**📊 Y Dimensions (Data Organization)**")
                    st.write("- Handle data layout within tiles")
                    st.write("- Control spans structure")
                    st.write("- Manage within-thread indexing")
                    st.write("- Used for logical data organization")
                    st.write(f"- Count: {st.session_state.tile_dist.NDimYs}")
                
                # P Dimension Mappings
                st.subheader("🗺️ P Dimension Mappings")
                st.write("Shows how each P dimension maps to R/H components:")
                
                p_mapping_data = []
                for i in range(st.session_state.tile_dist.NDimPs):
                    p_major_list = st.session_state.tile_dist.DstrEncode.Ps2RHssMajor[i]
                    p_minor_list = st.session_state.tile_dist.DstrEncode.Ps2RHssMinor[i]
                    
                    # Handle both list and scalar cases
                    if not isinstance(p_major_list, list):
                        p_major_list = [p_major_list]
                    if not isinstance(p_minor_list, list):
                        p_minor_list = [p_minor_list]
                    
                    mappings_detail = []
                    for j, (p_major, p_minor) in enumerate(zip(p_major_list, p_minor_list)):
                        if p_major == 0:
                            source = f"R[{p_minor}]"
                            length = st.session_state.tile_dist.DstrEncode.RsLengths[p_minor] if p_minor < len(st.session_state.tile_dist.DstrEncode.RsLengths) else "N/A"
                        else:
                            h_idx = p_major - 1
                            source = f"H{h_idx}[{p_minor}]"
                            length = st.session_state.tile_dist.DstrEncode.HsLengthss[h_idx][p_minor] if p_minor < len(st.session_state.tile_dist.DstrEncode.HsLengthss[h_idx]) else "N/A"
                        
                        mappings_detail.append(f"{source}={length}")
                    
                    p_mapping_data.append({
                        "P Dimension": f"P{i}",
                        "Component Mappings": ", ".join(mappings_detail),
                        "Major Indices": str(p_major_list),
                        "Minor Indices": str(p_minor_list),
                        "Purpose": "Thread Distribution" if i == 0 else "Warp Distribution" if i == 1 else f"Level {i} Distribution"
                    })
                
                st.table(p_mapping_data)
                
                # R Dimension Ownership
                if st.session_state.tile_dist.NDimRs > 0:
                    st.subheader("🔒 R Dimension Ownership")
                    st.write("Shows which P dimensions control which R dimensions:")
                    
                    does_p_own_r = detail['does_p_own_r_']
                    
                    ownership_data = []
                    for p_idx in range(st.session_state.tile_dist.NDimPs):
                        owned_rs = []
                        for r_idx in range(st.session_state.tile_dist.NDimRs):
                            if does_p_own_r[p_idx][r_idx]:
                                r_length = st.session_state.tile_dist.DstrEncode.RsLengths[r_idx]
                                owned_rs.append(f"R{r_idx}({r_length})")
                        
                        ownership_data.append({
                            "P Dimension": f"P{p_idx}",
                            "Owns R Dimensions": ", ".join(owned_rs) if owned_rs else "None",
                            "Replication Role": "Controls replication" if owned_rs else "No replication control"
                        })
                    
                    st.table(ownership_data)
                
                # P-to-R Derivatives
                if st.session_state.tile_dist.NDimRs > 0:
                    st.subheader("📈 P-to-R Index Derivatives")
                    st.write("Shows how P indices mathematically map to R indices:")
                    
                    ps_over_rs_derivative = detail['ps_over_rs_derivative_']
                    
                    # Check if any derivatives are non-zero
                    has_nonzero_derivatives = any(
                        ps_over_rs_derivative[p][r] != 0 
                        for p in range(st.session_state.tile_dist.NDimPs) 
                        for r in range(st.session_state.tile_dist.NDimRs)
                    )
                    
                    if has_nonzero_derivatives:
                        st.success("✅ This encoding has P→R mappings (non-zero derivatives)")
                    else:
                        st.info("ℹ️ This encoding has no direct P→R mappings (all derivatives are zero)")
                        st.write("**Why derivatives are zero:**")
                        st.write("- P dimensions only map to H dimensions, not R dimensions")
                        st.write("- R dimensions are replicated but not controlled by P indices")
                        st.write("- Try 'R Sequence with Variables' example to see non-zero derivatives")
                    
                    # Create a matrix visualization
                    st.write("**Derivative Matrix: `R_index = sum(P_index[i] * derivative[i][r])`**")
                    
                    derivative_matrix = []
                    header = ["P Dimension"] + [f"→ R{r}" for r in range(st.session_state.tile_dist.NDimRs)]
                    
                    for p_idx in range(st.session_state.tile_dist.NDimPs):
                        row = [f"P{p_idx}"]
                        for r_idx in range(st.session_state.tile_dist.NDimRs):
                            derivative_val = ps_over_rs_derivative[p_idx][r_idx]
                            if derivative_val == 0:
                                row.append("0 (no mapping)")
                            else:
                                row.append(f"**{derivative_val}**")
                        derivative_matrix.append(dict(zip(header, row)))
                    
                    st.table(derivative_matrix)
                    
                    # Example calculation
                    st.write("**Example Calculation:**")
                    if has_nonzero_derivatives:
                        st.code(f"""
# If P indices are [p0, p1, ...], then R indices are calculated as:
for r_dim in range({st.session_state.tile_dist.NDimRs}):
    r_index[r_dim] = 0
    for p_dim in range({st.session_state.tile_dist.NDimPs}):
        r_index[r_dim] += p_index[p_dim] * derivative[p_dim][r_dim]

# Example with current derivative matrix:
{f"# R0 = P0 * {ps_over_rs_derivative[0][0]} + P1 * {ps_over_rs_derivative[1][0] if st.session_state.tile_dist.NDimPs > 1 else 0}" if st.session_state.tile_dist.NDimRs > 0 else ""}
{f"# R1 = P0 * {ps_over_rs_derivative[0][1] if st.session_state.tile_dist.NDimRs > 1 else 0} + P1 * {ps_over_rs_derivative[1][1] if st.session_state.tile_dist.NDimPs > 1 and st.session_state.tile_dist.NDimRs > 1 else 0}" if st.session_state.tile_dist.NDimRs > 1 else ""}

# In this case: {"All R indices will be 0 since all derivatives are 0" if not has_nonzero_derivatives else "P indices directly control R indices"}
                        """, language="python")
                    else:
                        st.code(f"""
# Since all derivatives are 0, R indices are always 0:
for r_dim in range({st.session_state.tile_dist.NDimRs}):
    r_index[r_dim] = 0  # No P dimension controls this R dimension

# This means:
# - All threads access the same R-space elements
# - R dimensions are replicated across all threads
# - P dimensions only affect H-space (data layout within tiles)
                        """, language="python")
                else:
                    st.info("No R dimensions in this encoding - derivatives not applicable")
                
                # Thread Distribution Visualization
                st.subheader("🎯 Thread Distribution Pattern")
                
                # Determine thread pattern based on NDimP
                if st.session_state.tile_dist.NDimPs == 1:
                    st.write("**Single-Level Distribution (NDimP = 1):**")
                    st.write("- P0 typically controls thread-level distribution")
                    st.write("- Each thread gets a different P0 index (lane_id)")
                    st.write("- Used for warp-level parallelism")
                    
                    # Show example thread mapping
                    st.write("**Example Thread Mapping:**")
                    example_threads = []
                    for thread_id in range(min(8, 32)):  # Show first 8 threads
                        p0_val = thread_id  # Simplified
                        example_threads.append({
                            "Thread ID": thread_id,
                            "P0 Index": p0_val,
                            "Role": f"Handles data slice {p0_val}"
                        })
                    st.table(example_threads[:8])
                    
                elif st.session_state.tile_dist.NDimPs == 2:
                    st.write("**Two-Level Distribution (NDimP = 2):**")
                    st.write("- P0 typically controls warp-level distribution")
                    st.write("- P1 typically controls thread-level distribution")
                    st.write("- Used for block-level parallelism")
                    
                    # Show example warp/thread mapping
                    st.write("**Example Warp/Thread Mapping:**")
                    example_mapping = []
                    for warp_id in range(min(4, 8)):  # Show first 4 warps
                        for thread_in_warp in range(min(4, 32)):  # Show first 4 threads per warp
                            example_mapping.append({
                                "Warp ID": warp_id,
                                "Thread in Warp": thread_in_warp,
                                "P0 Index": warp_id,
                                "P1 Index": thread_in_warp,
                                "Global Thread": warp_id * 32 + thread_in_warp
                            })
                    st.table(example_mapping[:16])  # Show first 16 entries
                
                # P and Y Interaction
                st.subheader("🔗 How P and Y Work Together")
                
                st.write("**Complete Index Calculation Process:**")
                st.code(f"""
1. **P Dimensions (Thread/Warp Assignment):**
   - P indices determine which thread/warp you are
   - P{0 if st.session_state.tile_dist.NDimPs > 0 else 'X'} = {f"warp_id() or lane_id()" if st.session_state.tile_dist.NDimPs > 0 else "N/A"}
   {f"- P1 = lane_id()" if st.session_state.tile_dist.NDimPs > 1 else ""}

2. **Y Dimensions (Within-Tile Organization):**
   - Y indices determine position within your tile
   - Y dimensions create the spans structure
   - Each thread iterates through Y indices

3. **Combined Calculation:**
   final_index = calculate_index(P_indices, Y_indices)
   
   This uses:
   - PsYs2XsAdaptor: Maps (P,Y) → X coordinates  
   - Ys2DDescriptor: Maps Y → D (distributed index)
   - Final: D * X_stride + X_offset
                """, language="text")
                
                # Show actual example with current encoding
                st.write("**Example with Current Encoding:**")
                
                # Create a simple example
                example_calc = []
                max_examples = 6
                
                for p_example in range(min(2, max_examples)):
                    for y_example in range(min(3, max_examples - p_example * 3)):
                        if st.session_state.tile_dist.NDimPs == 1:
                            p_coords = [p_example]
                        elif st.session_state.tile_dist.NDimPs == 2:
                            p_coords = [p_example // 2, p_example % 2]
                        else:
                            p_coords = [p_example]
                        
                        if st.session_state.tile_dist.NDimYs > 0:
                            y_coords = [y_example] + [0] * (st.session_state.tile_dist.NDimYs - 1)
                        else:
                            y_coords = []
                        
                        try:
                            final_idx = st.session_state.tile_dist.calculate_index(p_coords, y_coords)
                            example_calc.append({
                                "P Coordinates": str(p_coords),
                                "Y Coordinates": str(y_coords),
                                "Final Index": final_idx,
                                "Meaning": f"Thread {p_coords} accesses element {y_coords} → memory[{final_idx}]"
                            })
                        except Exception as e:
                            example_calc.append({
                                "P Coordinates": str(p_coords),
                                "Y Coordinates": str(y_coords),
                                "Final Index": "Error",
                                "Meaning": f"Calculation failed: {str(e)[:50]}..."
                            })
                        
                        if len(example_calc) >= max_examples:
                            break
                    if len(example_calc) >= max_examples:
                        break
                
                st.table(example_calc)
                
            else:
                st.info("No P dimensions in this encoding (NDimP = 0)")
                st.write("This encoding only uses Y dimensions for data organization.")
                st.write("All threads would access the same data pattern, differentiated only by Y indices.")
        
        with tab5:
            st.subheader("🗺️ PsYs→Xs Adaptor")
            
            st.markdown("""
            **The PsYs→Xs Adaptor is the core coordinate transformation that maps:**
            - **Input (Top View)**: `[P0, P1, ..., Y0, Y1, ...]` - Logical partition and within-tile coordinates
            - **Output (Bottom View)**: `[X0, X1, ...]` - Physical tensor coordinates
            
            This adaptor **IS** the implementation of your tile distribution encoding!
            """)
            
            # Get the adaptor from the tile distribution
            adaptor = st.session_state.tile_dist.PsYs2XsAdaptor
            
            # Top View (Input Space)
            st.subheader("📥 Top View (Input Space)")
            st.write("**Logical coordinates that threads provide:**")
            
            top_view = adaptor['TopView']
            top_id_to_name = top_view['TopDimensionIdToName']
            top_name_lengths = top_view['TopDimensionNameLengths']
            effective_order = top_view['_effective_display_order_ids_']
            
            # Create a table showing the top view structure
            top_view_data = []
            for i, hid in enumerate(effective_order):
                name = top_id_to_name.get(str(hid), top_id_to_name.get(hid, f"Hidden{hid}"))
                length = top_name_lengths.get(name, "Unknown")
                
                if name.startswith("P"):
                    coord_type = "Partition (Thread/Warp ID)"
                    source = "get_warp_id() or get_lane_id()"
                elif name.startswith("Y"):
                    coord_type = "Within-tile position"
                    source = "Iteration variable"
                else:
                    coord_type = "Hidden coordinate"
                    source = "Internal"
                
                top_view_data.append({
                    "Position": i,
                    "Hidden ID": hid,
                    "Name": name,
                    "Length": length,
                    "Type": coord_type,
                    "Source": source
                })
            
            st.table(top_view_data)
            
            # Bottom View (Output Space)
            st.subheader("📤 Bottom View (Output Space)")
            st.write("**Physical tensor coordinates for data access:**")
            
            bottom_view = adaptor['BottomView']
            bottom_id_to_name = bottom_view['BottomDimensionIdToName']
            bottom_name_lengths = bottom_view['BottomDimensionNameLengths']
            
            bottom_view_data = []
            for dim_id in sorted([int(k) for k in bottom_id_to_name.keys()]):
                # Try both string and integer keys to handle different formats
                name = None
                if str(dim_id) in bottom_id_to_name:
                    name = bottom_id_to_name[str(dim_id)]
                elif dim_id in bottom_id_to_name:
                    name = bottom_id_to_name[dim_id]
                else:
                    name = f"X{dim_id}"  # Fallback name
                
                length = bottom_name_lengths.get(name, "Unknown")
                
                bottom_view_data.append({
                    "Dimension ID": dim_id,
                    "Name": name,
                    "Length": length,
                    "Purpose": f"Tensor coordinate for dimension {dim_id}"
                })
            
            st.table(bottom_view_data)
            
            # Transformations (The Mapping Logic)
            st.subheader("🔄 Transformations (The Mapping Logic)")
            st.write("**The sequence of coordinate transformations:**")
            
            transforms = adaptor['Transformations']
            
            if transforms:
                st.write(f"**Number of transformations: {len(transforms)}**")
                
                for i, transform in enumerate(transforms):
                    with st.expander(f"Transform {i+1}: {transform['Name']}"):
                        st.write(f"**Type:** {transform['Name']}")
                        st.write(f"**Metadata:** {transform['MetaData']}")
                        st.write(f"**Source Dimension IDs:** {transform['SrcDimIds']}")
                        st.write(f"**Destination Dimension IDs:** {transform['DstDimIds']}")
                        
                        # Explain what this transform does
                        if transform['Name'] == 'Replicate':
                            st.info("🔄 **Replicate**: Creates R-dimensions from nothing (for replication across threads)")
                        elif transform['Name'] == 'Unmerge':
                            st.info("📊 **Unmerge**: Splits X-dimensions into H-components (tensor layout)")
                        elif transform['Name'] == 'Merge':
                            st.info("🔗 **Merge**: Combines R/H components into P-dimensions (thread assignment)")
                        elif transform['Name'] == 'PassThrough':
                            st.info("➡️ **PassThrough**: Direct mapping without transformation")
                        elif transform['Name'] == 'Embed':
                            st.info("📍 **Embed**: Adds padding or offset to coordinates")
                        else:
                            st.info(f"🔧 **{transform['Name']}**: Custom transformation")
            else:
                st.warning("No transformations found in the adaptor")
            
            # Coordinate Transformation Example
            st.subheader("🎯 Coordinate Transformation Example")
            st.write("**How hardware thread IDs become tensor coordinates:**")
            
            # Create an example transformation
            example_input = []
            example_description = []
            
            # Build example P coordinates
            if st.session_state.tile_dist.NDimPs == 1:
                example_input.extend([0])  # Example: lane_id = 0
                example_description.extend(["lane_id=0"])
            elif st.session_state.tile_dist.NDimPs == 2:
                example_input.extend([0, 5])  # Example: warp_id=0, lane_id=5
                example_description.extend(["warp_id=0", "lane_id=5"])
            elif st.session_state.tile_dist.NDimPs > 2:
                example_input.extend([0] * st.session_state.tile_dist.NDimPs)
                example_description.extend([f"P{i}=0" for i in range(st.session_state.tile_dist.NDimPs)])
            
            # Build example Y coordinates
            if st.session_state.tile_dist.NDimYs > 0:
                example_input.extend([0] * st.session_state.tile_dist.NDimYs)
                example_description.extend([f"Y{i}=0" for i in range(st.session_state.tile_dist.NDimYs)])
            
            st.code(f"""
Example Transformation:

1. Hardware Input:
   {' + '.join(example_description)}
   → Input coordinates: {example_input}

2. Apply Transformation Chain:
   - Transform chain processes coordinates through {len(transforms)} steps
   - Each transform maps between hidden coordinate spaces
   - Final step extracts bottom (X) coordinates

3. Result:
   → Output coordinates: [X0, X1, ...] (tensor access coordinates)
            """, language="text")
            
            # Try to calculate an actual example
            if st.button("🧮 Calculate Example Transformation"):
                try:
                    if st.session_state.tile_dist.NDimPs > 0:
                        p_coords = [0] * st.session_state.tile_dist.NDimPs
                        y_coords = [0] * st.session_state.tile_dist.NDimYs
                        
                        # Use the actual transformation method
                        x_coords = st.session_state.tile_dist._calculate_xs_from_ps_ys(p_coords, y_coords, adaptor)
                        
                        st.success(f"✅ Transformation successful!")
                        st.write(f"**Input:** P={p_coords}, Y={y_coords}")
                        st.write(f"**Output:** X={x_coords}")
                        
                        # Calculate final index
                        final_index = st.session_state.tile_dist.calculate_index(p_coords, y_coords)
                        st.write(f"**Final Memory Index:** {final_index}")
                        
                    else:
                        st.warning("No P dimensions available for transformation example")
                        
                except Exception as e:
                    st.error(f"❌ Transformation failed: {str(e)}")
                    st.code(traceback.format_exc())
            
            # Hardware Thread ID Mapping
            st.subheader("🔧 Hardware Thread ID Mapping")
            st.write("**How `get_warp_id()` and `get_lane_id()` map to P coordinates:**")
            
            if st.session_state.tile_dist.NDimPs == 1:
                st.code("""
// NDimP = 1: Single-level distribution
auto partition_index = array<index_t, 1>{get_lane_id()};
// P0 = lane_id (0-31 typically)

// Usage in adaptor:
auto ps_ys_coords = container_concat(partition_index, array<index_t, NDimY>{0});
// Result: [lane_id, y0, y1, ...]
                """, language="cpp")
            elif st.session_state.tile_dist.NDimPs == 2:
                st.code("""
// NDimP = 2: Two-level distribution  
auto partition_index = array<index_t, 2>{get_warp_id(), get_lane_id()};
// P0 = warp_id (0-N warps per block)
// P1 = lane_id (0-31 threads per warp)

// Usage in adaptor:
auto ps_ys_coords = container_concat(partition_index, array<index_t, NDimY>{0});
// Result: [warp_id, lane_id, y0, y1, ...]
                """, language="cpp")
            else:
                st.code(f"""
// NDimP = {st.session_state.tile_dist.NDimPs}: Multi-level distribution
// This is unusual - typically NDimP is 1 or 2
auto partition_index = array<index_t, {st.session_state.tile_dist.NDimPs}>{{...}};
                """, language="cpp")
            
            # Key Insights
            st.subheader("💡 Key Insights")
            
            st.markdown("""
            **🔑 The PsYs→Xs Adaptor is the heart of Composable Kernels:**
            
            1. **Hardware Integration**: Maps GPU thread IDs directly to logical coordinates
            2. **Encoding Implementation**: Transforms your tile distribution encoding into executable coordinate math
            3. **Memory Access Pattern**: Determines exactly which memory locations each thread accesses
            4. **Performance Critical**: This mapping affects memory coalescing, bank conflicts, and cache efficiency
            
            **🎯 Why This Matters:**
            - **Thread Coordination**: Ensures threads don't conflict when accessing data
            - **Memory Efficiency**: Optimizes memory access patterns for GPU hardware
            - **Scalability**: Allows the same code to work with different thread configurations
            - **Debugging**: Understanding this mapping helps debug performance issues
            
            **🔧 In Your Code:**
            ```cpp
            // This line uses the adaptor:
            auto window_adaptor_thread_coord = make_tensor_adaptor_coordinate(
                tile_distribution.get_ps_ys_to_xs_adaptor(),
                container_concat(make_tuple(get_warp_id(), get_lane_id()), 
                                generate_tuple([&](auto) { return number<0>{}; }, number<NDimY>{}))
            );
            ```
            """)
            
            # Raw Adaptor Data
            with st.expander("🔍 Raw Adaptor Data (JSON)"):
                st.json(adaptor)
        
        with tab6:
            st.subheader("🔧 Full Tile Distribution Encoding")
            
            # Use the print_encoding method output
            encoding_output = []
            
            # Capture the print output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                st.session_state.tile_dist.DstrEncode.print_encoding()
            encoding_text = f.getvalue()
            
            st.code(encoding_text, language="text")
        
        with tab7:
            st.subheader("🔍 Index Mapping Debug")
            
            st.write("**Debug indexing relationships:**")
            
            # Show some example index mappings
            try:
                # Get the debug information (it returns a dict, doesn't print)
                debug_info = debug_indexing_relationships(st.session_state.parsed_encoding, variables)
                
                # Display the debug information in a structured way
                if debug_info:
                    st.subheader("📍 Major Indices")
                    st.write("Shows which major index each sequence type uses:")
                    for seq_name, major_idx in debug_info.get("MajorIndices", {}).items():
                        st.write(f"  {seq_name}: major index {major_idx}")
                    
                    st.subheader("📊 Minor Indices (Sequence Contents)")
                    st.write("Shows the actual contents of each sequence:")
                    for seq_name, contents in debug_info.get("MinorIndices", {}).items():
                        st.write(f"  {seq_name}: {contents}")
                    
                    st.subheader("🔗 P and Y Dimension Mappings")
                    st.write("Shows how P and Y dimensions map to specific R/H positions:")
                    
                    index_mappings = debug_info.get("IndexMapping", {})
                    
                    # Display P mappings
                    p_keys = [k for k in index_mappings.keys() if k.startswith('P')]
                    if p_keys:
                        st.write("**P Dimension Mappings:**")
                        for p_key in sorted(p_keys):
                            mappings = index_mappings[p_key]
                            st.write(f"  {p_key}:")
                            for i, mapping in enumerate(mappings):
                                target = mapping.get("Target", "Unknown")
                                major_idx = mapping.get("MajorIndex", "?")
                                minor_idx = mapping.get("MinorIndex", "?")
                                symbolic = mapping.get("SymbolicValue", "N/A")
                                value = mapping.get("Value", "N/A")
                                st.write(f"    [{i}]: {target}[{minor_idx}] (major={major_idx}) = {symbolic} = {value}")
                    
                    # Display Y mappings
                    y_keys = [k for k in index_mappings.keys() if k.startswith('Y')]
                    if y_keys:
                        st.write("**Y Dimension Mappings:**")
                        for y_key in sorted(y_keys):
                            mappings = index_mappings[y_key]
                            st.write(f"  {y_key}:")
                            for mapping in mappings:
                                target = mapping.get("Target", "Unknown")
                                major_idx = mapping.get("MajorIndex", "?")
                                minor_idx = mapping.get("MinorIndex", "?")
                                symbolic = mapping.get("SymbolicValue", "N/A")
                                value = mapping.get("Value", "N/A")
                                st.write(f"    → {target}[{minor_idx}] (major={major_idx}) = {symbolic} = {value}")
                    
                    # Show raw debug data in expandable section
                    with st.expander("🔍 Raw Debug Data"):
                        st.json(debug_info)
                
                else:
                    st.warning("No debug information available")
                    
            except Exception as e:
                st.error(f"Debug output failed: {e}")
                st.code(traceback.format_exc())
            
            # Additional useful index mapping information
            st.subheader("📋 Additional Index Information")
            
            # Show span mappings in a clear table format
            detail = st.session_state.tile_dist.DstrEncode.detail
            
            # Create a comprehensive mapping table
            st.write("**Complete Y Dimension Analysis:**")
            
            mapping_analysis = []
            for i in range(st.session_state.tile_dist.NDimYs):
                rh_major = st.session_state.tile_dist.DstrEncode.Ys2RHsMajor[i]
                rh_minor = st.session_state.tile_dist.DstrEncode.Ys2RHsMinor[i]
                span_major = detail['ys_to_span_major_'][i]
                span_minor = detail['ys_to_span_minor_'][i]
                is_from_r = detail['is_ys_from_r_span_'][i]
                
                if rh_major == 0:
                    source = f"R[{rh_minor}]"
                    length = st.session_state.tile_dist.DstrEncode.RsLengths[rh_minor] if rh_minor < len(st.session_state.tile_dist.DstrEncode.RsLengths) else "N/A"
                else:
                    h_idx = rh_major - 1
                    source = f"H{h_idx}[{rh_minor}]"
                    length = st.session_state.tile_dist.DstrEncode.HsLengthss[h_idx][rh_minor] if rh_minor < len(st.session_state.tile_dist.DstrEncode.HsLengthss[h_idx]) else "N/A"
                
                mapping_analysis.append({
                    "Y Dim": f"Y{i}",
                    "RH Major": rh_major,
                    "RH Minor": rh_minor,
                    "Source": source,
                    "Length": length,
                    "Span Major": span_major,
                    "Span Minor": span_minor,
                    "From R Span": is_from_r
                })
            
            st.table(mapping_analysis)

if __name__ == "__main__":
    main() 