#!/usr/bin/env python3
"""
Streamlit app for visualizing tensor coordinate mappings.

This app allows interactive exploration of how different RMSNorm parameters
affect thread coordinate mappings in the tile distribution.
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys
import os

# Add the current directory to the Python path
sys.path.append('.')

from pytensor.tile_distribution_encoding import TileDistributionEncoding
from pytensor.tile_distribution import make_static_tile_distribution
from pytensor.tensor_coordinate import MultiIndex
from pytensor.partition_simulation import set_global_thread_position

def create_rmsnorm_encoding(repeat_m, warp_per_block_m, thread_per_warp_m, vector_m,
                           repeat_n, warp_per_block_n, thread_per_warp_n, vector_n):
    """Create RMSNorm encoding from parameters."""
    
    # Calculate H sequences
    hs_lengthss = [
        [repeat_m, warp_per_block_m, thread_per_warp_m, vector_m],
        [repeat_n, warp_per_block_n, thread_per_warp_n, vector_n]
    ]
    
    # Create encoding
    encoding = TileDistributionEncoding(
        rs_lengths=[],
        hs_lengthss=hs_lengthss,
        ps_to_rhss_major=[[1, 1], [1, 1]], 
        ps_to_rhss_minor=[[0, 2], [1, 3]],
        ys_to_rhs_major=[1, 1, 1, 1],
        ys_to_rhs_minor=[0, 1, 2, 3]
    )
    
    return encoding

def simulate_thread_coordinates(encoding, num_threads=16, max_y_steps=16):
    """Simulate thread coordinates for visualization."""
    
    # Create tile distribution
    tile_distribution = make_static_tile_distribution(encoding)
    adaptor = tile_distribution.ps_ys_to_xs_adaptor
    
    # Get Y dimension information
    ys_to_d_desc = tile_distribution.ys_to_d_descriptor
    y_lengths = ys_to_d_desc.get_lengths()
    
    thread_data = []
    
    for thread_id in range(num_threads):
        warp_id = thread_id // 64
        lane_id = thread_id % 64
        
        # Set thread position
        set_global_thread_position(warp_id, lane_id)
        
        # Get partition index
        partition_idx = tile_distribution.get_partition_index()
        
        # Simulate Y iterations
        for y_step in range(min(max_y_steps, np.prod(y_lengths))):
            # Convert linear Y index to multi-dimensional Y coordinates
            y_coords = []
            remaining = y_step
            for dim in range(len(y_lengths) - 1, -1, -1):
                y_coords.insert(0, remaining % y_lengths[dim])
                remaining //= y_lengths[dim]
            
            # Create full PS_YS coordinate
            ps_ys_coords = partition_idx + y_coords
            
            # Calculate X coordinates
            try:
                multi_idx = MultiIndex(len(ps_ys_coords), ps_ys_coords)
                x_coord = adaptor.calculate_bottom_index(multi_idx)
                x_coords = x_coord.to_list()
                
                thread_data.append({
                    'thread_id': thread_id,
                    'warp_id': warp_id,
                    'lane_id': lane_id,
                    'y_step': y_step,
                    'P0': partition_idx[0],
                    'P1': partition_idx[1],
                    'Y0': y_coords[0],
                    'Y1': y_coords[1],
                    'Y2': y_coords[2],
                    'Y3': y_coords[3],
                    'X0': x_coords[0] if len(x_coords) > 0 else 0,
                    'X1': x_coords[1] if len(x_coords) > 1 else 0,
                })
            except Exception as e:
                # Handle errors gracefully
                continue
    
    return pd.DataFrame(thread_data)

def create_coordinate_heatmap(df, tensor_shape):
    """Create a heatmap showing which threads access which coordinates."""
    
    # Create a 2D grid for the tensor
    max_x0 = min(tensor_shape[0] - 1, df['X0'].max()) if len(df) > 0 else tensor_shape[0] - 1
    max_x1 = min(tensor_shape[1] - 1, df['X1'].max()) if len(df) > 0 else tensor_shape[1] - 1
    
    # Create heatmap data
    heatmap_data = np.full((max_x0 + 1, max_x1 + 1), -1, dtype=int)
    
    for _, row in df.iterrows():
        x0, x1 = int(row['X0']), int(row['X1'])
        if 0 <= x0 <= max_x0 and 0 <= x1 <= max_x1:
            heatmap_data[x0, x1] = row['thread_id']
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        colorscale='viridis',
        showscale=True,
        colorbar=dict(title="Thread ID"),
        hovertemplate='X0: %{y}<br>X1: %{x}<br>Thread: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Thread Access Pattern Heatmap",
        xaxis_title="X1 (Column)",
        yaxis_title="X0 (Row)",
        width=600,
        height=500
    )
    
    return fig

def create_y_dimension_plot(df):
    """Create a plot showing Y dimension effects."""
    
    if len(df) == 0:
        return go.Figure()
    
    # Create subplots for each Y dimension
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Y0 Effect on X0", "Y1 Effect on X1", "Y2 Effect on X0", "Y3 Effect on X1"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Fix thread to 0 for cleaner visualization
    thread_df = df[df['thread_id'] == 0]
    
    if len(thread_df) > 0:
        # Y0 vs X0
        fig.add_trace(
            go.Scatter(x=thread_df['Y0'], y=thread_df['X0'], mode='markers+lines', name='Y0→X0'),
            row=1, col=1
        )
        
        # Y1 vs X1  
        fig.add_trace(
            go.Scatter(x=thread_df['Y1'], y=thread_df['X1'], mode='markers+lines', name='Y1→X1'),
            row=1, col=2
        )
        
        # Y2 vs X0
        fig.add_trace(
            go.Scatter(x=thread_df['Y2'], y=thread_df['X0'], mode='markers+lines', name='Y2→X0'),
            row=2, col=1
        )
        
        # Y3 vs X1
        fig.add_trace(
            go.Scatter(x=thread_df['Y3'], y=thread_df['X1'], mode='markers+lines', name='Y3→X1'),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Y Dimension Effects on X Coordinates (Thread 0)",
        height=500,
        showlegend=False
    )
    
    return fig

def create_partition_analysis(df):
    """Create analysis of partition (P) effects."""
    
    if len(df) == 0:
        return go.Figure()
    
    # Group by partition indices
    partition_stats = df.groupby(['P0', 'P1']).agg({
        'X0': ['min', 'max', 'mean'],
        'X1': ['min', 'max', 'mean'],
        'thread_id': 'first'
    }).round(2)
    
    # Flatten column names
    partition_stats.columns = ['_'.join(col).strip() for col in partition_stats.columns.values]
    partition_stats = partition_stats.reset_index()
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=partition_stats['X1_mean'],
        y=partition_stats['X0_mean'],
        mode='markers+text',
        text=[f"P[{row['P0']},{row['P1']}]" for _, row in partition_stats.iterrows()],
        textposition="top center",
        marker=dict(
            size=10,
            color=partition_stats['thread_id_first'],
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Thread ID")
        ),
        hovertemplate='P0: %{customdata[0]}<br>P1: %{customdata[1]}<br>Mean X0: %{y}<br>Mean X1: %{x}<extra></extra>',
        customdata=partition_stats[['P0', 'P1']].values
    ))
    
    fig.update_layout(
        title="Partition Index Effects on Mean X Coordinates",
        xaxis_title="Mean X1",
        yaxis_title="Mean X0",
        width=600,
        height=400
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Tensor Coordinate Visualization",
        page_icon="🧮",
        layout="wide"
    )
    
    st.title("🧮 Tensor Coordinate Mapping Visualization")
    st.markdown("Interactive visualization of thread coordinate mappings in RMSNorm tile distribution")
    
    # Sidebar for parameters
    st.sidebar.header("RMSNorm Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.subheader("M Dimension")
        repeat_m = st.slider("Repeat_M", 1, 8, 4)
        warp_per_block_m = st.slider("WarpPerBlock_M", 1, 4, 2)
        thread_per_warp_m = st.slider("ThreadPerWarp_M", 1, 64, 8)
        vector_m = st.slider("Vector_M", 1, 8, 4)
    
    with col2:
        st.subheader("N Dimension")
        repeat_n = st.slider("Repeat_N", 1, 8, 4)
        warp_per_block_n = st.slider("WarpPerBlock_N", 1, 4, 2)
        thread_per_warp_n = st.slider("ThreadPerWarp_N", 1, 64, 8)
        vector_n = st.slider("Vector_N", 1, 8, 4)
    
    # Simulation parameters
    st.sidebar.header("Simulation Parameters")
    num_threads = st.sidebar.slider("Number of Threads", 1, 32, 8)
    max_y_steps = st.sidebar.slider("Max Y Steps", 1, 64, 16)
    
    # Calculate tensor dimensions
    m_size = repeat_m * warp_per_block_m * thread_per_warp_m * vector_m
    n_size = repeat_n * warp_per_block_n * thread_per_warp_n * vector_n
    tensor_shape = [m_size, n_size]
    
    st.sidebar.markdown(f"**Calculated Tensor Shape:** {tensor_shape}")
    
    # Main content
    try:
        # Create encoding
        encoding = create_rmsnorm_encoding(
            repeat_m, warp_per_block_m, thread_per_warp_m, vector_m,
            repeat_n, warp_per_block_n, thread_per_warp_n, vector_n
        )
        
        # Simulate coordinates
        with st.spinner("Simulating thread coordinates..."):
            df = simulate_thread_coordinates(encoding, num_threads, max_y_steps)
        
        if len(df) == 0:
            st.error("No coordinate data generated. Try adjusting parameters.")
            return
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Coordinates", len(df))
        with col2:
            unique_coords = len(df[['X0', 'X1']].drop_duplicates())
            st.metric("Unique Coordinates", unique_coords)
        with col3:
            coverage = (unique_coords / (tensor_shape[0] * tensor_shape[1])) * 100
            st.metric("Coverage %", f"{coverage:.2f}%")
        with col4:
            conflicts = len(df) - unique_coords
            st.metric("Coordinate Conflicts", conflicts)
        
        # Visualizations
        st.header("Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Y Dimensions", "Partition Analysis", "Raw Data"])
        
        with tab1:
            st.subheader("Thread Access Pattern")
            heatmap_fig = create_coordinate_heatmap(df, tensor_shape)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        with tab2:
            st.subheader("Y Dimension Effects")
            y_fig = create_y_dimension_plot(df)
            st.plotly_chart(y_fig, use_container_width=True)
            
            # Show specific examples
            st.subheader("Y Dimension Change Examples")
            if len(df) > 0:
                thread_0_data = df[df['thread_id'] == 0].head(8)
                for _, row in thread_0_data.iterrows():
                    st.write(f"Y[{row['Y0']},{row['Y1']},{row['Y2']},{row['Y3']}] → X[{row['X0']},{row['X1']}]")
        
        with tab3:
            st.subheader("Partition Index Analysis")
            partition_fig = create_partition_analysis(df)
            st.plotly_chart(partition_fig, use_container_width=True)
        
        with tab4:
            st.subheader("Raw Coordinate Data")
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="thread_coordinates.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please check your parameters and try again.")

if __name__ == "__main__":
    main() 