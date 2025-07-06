from manim import *
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the TileDistributionEncoding class
from pytensor.tile_distribution_encoding import TileDistributionEncoding, make_reduce_tile_distribution_encoding

# --- Encodings Definition ---
# Input encoding based on a C++ example for a GEMM kernel
# Note: S::... values are placeholders, so we use constant integers.
in_encoding = TileDistributionEncoding(
    rs_lengths=[],
    hs_lengthss=[[2, 2, 4, 4], [2, 2, 4, 4]], # M, N dims with sub-dims
    ps_to_rhss_major=[[1, 2], [1, 2]],
    ps_to_rhss_minor=[[1, 1], [2, 2]],
    ys_to_rhs_major=[1, 1, 2, 2],
    ys_to_rhs_minor=[0, 3, 0, 3]
)

# X dimension to reduce (0-indexed)
reduce_dim_xs = [1] # This corresponds to sequence<1>{}


# --- Manim Scene ---
class TileDistributionReduction(Scene):
    def construct(self):
        # Title
        title = Text("Tile Distribution Encoding Reduction", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Show the reduction process with compact graph visualization
        self.show_graph_reduction_animation()
        
    def show_graph_reduction_animation(self):
        """Show the complete reduction process with animated graph transformations."""
        
        # Step 1: Show initial graph on the left
        self.show_initial_graph(in_encoding)
        self.wait(2)
        
        # Step 2: Highlight reduction targets on the same graph
        self.highlight_reduction_targets(in_encoding, reduce_dim_xs)
        self.wait(2)

        # Step 3: Animate the transformation to the final reduced graph on the right
        self.animate_to_final_graph(in_encoding, reduce_dim_xs)
        self.wait(3)
        
    def show_initial_graph(self, encoding):
        """Show the initial graph representation on the left side of the screen."""
        initial_label = Text("Initial Encoding", font_size=24)
        initial_label.move_to(LEFT * 4 + UP * 2.8)
        
        initial_graph = self.create_compact_encoding_graph(encoding, LEFT * 4)
        
        self.play(Write(initial_label))
        self.play(Create(initial_graph), run_time=2)
        
        self.initial_label = initial_label
        self.initial_graph = initial_graph

    def create_compact_encoding_graph(self, encoding, center_pos, indicate_reduced=False, reduce_dims=None):
        """Create a compact graph from a TileDistributionEncoding, with optional highlighting."""
        if reduce_dims is None:
            reduce_dims = []
            
        graph_group = VGroup()
        p_y_level, hidden_level = center_pos + UP * 1.5, center_pos + DOWN * 0.5
        
        # P nodes
        p_boxes = VGroup()
        for i in range(encoding.ndim_p):
            p_box = Rectangle(width=0.6, height=0.4, color=PURPLE, fill_opacity=0.5)
            p_text = Text(f"P{i}", font_size=12).move_to(p_box.get_center())
            p_boxes.add(VGroup(p_box, p_text))
        if p_boxes.submobjects:
            p_boxes.arrange(RIGHT, buff=0.2).move_to(p_y_level + LEFT * 1.2)

        # Y nodes
        y_boxes = VGroup()
        reduced_y_indices = {i for i, major in enumerate(encoding.ys_to_rhs_major) if major - 1 in reduce_dims}
        for i in range(encoding.ndim_y):
            color = RED if indicate_reduced and i in reduced_y_indices else ORANGE
            y_box = Rectangle(width=0.6, height=0.4, color=color, fill_opacity=0.5)
            y_text = Text(f"Y{i}", font_size=12).move_to(y_box.get_center())
            y_boxes.add(VGroup(y_box, y_text))
        if y_boxes.submobjects:
            y_boxes.arrange(RIGHT, buff=0.2).move_to(p_y_level + RIGHT * 1.2)

        # R hidden nodes
        r_hidden_boxes = VGroup()
        for i, r_val in enumerate(encoding.rs_lengths):
            r_box = Rectangle(width=0.5, height=0.4, color=YELLOW, fill_opacity=0.5)
            r_text = Text(str(r_val), font_size=12).move_to(r_box.get_center())
            r_hidden_boxes.add(VGroup(r_box, r_text))
        if r_hidden_boxes.submobjects:
            r_hidden_boxes.arrange(RIGHT, buff=0.15)

        # H hidden nodes
        h_hidden_boxes = VGroup()
        for h_idx, h_lengths in enumerate(encoding.hs_lengthss):
            is_reduced_h_dim = indicate_reduced and h_idx in reduce_dims
            h_dim_group = VGroup()
            for i, h_val in enumerate(h_lengths):
                h_box = Rectangle(width=0.4, height=0.4, color=RED if is_reduced_h_dim else GREEN, fill_opacity=0.5)
                h_text = Text(str(h_val), font_size=12).move_to(h_box.get_center())
                h_dim_group.add(VGroup(h_box, h_text))
            if h_dim_group.submobjects:
                h_dim_group.arrange(RIGHT, buff=0.1)
            h_hidden_boxes.add(h_dim_group)
        if h_hidden_boxes.submobjects:
            h_hidden_boxes.arrange(RIGHT, buff=0.6)
            
        rh_group = VGroup()
        if r_hidden_boxes.submobjects: rh_group.add(r_hidden_boxes)
        if h_hidden_boxes.submobjects: rh_group.add(h_hidden_boxes)
        rh_group.arrange(RIGHT, buff=0.8).move_to(hidden_level)

        # Add R and H dimension labels
        rh_labels = VGroup()
        if r_hidden_boxes.submobjects:
            r_label = Text("R", font_size=16).next_to(r_hidden_boxes, UP, buff=0.2)
            rh_labels.add(r_label)
        if h_hidden_boxes.submobjects:
            for i, h_major_group in enumerate(h_hidden_boxes):
                if h_major_group.submobjects:
                    color = RED if indicate_reduced and i in reduce_dims else WHITE
                    h_label = Text(f"H{i}", font_size=16, color=color).next_to(h_major_group, UP, buff=0.2)
                    rh_labels.add(h_label)
        
        graph_group.add(p_boxes, y_boxes, rh_group, rh_labels)
        graph_group.p_boxes, graph_group.y_boxes = p_boxes, y_boxes
        graph_group.r_hidden_boxes, graph_group.h_hidden_boxes = r_hidden_boxes, h_hidden_boxes
        graph_group.rh_labels = rh_labels
        
        # Edges
        p_arrows, y_arrows = self.create_edges(encoding, p_boxes, y_boxes, r_hidden_boxes, h_hidden_boxes)
        graph_group.add(p_arrows, y_arrows)
        
        graph_group.p_arrows, graph_group.y_arrows = p_arrows, y_arrows
        return graph_group
        
    def create_edges(self, encoding, p_boxes, y_boxes, r_nodes, h_nodes):
        """Helper to create P and Y edges for a graph."""
        p_arrows, y_arrows = VGroup(), VGroup()
        if p_boxes.submobjects:
            for i in range(encoding.ndim_p):
                start_node = p_boxes[i]
                for major, minor in zip(encoding.ps_to_rhss_major[i], encoding.ps_to_rhss_minor[i]):
                    if major < 0 or minor < 0: continue
                    end_node = r_nodes[minor] if major == 0 else h_nodes[major - 1][minor]
                    p_arrows.add(Arrow(start_node.get_bottom(), end_node.get_top(), buff=0.1, stroke_width=1.5, tip_length=0.1, color=PURPLE))

        if y_boxes.submobjects:
            for i in range(encoding.ndim_y):
                start_node = y_boxes[i]
                major, minor = encoding.ys_to_rhs_major[i], encoding.ys_to_rhs_minor[i]
                if major < 0 or minor < 0: continue
                end_node = r_nodes[minor] if major == 0 else h_nodes[major - 1][minor]
                y_arrows.add(Arrow(start_node.get_bottom(), end_node.get_top(), buff=0.1, stroke_width=1.5, tip_length=0.1, color=ORANGE))
        return p_arrows, y_arrows
    
    def highlight_reduction_targets(self, encoding, reduce_dims):
        """Highlight nodes and edges that will be affected by the reduction by changing their color."""
        
        animations = []
        
        # Find H groups and their labels to highlight
        for h_idx in reduce_dims:
            if h_idx < len(self.initial_graph.h_hidden_boxes):
                h_group = self.initial_graph.h_hidden_boxes[h_idx]
                for node in h_group:  # node is a VGroup(box, text)
                    animations.append(node[0].animate.set_fill(RED, opacity=0.6))
                
                for label in self.initial_graph.rh_labels:
                    if f"H{h_idx}" in label.text:
                        animations.append(label.animate.set_color(RED))
                        break
        
        # Find Y nodes that are mapped to the reduced H dimensions
        reduced_y_indices = {i for i, major in enumerate(encoding.ys_to_rhs_major) if major - 1 in reduce_dims}
        for y_idx in reduced_y_indices:
            if y_idx < len(self.initial_graph.y_boxes):
                y_node = self.initial_graph.y_boxes[y_idx]  # y_node is a VGroup(box, text)
                animations.append(y_node[0].animate.set_fill(RED, opacity=0.6))

        self.play(*animations, run_time=1.5)
        
    def animate_to_final_graph(self, encoding, reduce_dims):
        """Animate the transformation from initial to final graph."""
        # Final graph setup on the right
        final_label = Text("Final Reduced Encoding", font_size=24).move_to(RIGHT * 4 + UP * 2.8)
        
        final_encoding = make_reduce_tile_distribution_encoding(encoding, reduce_dims)
        final_graph = self.create_compact_encoding_graph(final_encoding, RIGHT * 4)
        
        final_graph_nodes = VGroup(
            final_graph.p_boxes,
            final_graph.y_boxes,
            final_graph.r_hidden_boxes,
            final_graph.h_hidden_boxes,
            final_graph.rh_labels
        )
        final_graph_edges = VGroup(final_graph.p_arrows, final_graph.y_arrows)
        
        # Explanation text at the bottom center
        explanation = VGroup(
            Text("1. Reduced H nodes become new R nodes or are discarded.", font_size=16),
            Text("2. Y nodes mapped to reduced H are removed.", font_size=16),
            Text("3. Edges are re-mapped to the new structure.", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT).move_to(DOWN * 3)

        self.play(Write(final_label))
        self.play(Write(explanation))
        self.wait(1)
        
        # Animate the creation of the final graph without transforming
        self.play(FadeIn(final_graph_nodes), run_time=1.5)
        self.play(Create(final_graph_edges), run_time=1.5)
        self.wait(2)

        self.final_graph = final_graph
        self.final_label = final_label
        self.explanation = explanation 