from manim import *
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the TileDistributionEncoding class
from pytensor.tile_distribution_encoding import TileDistributionEncoding, make_embed_tile_distribution_encoding

# --- Encodings Definition ---
# Outer encoding based on a block-level distribution
outer_encoding = TileDistributionEncoding(
    rs_lengths=[4],
    hs_lengthss=[[2, 8], [2]],
    ps_to_rhss_major=[[1, 0]],
    ps_to_rhss_minor=[[1, 0]],
    ys_to_rhs_major=[1, 2],
    ys_to_rhs_minor=[0, 0]
)

# Inner encoding based on a warp-level distribution
inner_encoding = TileDistributionEncoding(
    rs_lengths=[],
    hs_lengthss=[[4], [2, 2]],
    ps_to_rhss_major=[[2, 1]],
    ps_to_rhss_minor=[[0, 0]],
    ys_to_rhs_major=[2],
    ys_to_rhs_minor=[1]
)


# --- Manim Scene ---
class TileDistributionEmbedding(Scene):
    def construct(self):
        # Title
        title = Text("Tile Distribution Encoding Embedding", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Show the embedding process with the compact graph visualization
        self.show_graph_embedding_animation()
        
    def show_graph_embedding_animation(self):
        """Show the complete embedding process with animated graph merging."""
        
        # Step 1: Show initial graphs
        self.show_initial_graphs_properly(outer_encoding, inner_encoding)
        self.wait(2)
        
        # Step 2: Shrink graphs to make space
        self.shrink_graphs_for_merging()
        self.wait(1)
        
        # Step 3: Animate R merging
        self.animate_r_node_merging(outer_encoding, inner_encoding)
        self.wait(2)
        
        # Step 4: Animate H merging
        self.animate_h_node_merging(outer_encoding, inner_encoding)
        self.wait(2)
        
        # Step 5: Animate P node creation
        self.animate_p_node_creation(outer_encoding, inner_encoding)
        self.wait(2)
        
        # Step 6: Animate Y node creation
        self.animate_y_node_creation(outer_encoding, inner_encoding)
        self.wait(2)
        
        # Step 7: Complete the final graph with edges
        self.complete_final_graph(outer_encoding, inner_encoding)
        self.wait(3)
        
    def show_initial_graphs_properly(self, outer_enc, inner_enc):
        """Show initial graphs with proper spacing and sizing."""
        
        outer_label = Text("Outer Encoding", font_size=20, color=BLUE)
        outer_label.move_to(LEFT * 4 + UP * 2.8)
        
        inner_label = Text("Inner Encoding", font_size=20, color=RED)
        inner_label.move_to(RIGHT * 4 + UP * 2.8)
        
        outer_graph = self.create_compact_encoding_graph(outer_enc, LEFT * 4)
        inner_graph = self.create_compact_encoding_graph(inner_enc, RIGHT * 4)
        
        self.play(Write(outer_label), Write(inner_label))
        self.play(DrawBorderThenFill(outer_graph), run_time=2)
        
        self.outer_label, self.inner_label = outer_label, inner_label
        self.outer_graph, self.inner_graph = outer_graph, inner_graph
        
    def create_compact_encoding_graph(self, encoding, center_pos):
        """Create a compact graph representation from a TileDistributionEncoding object."""
        graph_group = VGroup()
        
        p_y_level, hidden_level = UP * 1.2, ORIGIN
        
        # Create P nodes
        p_boxes = VGroup()
        for i in range(encoding.ndim_p):
            p_box = Rectangle(width=0.5, height=0.4, color=PURPLE, fill_opacity=0.5)
            p_text = Text(f"P{i}", font_size=12).move_to(p_box.get_center())
            p_boxes.add(VGroup(p_box, p_text))
        if p_boxes.submobjects:
            p_boxes.arrange(RIGHT, buff=0.2).move_to(center_pos + p_y_level + LEFT * 0.8)

        # Create Y nodes
        y_boxes = VGroup()
        for i in range(encoding.ndim_y):
            y_box = Rectangle(width=0.5, height=0.4, color=ORANGE, fill_opacity=0.5)
            y_text = Text(f"Y{i}", font_size=12).move_to(y_box.get_center())
            y_boxes.add(VGroup(y_box, y_text))
        if y_boxes.submobjects:
            y_boxes.arrange(RIGHT, buff=0.2).move_to(center_pos + p_y_level + RIGHT * 0.8)

        # Create R hidden nodes
        r_hidden_boxes = VGroup()
        for i, r_val in enumerate(encoding.rs_lengths):
            r_box = Rectangle(width=0.5, height=0.4, color=YELLOW, fill_opacity=0.5)
            r_text = Text(str(r_val), font_size=12).move_to(r_box.get_center())
            r_hidden_boxes.add(VGroup(r_box, r_text))
        if r_hidden_boxes.submobjects:
            r_hidden_boxes.arrange(RIGHT, buff=0.15)

        # Create H hidden nodes
        h_hidden_boxes = VGroup()
        for h_idx, h_lengths in enumerate(encoding.hs_lengthss):
            h_dim_group = VGroup()
            for i, h_val in enumerate(h_lengths):
                h_box = Rectangle(width=0.4, height=0.4, color=GREEN, fill_opacity=0.5)
                h_text = Text(str(h_val), font_size=12).move_to(h_box.get_center())
                h_dim_group.add(VGroup(h_box, h_text))
            if h_dim_group.submobjects:
                h_dim_group.arrange(RIGHT, buff=0.1)
            h_hidden_boxes.add(h_dim_group)
        if h_hidden_boxes.submobjects:
            h_hidden_boxes.arrange(RIGHT, buff=0.6)

        # Position R and H groups: R on the left, H on the right
        rh_group = VGroup()
        if r_hidden_boxes.submobjects:
            rh_group.add(r_hidden_boxes)
        if h_hidden_boxes.submobjects:
            rh_group.add(h_hidden_boxes)
        rh_group.arrange(RIGHT, buff=0.8).move_to(center_pos + hidden_level)

        # Add R and H dimension labels
        rh_labels = VGroup()
        if r_hidden_boxes.submobjects:
            r_label = Text("R", font_size=16).next_to(r_hidden_boxes, UP, buff=0.2)
            rh_labels.add(r_label)
        if h_hidden_boxes.submobjects:
            for i, h_major_group in enumerate(h_hidden_boxes):
                if h_major_group.submobjects:
                    h_label = Text(f"H{i}", font_size=16).next_to(h_major_group, UP, buff=0.2)
                    rh_labels.add(h_label)
                    
        # P-to-RH arrows
        p_arrows = VGroup()
        if p_boxes.submobjects:
            for i in range(encoding.ndim_p):
                start_node = p_boxes[i]
                for major, minor in zip(encoding.ps_to_rhss_major[i], encoding.ps_to_rhss_minor[i]):
                    end_node = r_hidden_boxes[minor] if major == 0 else h_hidden_boxes[major - 1][minor]
                    p_arrows.add(Arrow(start_node.get_bottom(), end_node.get_top(), buff=0.1, stroke_width=1.5, tip_length=0.15, color=PURPLE))

        # Y-to-RH arrows
        y_arrows = VGroup()
        if y_boxes.submobjects:
            for i in range(encoding.ndim_y):
                start_node = y_boxes[i]
                major, minor = encoding.ys_to_rhs_major[i], encoding.ys_to_rhs_minor[i]
                end_node = r_hidden_boxes[minor] if major == 0 else h_hidden_boxes[major - 1][minor]
                y_arrows.add(Arrow(start_node.get_bottom(), end_node.get_top(), buff=0.1, stroke_width=1.5, tip_length=0.15, color=ORANGE))

        graph_group.add(p_boxes, y_boxes, rh_group, p_arrows, y_arrows, rh_labels)
        graph_group.p_boxes, graph_group.y_boxes = p_boxes, y_boxes
        graph_group.r_hidden_boxes, graph_group.h_hidden_boxes = r_hidden_boxes, h_hidden_boxes
        graph_group.rh_labels = rh_labels
        return graph_group
        
    def shrink_graphs_for_merging(self):
        result_label = Text("Combined Result", font_size=24).move_to(DOWN * 2.8)
        self.play(
            self.outer_graph.animate.scale(0.7).move_to(LEFT * 4 + UP * 1.2),
            self.inner_graph.animate.scale(0.7).move_to(RIGHT * 4 + UP * 1.2),
            self.outer_label.animate.scale(0.8).move_to(LEFT * 4 + UP * 2.8),
            self.inner_label.animate.scale(0.8).move_to(RIGHT * 4 + UP * 2.8),
            Write(result_label), run_time=1.5
        )
        self.result_center = DOWN * 2.2 + LEFT * 2.5
        self.result_elements = VGroup()

    def animate_r_node_merging(self, outer_enc, inner_enc):
        final_enc = make_embed_tile_distribution_encoding(outer_enc, inner_enc)
        step_text = Text("Step 1: Merge R nodes", font_size=18, color=YELLOW).move_to(self.result_center + UP * 2.5)
        self.play(Write(step_text))
        
        # Indicate source nodes
        self.play(
            *[Indicate(node, color=YELLOW) for node in self.outer_graph.r_hidden_boxes],
            *[Indicate(node, color=YELLOW) for node in self.inner_graph.r_hidden_boxes]
        )
        
        # Create final nodes for the result
        result_r_nodes = VGroup()
        for i, r_val in enumerate(final_enc.rs_lengths):
            r_box = Rectangle(width=0.5, height=0.4, color=YELLOW, fill_opacity=0.6)
            r_text = Text(str(r_val), font_size=12).move_to(r_box.get_center())
            result_r_nodes.add(VGroup(r_box, r_text))
        
        # Animate merging
        moving_copies = VGroup(self.outer_graph.r_hidden_boxes.copy(), self.inner_graph.r_hidden_boxes.copy())
        self.play(moving_copies.animate.move_to(self.result_center).scale(0.9))
        
        if result_r_nodes.submobjects:
            result_r_nodes.arrange(RIGHT, buff=0.15).move_to(self.result_center)
        
        self.play(FadeOut(moving_copies), FadeIn(result_r_nodes))
        
        self.result_r_nodes = result_r_nodes
        self.result_elements.add(result_r_nodes)
        self.step1_text = step_text

    def animate_h_node_merging(self, outer_enc, inner_enc):
        final_enc = make_embed_tile_distribution_encoding(outer_enc, inner_enc)
        self.play(FadeOut(self.step1_text))
        step_text = Text("Step 2: Concatenate H nodes", font_size=18, color=GREEN).move_to(self.result_center + UP * 2.5)
        self.play(Write(step_text))

        # Build final layout to get target positions
        final_h_layout = VGroup()
        for h_idx, h_lengths in enumerate(final_enc.hs_lengthss):
            h_dim_group = VGroup()
            for h_val in h_lengths:
                h_box = Rectangle(width=0.4, height=0.4, color=GREEN, fill_opacity=0.6)
                h_text = Text(str(h_val), font_size=12).move_to(h_box.get_center())
                h_dim_group.add(VGroup(h_box, h_text))
            if h_dim_group.submobjects:
                h_dim_group.arrange(RIGHT, buff=0.1)
            final_h_layout.add(h_dim_group)
        if final_h_layout.submobjects:
            final_h_layout.arrange(RIGHT, buff=0.6)

        # Position final layout relative to R nodes
        if self.result_r_nodes.submobjects:
            final_h_layout.next_to(self.result_r_nodes, RIGHT, buff=0.8)
        else:
            final_h_layout.move_to(self.result_center)

        # Animate copies moving from sides to final positions
        all_moved_copies = VGroup()
        result_h_nodes_structured = VGroup()

        for h_idx in range(len(final_enc.hs_lengthss)):
            outer_h_group = self.outer_graph.h_hidden_boxes[h_idx] if h_idx < len(self.outer_graph.h_hidden_boxes) else VGroup()
            inner_h_group = self.inner_graph.h_hidden_boxes[h_idx] if h_idx < len(self.inner_graph.h_hidden_boxes) else VGroup()
            self.play(*[Indicate(node) for node in outer_h_group], *[Indicate(node) for node in inner_h_group], run_time=0.7)

            outer_copies = outer_h_group.copy()
            inner_copies = inner_h_group.copy()
            
            moved_dim_group = VGroup()
            animations = []
            
            # Animate outer copies
            for i, copy_node in enumerate(outer_copies):
                target_pos = final_h_layout[h_idx][i].get_center()
                animations.append(copy_node.animate.move_to(target_pos))
                moved_dim_group.add(copy_node)

            # Animate inner copies
            offset = len(outer_copies)
            for i, copy_node in enumerate(inner_copies):
                target_pos = final_h_layout[h_idx][offset + i].get_center()
                animations.append(copy_node.animate.move_to(target_pos))
                moved_dim_group.add(copy_node)

            if animations:
                self.play(*animations, run_time=1)
            
            all_moved_copies.add(moved_dim_group)
            result_h_nodes_structured.add(moved_dim_group)

        self.result_h_nodes = result_h_nodes_structured
        self.result_elements.add(self.result_r_nodes, all_moved_copies)
        
        # Add R and H dimension labels for the result graph
        rh_labels = VGroup()
        if self.result_r_nodes.submobjects:
            r_label = Text("R", font_size=16).next_to(self.result_r_nodes, UP, buff=0.2)
            rh_labels.add(r_label)
        if self.result_h_nodes.submobjects:
            for i, h_major_group in enumerate(self.result_h_nodes):
                if h_major_group.submobjects:
                    h_label = Text(f"H{i}", font_size=16).next_to(h_major_group, UP, buff=0.2)
                    rh_labels.add(h_label)
        
        self.play(Write(rh_labels))

        self.result_rh_labels = rh_labels
        self.result_elements.add(rh_labels)
        
        self.step2_text = step_text
        
    def animate_p_node_creation(self, outer_enc, inner_enc):
        final_enc = make_embed_tile_distribution_encoding(outer_enc, inner_enc)
        self.play(FadeOut(self.step2_text))
        step_text = Text("Step 3: Combine P nodes", font_size=18, color=PURPLE).move_to(self.result_center + UP * 2.5)
        self.play(Write(step_text))

        # Indicate source nodes
        self.play(
            *[Indicate(node, color=PURPLE) for node in self.outer_graph.p_boxes],
            *[Indicate(node, color=PURPLE) for node in self.inner_graph.p_boxes]
        )

        result_p_nodes = VGroup(*[
            VGroup(Rectangle(width=0.5, height=0.4, color=PURPLE, fill_opacity=0.6), Text(f"P{i}", font_size=12).move_to(ORIGIN))
            for i in range(final_enc.ndim_p)
        ]).arrange(RIGHT, buff=0.2).next_to(self.result_elements, UP, buff=0.8)
        
        # Animate copies moving to final positions
        outer_p_copies = self.outer_graph.p_boxes.copy()
        inner_p_copies = self.inner_graph.p_boxes.copy()
        
        moving_nodes = VGroup()
        animations = []
        for i, p_copy in enumerate(outer_p_copies):
            animations.append(p_copy.animate.move_to(result_p_nodes[i].get_center()))
            moving_nodes.add(p_copy)
        
        offset = len(outer_p_copies)
        for i, p_copy in enumerate(inner_p_copies):
            animations.append(p_copy.animate.move_to(result_p_nodes[offset + i].get_center()))
            moving_nodes.add(p_copy)
        
        self.play(*animations, run_time=1.2)
        self.play(FadeOut(moving_nodes), FadeIn(result_p_nodes))

        self.result_p_nodes = result_p_nodes
        self.result_elements.add(result_p_nodes)
        self.step3_text = step_text

    def animate_y_node_creation(self, outer_enc, inner_enc):
        final_enc = make_embed_tile_distribution_encoding(outer_enc, inner_enc)
        self.play(FadeOut(self.step3_text))
        step_text = Text("Step 4: Combine Y nodes", font_size=18, color=ORANGE).move_to(self.result_center + UP * 2.5)
        self.play(Write(step_text))

        # Indicate source nodes
        self.play(
            *[Indicate(node, color=ORANGE) for node in self.outer_graph.y_boxes],
            *[Indicate(node, color=ORANGE) for node in self.inner_graph.y_boxes]
        )

        result_y_nodes = VGroup(*[
            VGroup(Rectangle(width=0.5, height=0.4, color=ORANGE, fill_opacity=0.6), Text(f"Y{i}", font_size=12).move_to(ORIGIN))
            for i in range(final_enc.ndim_y)
        ]).arrange(RIGHT, buff=0.2).next_to(self.result_p_nodes, RIGHT, buff=1.0)

        # Animate copies moving to final positions
        outer_y_copies = self.outer_graph.y_boxes.copy()
        inner_y_copies = self.inner_graph.y_boxes.copy()
        
        moving_nodes = VGroup()
        animations = []
        for i, y_copy in enumerate(outer_y_copies):
            animations.append(y_copy.animate.move_to(result_y_nodes[i].get_center()))
            moving_nodes.add(y_copy)
            
        offset = len(outer_y_copies)
        for i, y_copy in enumerate(inner_y_copies):
            animations.append(y_copy.animate.move_to(result_y_nodes[offset + i].get_center()))
            moving_nodes.add(y_copy)

        self.play(*animations, run_time=1.2)
        self.play(FadeOut(moving_nodes), FadeIn(result_y_nodes))

        self.result_y_nodes = result_y_nodes
        self.result_elements.add(result_y_nodes)
        self.step4_text = step_text
        
    def complete_final_graph(self, outer_enc, inner_enc):
        final_enc = make_embed_tile_distribution_encoding(outer_enc, inner_enc)
        self.play(FadeOut(self.step4_text))
        
        final_text = Text("Final: Connect with updated edges", font_size=18).move_to(self.result_center + UP * 2.5)
        self.play(Write(final_text))
        
        # Create P edges sequentially
        for p_idx in range(final_enc.ndim_p):
            start_node = self.result_p_nodes[p_idx]
            self.play(Indicate(start_node, color=PURPLE, scale_factor=1.2), run_time=0.5)
            edges_for_p = VGroup()
            for major, minor in zip(final_enc.ps_to_rhss_major[p_idx], final_enc.ps_to_rhss_minor[p_idx]):
                end_node = self.result_r_nodes[minor] if major == 0 else self.result_h_nodes[major - 1][minor]
                edges_for_p.add(Arrow(start_node.get_bottom(), end_node.get_top(), buff=0.1, stroke_width=1.5, tip_length=0.15, color=PURPLE))
            self.play(Create(edges_for_p), run_time=0.8)

        # Create Y edges sequentially
        for y_idx in range(final_enc.ndim_y):
            start_node = self.result_y_nodes[y_idx]
            self.play(Indicate(start_node, color=ORANGE, scale_factor=1.2), run_time=0.5)
            major, minor = final_enc.ys_to_rhs_major[y_idx], final_enc.ys_to_rhs_minor[y_idx]
            end_node = self.result_r_nodes[minor] if major == 0 else self.result_h_nodes[major - 1][minor]
            self.play(Create(Arrow(start_node.get_bottom(), end_node.get_top(), buff=0.1, stroke_width=1.5, tip_length=0.15, color=ORANGE)), run_time=0.8)
        
        self.wait(2)


# --- Helper Classes ---

R_NODE_COLOR, H_NODE_COLOR, P_NODE_COLOR, Y_NODE_COLOR = "#F9E79F", "#A9DFBF", "#D7BDE2", "#F5B7B1"
NODE_TEXT_COLOR = BLACK

class Edge:
    """Represents a directed edge between two nodes in the graph."""
    def __init__(self, start_node, end_node, start_node_key, end_node_key):
        self.start_node_key = start_node_key
        self.end_node_key = end_node_key
        self.arrow = self._create_arrow(start_node, end_node)

    def _create_arrow(self, start_node, end_node):
        if start_node is None or end_node is None:
            return None
            
        start_point, end_point = start_node.get_center(), end_node.get_center()
        vec = end_point - start_point
        
        if np.all(vec == 0): return Line(start_point, end_point)

        if abs(vec[0]) > abs(vec[1]):
            start_point = start_node.get_right() if vec[0] > 0 else start_node.get_left()
            end_point = end_node.get_left() if vec[0] > 0 else end_node.get_right()
        else:
            start_point = start_node.get_top() if vec[1] > 0 else start_node.get_bottom()
            end_point = end_node.get_bottom() if vec[1] > 0 else end_node.get_top()
        
        return Arrow(start_point, end_point, buff=0.1, stroke_width=2, max_tip_length_to_length_ratio=0.1, color=WHITE)


class EncodingGraph:
    """Manages the creation and layout of a graph for a TileDistributionEncoding."""
    def __init__(self, encoding, name):
        self.encoding = encoding
        self.name = name
        self.r_nodes, self.h_nodes, self.p_nodes, self.y_nodes = {}, {}, {}, {}
        self.all_nodes = {}
        self.container = self._create_container()
        self.graph = self._create_graph()

    def _create_node(self, text, color):
        node_shape = Rectangle(width=1.5, height=0.7, color=color, fill_opacity=0.8)
        node_text = Text(text, font_size=14, color=NODE_TEXT_COLOR)
        return VGroup(node_shape, node_text)

    def _create_container(self):
        container_rect = Rectangle(width=7, height=5.5, color=GREY).set_opacity(0.2)
        title_text = Text(self.name, font_size=20).next_to(container_rect, UP, buff=0.1)
        
        for r_idx, r_len in enumerate(self.encoding.rs_lengths):
            key = f"r{r_idx}"
            self.r_nodes[key] = self._create_node(f"R{r_idx}\\n(len={r_len})", R_NODE_COLOR)
            self.all_nodes[key] = self.r_nodes[key]

        for h_major, hs in enumerate(self.encoding.hs_lengthss):
            for h_minor, h_len in enumerate(hs):
                key = f"h{h_major}_{h_minor}"
                self.h_nodes[key] = self._create_node(f"H{h_major},{h_minor}\\n(len={h_len})", H_NODE_COLOR)
                self.all_nodes[key] = self.h_nodes[key]

        for p_idx in range(self.encoding.ndim_p):
            key = f"p{p_idx}"
            self.p_nodes[key] = self._create_node(f"P{p_idx}", P_NODE_COLOR)
            self.all_nodes[key] = self.p_nodes[key]

        for y_idx in range(self.encoding.ndim_y):
            key = f"y{y_idx}"
            self.y_nodes[key] = self._create_node(f"Y{y_idx}", Y_NODE_COLOR)
            self.all_nodes[key] = self.y_nodes[key]
            
        return VGroup(container_rect, title_text, VGroup(*self.all_nodes.values()))

    def _create_graph(self):
        positioning_group, h_group = VGroup(), VGroup()

        if self.h_nodes:
            h_nodes_by_major = {}
            for key, node in self.h_nodes.items():
                h_major = int(key.split('_')[0][1:])
                h_nodes_by_major.setdefault(h_major, []).append(node)
            major_groups = [VGroup(*h_nodes_by_major[major]).arrange(RIGHT, buff=0.2) for major in sorted(h_nodes_by_major.keys())]
            h_group = VGroup(*major_groups).arrange(DOWN, buff=0.5)
            positioning_group.add(h_group)
        
        if self.r_nodes:
            r_group = VGroup(*self.r_nodes.values()).arrange(RIGHT, buff=0.5)
            if h_group.submobjects: r_group.next_to(h_group, RIGHT, buff=1.0)
            positioning_group.add(r_group)

        positioning_group.move_to(ORIGIN)
        all_mobjects = VGroup(positioning_group)
        
        p_group, y_group = VGroup(), VGroup()
        if self.p_nodes:
            p_group = VGroup(*self.p_nodes.values()).arrange(RIGHT, buff=0.5).next_to(positioning_group, UP, buff=1.0)
            all_mobjects.add(p_group)
        if self.y_nodes:
            y_group = VGroup(*self.y_nodes.values()).arrange(RIGHT, buff=0.5).next_to(positioning_group, DOWN, buff=1.0)
            all_mobjects.add(y_group)

        if not self.r_nodes and not self.h_nodes and self.p_nodes and self.y_nodes:
             p_group.next_to(y_group, UP, buff=1.5)

        return all_mobjects
    
    def _get_edges(self, from_nodes, major_map_name, minor_map_name):
        edges = []
        major_map = getattr(self.encoding, major_map_name)
        minor_map = getattr(self.encoding, minor_map_name)

        for idx_str, start_node in from_nodes.items():
            idx = int(idx_str[1:])

            is_list_of_lists = isinstance(major_map[0], list)
            majors = major_map[idx] if is_list_of_lists else [major_map[idx]]
            minors = minor_map[idx] if is_list_of_lists else [minor_map[idx]]

            for rh_major, rh_minor in zip(majors, minors):
                end_node_key = f"r{rh_minor}" if rh_major == 0 else f"h{rh_major - 1}_{rh_minor}"
                end_node = self.all_nodes.get(end_node_key)
                if end_node:
                    edges.append(Edge(start_node, end_node, idx_str, end_node_key))
        return edges

    def get_p_edges(self):
        return self._get_edges(self.p_nodes, 'ps_to_rhss_major', 'ps_to_rhss_minor')

    def get_y_edges(self):
        return self._get_edges(self.y_nodes, 'ys_to_rhs_major', 'ys_to_rhs_minor') 