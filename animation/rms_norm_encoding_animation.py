from manim import *
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the TileDistributionEncoding class
from pytensor.tile_distribution_encoding import TileDistributionEncoding

# --- RMSNorm Example Encoding (from C++ examples) ---
# Real-World Example (RMSNorm) with default variable values
rms_norm_encoding = TileDistributionEncoding(
    rs_lengths=[],                        # Empty R (sequence<>)
    hs_lengthss=[[4, 2, 8, 4], [4, 2, 8, 4]],  # H0 and H1
    ps_to_rhss_major=[[1, 2], [1, 2]],   # P0/P1 map to H0,H1
    ps_to_rhss_minor=[[1, 1], [2, 2]],   # P0 maps to H0[1],H1[1]; P1 maps to H0[2],H1[2]
    ys_to_rhs_major=[1, 1, 2, 2],        # Y0,Y1 map to H0; Y2,Y3 map to H1
    ys_to_rhs_minor=[0, 3, 0, 3]         # Y0,Y2 map to index 0; Y1,Y3 map to index 3
)

class RMSNormEncodingExplanation(Scene):
    def construct(self):
        # Title
        title = Text("Visualisation of tile distribution encoding", font_size=32).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Main animation sequence
        self.show_graph_creation_animation()
        self.wait(3)

    def show_graph_creation_animation(self):
        """Show the complete graph creation process with contextual highlighting."""
        
        # Define layout regions
        code_pos = LEFT * 4.5 + UP * 1
        graph_pos = RIGHT * 2.5 + DOWN * 0.5
        text_pos = DOWN * 3.2

        # Step 1: Show C++ encoding with major indices explained in comments
        self.show_cpp_encoding(code_pos)
        self.wait(1)

        # Step 2: Create nodes while highlighting the code
        self.create_nodes_stepwise(graph_pos, text_pos)
        self.wait(1)

        # Step 3: Create edges with detailed major/minor highlighting
        self.create_edges_stepwise(text_pos)
        self.wait(2)

    def show_cpp_encoding(self, position):
        """Display the C++ code with addressable parts for highlighting."""
        
        # Create C++ code display with addressable VGroups for highlighting
        self.code_lines = VGroup(
            Text("tile_distribution_encoding<", font_size=12),
            Text("  sequence<>,                      // R (major=0)", font_size=12, color=YELLOW),
            Text("  tuple<sequence<4,2,8,4>,         // H0 (major=1)", font_size=12, color=GREEN),
            Text("        sequence<4,2,8,4>>,        // H1 (major=2)", font_size=12, color=GREEN),
            VGroup( # P Majors
                Text("  tuple<sequence<", font_size=12), Text("1", font_size=12), Text(",", font_size=12), Text("2", font_size=12), Text(">,", font_size=12),
                Text(" sequence<", font_size=12), Text("1", font_size=12), Text(",", font_size=12), Text("2", font_size=12), Text(">>, // P major", font_size=12)
            ).arrange(RIGHT, buff=0.05),
            VGroup( # P Minors
                Text("  tuple<sequence<", font_size=12), Text("1", font_size=12), Text(",", font_size=12), Text("1", font_size=12), Text(">,", font_size=12),
                Text(" sequence<", font_size=12), Text("2", font_size=12), Text(",", font_size=12), Text("2", font_size=12), Text(">>, // P minor", font_size=12)
            ).arrange(RIGHT, buff=0.05),
            VGroup( # Y Major
                Text("  sequence<", font_size=12), Text("1", font_size=12), Text(",", font_size=12), Text("1", font_size=12), Text(",", font_size=12), 
                Text("2", font_size=12), Text(",", font_size=12), Text("2", font_size=12), Text(">, // Y major", font_size=12)
            ).arrange(RIGHT, buff=0.05),
            VGroup( # Y Minor
                Text("  sequence<", font_size=12), Text("0", font_size=12), Text(",", font_size=12), Text("3", font_size=12), Text(",", font_size=12), 
                Text("0", font_size=12), Text(",", font_size=12), Text("3", font_size=12), Text(">>{} // Y minor", font_size=12)
            ).arrange(RIGHT, buff=0.05)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to(position)

        # Set colors for different parts
        self.code_lines[4].set_color(PURPLE)
        self.code_lines[5].set_color(PURPLE)
        self.code_lines[6].set_color(ORANGE)
        self.code_lines[7].set_color(ORANGE)

        self.play(Write(self.code_lines))

    def create_nodes_stepwise(self, graph_pos, text_pos):
        """Create H, P, and Y nodes sequentially, highlighting code."""

        # Step for R nodes
        step_text = Text("Step 1: Check for R (Replication) nodes", font_size=16).move_to(text_pos)
        self.play(Write(step_text))
        self.play(Indicate(self.code_lines[1], scale_factor=1.1, color=YELLOW))
        r_explanation = Text("The empty sequence means no R nodes are created.",
                             font_size=14, color=GRAY).next_to(step_text, DOWN, buff=0.3)
        self.play(Write(r_explanation))
        self.wait(2)
        self.play(FadeOut(step_text), FadeOut(r_explanation))
        
        # Step 2: Create H nodes
        step_text = Text("Step 2: Create H (Hierarchy) nodes from hs_lengthss", font_size=16).move_to(text_pos)
        self.play(Write(step_text))
        self.play(Indicate(self.code_lines[2], scale_factor=1.1, color=GREEN), 
                  Indicate(self.code_lines[3], scale_factor=1.1, color=GREEN))

        h_nodes = VGroup()
        for h_major, h_lengths in enumerate(rms_norm_encoding.hs_lengthss):
            h_group = VGroup(*[VGroup(Rectangle(width=0.4, height=0.4, color=GREEN, fill_opacity=0.7), 
                                     Text(str(L), font_size=12, color=BLACK)) for L in h_lengths]).arrange(RIGHT, buff=0.1)
            h_nodes.add(h_group)
        h_nodes.arrange(RIGHT, buff=0.8).move_to(graph_pos)
        
        h_labels = VGroup(Text("H0", font_size=16, color=GREEN).next_to(h_nodes[0], DOWN, buff=0.2),
                          Text("H1", font_size=16, color=GREEN).next_to(h_nodes[1], DOWN, buff=0.2))
        
        self.play(Create(h_nodes), Create(h_labels))
        h_explanation = Text("Each sequence has 4 values, creating 4 nodes for H0 and 4 for H1.",
                            font_size=14, color=GRAY).next_to(h_labels, DOWN, buff=0.3)
        self.play(Write(h_explanation))
        self.wait(2)
        self.play(FadeOut(h_explanation))
        self.h_nodes, self.h_labels = h_nodes, h_labels
        self.play(FadeOut(step_text))

        # Step 3: Create P nodes
        step_text = Text("Step 3: Create P (Parallelism) nodes", font_size=16).move_to(text_pos)
        p_nodes = VGroup(*[VGroup(Rectangle(width=0.6, height=0.4, color=PURPLE, fill_opacity=0.7),
                                  Text(f"P{i}", font_size=12, color=BLACK)) for i in range(rms_norm_encoding.ndim_p)])
        p_nodes.arrange(RIGHT, buff=0.2).move_to(graph_pos + UP * 1.8 + LEFT * 2.0)
        
        self.play(Write(step_text))
        self.play(Indicate(VGroup(self.code_lines[4], self.code_lines[5]), scale_factor=1.05, color=PURPLE))
        self.play(Create(p_nodes))
        self.p_nodes = p_nodes
        self.play(FadeOut(step_text))

        # Step 4: Create Y nodes
        step_text = Text("Step 4: Create Y (Yield) nodes", font_size=16).move_to(text_pos)
        y_nodes = VGroup(*[VGroup(Rectangle(width=0.6, height=0.4, color=ORANGE, fill_opacity=0.7),
                                  Text(f"Y{i}", font_size=12, color=BLACK)) for i in range(rms_norm_encoding.ndim_y)])
        y_nodes.arrange(RIGHT, buff=0.2).move_to(graph_pos + UP * 1.8 + RIGHT * 2.0)
        
        self.play(Write(step_text))
        self.play(Indicate(VGroup(self.code_lines[6], self.code_lines[7]), scale_factor=1.05, color=ORANGE))
        self.play(Create(y_nodes))
        y_explanation = Text("The Y sequences have 4 entries, creating 4 Y nodes.",
                            font_size=14, color=GRAY).next_to(y_nodes, DOWN, buff=0.3)
        y_explanation.shift(LEFT * 1.5)
        self.play(Write(y_explanation))
        self.wait(2)
        self.play(FadeOut(y_explanation))
        self.y_nodes = y_nodes
        self.play(FadeOut(step_text))

    def create_edges_stepwise(self, text_pos):
        """Create edges with detailed highlighting of major/minor indices."""

        # Create P edges
        step_text = Text("Step 5: Create P → H edges using major and minor indices", font_size=16).move_to(text_pos)
        self.play(Write(step_text))
        
        # P0 Edges
        self.play(Indicate(self.p_nodes[0]))
        # P0 -> H0[1] (major=1, minor=1)
        self.highlight_and_draw_edge(self.p_nodes[0], self.h_nodes[0][1], self.code_lines[4][1], self.code_lines[5][1], PURPLE)
        # P0 -> H1[1] (major=2, minor=1)
        self.highlight_and_draw_edge(self.p_nodes[0], self.h_nodes[1][1], self.code_lines[4][3], self.code_lines[5][3], PURPLE)
        
        # P1 Edges
        self.play(Indicate(self.p_nodes[1]))
        # P1 -> H0[2] (major=1, minor=2)
        self.highlight_and_draw_edge(self.p_nodes[1], self.h_nodes[0][2], self.code_lines[4][6], self.code_lines[5][6], PURPLE)
        # P1 -> H1[2] (major=2, minor=2)
        self.highlight_and_draw_edge(self.p_nodes[1], self.h_nodes[1][2], self.code_lines[4][8], self.code_lines[5][8], PURPLE)
        
        self.play(FadeOut(step_text))

        # Create Y edges
        step_text = Text("Step 6: Create Y → H edges using major and minor indices", font_size=16).move_to(text_pos)
        self.play(Write(step_text))

        # Y0 -> H0[0] (major=1, minor=0)
        self.highlight_and_draw_edge(self.y_nodes[0], self.h_nodes[0][0], self.code_lines[6][1], self.code_lines[7][1], ORANGE)
        # Y1 -> H0[3] (major=1, minor=3)
        self.highlight_and_draw_edge(self.y_nodes[1], self.h_nodes[0][3], self.code_lines[6][3], self.code_lines[7][3], ORANGE)
        # Y2 -> H1[0] (major=2, minor=0)
        self.highlight_and_draw_edge(self.y_nodes[2], self.h_nodes[1][0], self.code_lines[6][5], self.code_lines[7][5], ORANGE)
        # Y3 -> H1[3] (major=2, minor=3)
        self.highlight_and_draw_edge(self.y_nodes[3], self.h_nodes[1][3], self.code_lines[6][7], self.code_lines[7][7], ORANGE)

        self.play(FadeOut(step_text))

        # Final summary
        final_text = Text("Graph construction complete!", font_size=18).move_to(text_pos)
        self.play(Write(final_text))

    def highlight_and_draw_edge(self, start_node, end_node, major_code, minor_code, color):
        """Highlight code indices and draw the corresponding edge."""
        
        # Create temporary highlight boxes
        major_highlight = SurroundingRectangle(major_code, color=color, buff=0.05)
        minor_highlight = SurroundingRectangle(minor_code, color=color, buff=0.05)
        
        # Animate highlights
        self.play(Indicate(start_node), Create(major_highlight), Create(minor_highlight))
        self.play(Indicate(end_node))
        
        # Draw edge with consistent absolute tip length
        edge = Arrow(start_node.get_bottom(), end_node.get_top(), buff=0.1, color=color, stroke_width=2, tip_length=0.15)
        self.play(Create(edge))
        
        # Fade out highlights
        self.play(FadeOut(major_highlight), FadeOut(minor_highlight))

 