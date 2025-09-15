from manim import *
import numpy as np
import re

class ConvolutionTensorDescriptorAnimation(Scene):
    # Class-level constants for colors and layout
    BUFFER_COLOR = "#2E7D32"
    TENSOR_COLOR = "#1976D2"
    WINDOW_COLOR = "#F57C00"
    STRIDE_COLOR = "#AB47BC"
    LENGTH_COLOR = "#00ACC1"
    TILE_COLOR = "#4CAF50"
    
    CODE_FONT_SIZE = 14
    EXPLANATION_FONT_SIZE = 13

    def construct(self):
        # 1. Setup title
        title = Text("Tensor Views: A Visual Explanation", font_size=36)
        self.play(Write(title))
        self.wait(1.5)
        self.play(title.animate.scale(0.6).to_corner(UL, buff=0.4))

        # --- Scene 1: Buffer to 2D Grid ---
        self.show_intro_text("Data in memory is linear.\nLet's create a 2D grid view from it.")
        buffer_view = self.create_buffer_view(animate=True)
        grid_cells, code_group, explanation_text = self.transform_to_2d(buffer_view)
        self.wait(2.5)
        
        # --- Scene 2: Non-overlapping Tiles ---
        self.play(FadeOut(grid_cells), FadeOut(code_group), FadeOut(explanation_text), FadeOut(buffer_view))
        self.show_intro_text(
            "Let's start simple by viewing the memory as a non-overlapping\n"
            "tiled grid, just by changing the lengths and strides to 4D."
        )
        base_grid = self.create_base_grid(animate=True)
        tiled_grid, code_group, explanation_text = self.show_non_overlapping_tiles(base_grid)
        self.wait(2.5)

        # --- Scene 3: Overlapping Windows ---
        self.play(FadeOut(tiled_grid), FadeOut(code_group), FadeOut(explanation_text))
        self.show_intro_text(
            "Now for the overlapping view needed for convolution.\n"
            "Again, we just adjust the tensor view's parameters."
        )
        base_grid = self.create_base_grid(animate=True)
        windowed_grid, code_group, explanation_text = self.show_overlapping_windows(base_grid)
        self.wait(2.5)

        # --- Scene 4: Im2col Transformation ---
        self.play(FadeOut(windowed_grid), FadeOut(code_group), FadeOut(explanation_text))
        self.show_intro_text(
            "To perform convolution using our fast GEMM kernels, we flatten the windows.\n"
            "This is the 'im2col' transformation, done with MergeTransforms."
        )
        im2col_matrix, code_group = self.show_im2col_transformation(windowed_grid.get_center())
        self.wait(3)
        
        # --- Scene 5: Final TensorView creation ---
        self.play(FadeOut(code_group), FadeOut(im2col_matrix))
        self.show_intro_text("Finally, the descriptor is used with the original buffer\nto create the final TensorView, without copying data.")
        buffer_view = self.create_buffer_view(animate=True)
        final_matrix = self.create_im2col_matrix(ORIGIN + UP, animate=False)
        self.play(FadeIn(final_matrix))
        self.show_tensor_view_creation(final_matrix, buffer_view)
        self.wait(3)


    def show_intro_text(self, text):
        """Displays a full-screen introductory text and fades out."""
        intro = Text(text, font_size=28, line_spacing=1.2).move_to(ORIGIN)
        self.play(Write(intro))
        self.wait(3.5)
        self.play(FadeOut(intro))

    def create_buffer_view(self, animate=False):
        """Creates a VGroup for the buffer_view at the bottom."""
        data = np.arange(1, 37)
        memory_cells = VGroup()
        for i in range(12):
            rect = Rectangle(width=0.35, height=0.5, stroke_color=WHITE, stroke_width=1).set_fill(self.BUFFER_COLOR, 0.5)
            num = Text(str(data[i]), font_size=12).move_to(rect.get_center())
            cell = VGroup(rect, num)
            memory_cells.add(cell)
        
        memory_cells.arrange(RIGHT, buff=0.05)
        dots = Text("...", font_size=14).next_to(memory_cells, RIGHT, buff=0.1)
        buffer_group = VGroup(memory_cells, dots).to_edge(DOWN, buff=0.5)
        label = Text("buffer_view: Linear Memory", font_size=14, color=self.BUFFER_COLOR).next_to(buffer_group, UP, buff=0.2)
        
        if animate:
            self.play(FadeIn(buffer_group), FadeIn(label))
        return VGroup(buffer_group, label)

    def create_code_block(self, lines):
        """Helper to create a styled code block with addressable stride parts."""
        code_vgroup = VGroup()
        strides_parts = []

        for line_text in lines:
            # For lines containing strides, we parse them to highlight individual numbers
            if "strides" in line_text and "[" in line_text and not line_text.strip().startswith("#"):
                prefix_text = line_text.split("[")[0]
                values_str = "[" + line_text.split("[")[1]
                
                line = VGroup(Text(prefix_text, font_size=self.CODE_FONT_SIZE, font="Monospace"))
                
                value_list = values_str.strip("[] ").split(",")
                
                line.add(Text("[", font_size=self.CODE_FONT_SIZE, font="Monospace"))
                for i, val_str in enumerate(value_list):
                    part = Text(val_str.strip(), font_size=self.CODE_FONT_SIZE, font="Monospace")
                    strides_parts.append(part)
                    line.add(part)
                    if i < len(value_list) - 1:
                        line.add(Text(",", font_size=self.CODE_FONT_SIZE, font="Monospace"))
                line.add(Text("]", font_size=self.CODE_FONT_SIZE, font="Monospace"))

                line.arrange(RIGHT, buff=0.08)
            else:
                # Regular lines are just single Text objects
                line = Text(line_text, font_size=self.CODE_FONT_SIZE, font="Monospace")
            code_vgroup.add(line)

        code_vgroup.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        
        # Colorize lines based on content
        for i, line_text in enumerate(lines):
            target_line = code_vgroup[i]
            if "lengths" in line_text: target_line.set_color(self.LENGTH_COLOR)
            if "strides" in line_text: target_line.set_color(self.STRIDE_COLOR)
            if "#" in line_text: target_line.set_color(GRAY)
        
        box = SurroundingRectangle(code_vgroup, buff=0.25, stroke_color=WHITE, stroke_width=1, fill_color="#2b2b2b", fill_opacity=0.9)
        return VGroup(box, code_vgroup), VGroup(*strides_parts)
    
    def create_base_grid(self, animate=False):
        """Creates the 6x6 base grid, optionally animating it."""
        grid_cells = VGroup()
        data = np.arange(1, 37).reshape(6, 6)
        for i in range(6):
            row = VGroup(*[
                VGroup(
                    Rectangle(width=0.5, height=0.5, stroke_color=WHITE, stroke_width=1, fill_color=self.TENSOR_COLOR, fill_opacity=0.3),
                    Text(str(data[i][j]), font_size=14)
                ) for j in range(6)
            ])
            grid_cells.add(row)
        
        grid_vgroup = VGroup(*[row.copy().arrange(RIGHT, buff=0) for row in grid_cells]).arrange(DOWN, buff=0).to_edge(LEFT, buff=0.8).shift(UP*0.2)
        if animate:
            self.play(FadeIn(grid_vgroup))
        return grid_vgroup

    def transform_to_2d(self, buffer_view):
        """Animates buffer transforming to a 2D grid."""
        grid_cells = self.create_base_grid(animate=False)

        code_group, _ = self.create_code_block([
            "# Define the descriptor for a 2D view",
            "lengths = [6, 6]",
            "strides = [6, 1]",
            "desc = make_naive_tensor_descriptor(lengths, strides)",
            "view = make_tensor_view(data=buffer_view, tensor_desc=desc)"
        ])
        code_group.to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        explanation_text = Text(
            "strides=[6, 1]:\n• Jump 6 elements for next row.\n• Jump 1 element for next col.",
            font_size=self.EXPLANATION_FONT_SIZE, color=GRAY, line_spacing=1.2
        ).next_to(code_group, DOWN, buff=0.4, aligned_edge=LEFT)

        self.play(
            FadeIn(grid_cells),
            FadeIn(code_group, shift=LEFT),
            FadeIn(explanation_text, shift=LEFT)
        )
        return grid_cells, code_group, explanation_text

    def show_non_overlapping_tiles(self, base_grid):
        """Explains strides and then shows the final tiled grid."""
        code_group, strides_parts = self.create_code_block([
            "# Define a 4D descriptor for non-overlapping tiles",
            "lengths = [3, 3, 2, 2]",
            "strides = [12, 2, 6, 1]",
            "tiles_desc = make_naive_tensor_descriptor(lengths, strides)",
            "tiles_view = make_tensor_view(data=buffer_view, tensor_desc=tiles_desc)"
        ])
        code_group.to_edge(RIGHT, buff=0.8).shift(UP * 1.0)
        self.play(FadeIn(code_group))
        
        explanation_inter = self.explain_strides(["strides for tiles:", "• Move between large tiles."], [], code_group)
        self.play(FadeIn(explanation_inter))
        self.animate_stride_movement(base_grid, strides_parts, 0, 1, 2, self.TILE_COLOR, "stride = 12 (2*6)", "stride = 2 (2*1)", tile_size=2)
        self.play(FadeOut(explanation_inter))

        explanation_intra = self.explain_strides(["strides within tile:", "• Move between elements inside one tile."], [], code_group)
        self.play(FadeIn(explanation_intra))
        self.animate_stride_movement(base_grid, strides_parts, 2, 3, 1, self.TILE_COLOR, "stride = 6", "stride = 1", tile_size=2, is_intra=True)
        
        # Fade out base grid before showing tiled grid
        self.play(FadeOut(base_grid), FadeOut(explanation_intra))
        tiled_grid_vis = self.create_tiled_grid_vis(base_grid.get_center())
        self.play(FadeIn(tiled_grid_vis))
        
        # Bring back the explanation
        self.play(FadeIn(explanation_intra))
        
        return tiled_grid_vis, code_group, explanation_intra

    def show_overlapping_windows(self, base_grid):
        """Explains strides and then shows the final windowed grid."""
        code_group, strides_parts = self.create_code_block([
            "# Define a 4D descriptor for overlapping windows",
            "lengths = [4, 4, 3, 3]",
            "strides = [6, 1, 6, 1]",
            "windows_desc = make_naive_tensor_descriptor(lengths, strides)",
            "windows_view = make_tensor_view(data=buffer_view, tensor_desc=windows_desc)"
        ])
        code_group.to_edge(RIGHT, buff=0.8).shift(UP * 1.0)
        self.play(FadeIn(code_group))

        explanation_inter = self.explain_strides(["strides for windows:", "• Move between window start positions."], [], code_group)
        self.play(FadeIn(explanation_inter))
        self.animate_stride_movement(base_grid, strides_parts, 0, 1, 1, self.WINDOW_COLOR, "stride = 6 (1*6)", "stride = 1 (1*1)", tile_size=3)
        self.play(FadeOut(explanation_inter))

        explanation_intra = self.explain_strides(["strides within window:", "• Move between elements inside one window."], [], code_group)
        self.play(FadeIn(explanation_intra))
        self.animate_stride_movement(base_grid, strides_parts, 2, 3, 1, self.WINDOW_COLOR, "stride = 6", "stride = 1", tile_size=3, is_intra=True)

        # Fade out base grid before showing windowed grid
        self.play(FadeOut(base_grid), FadeOut(explanation_intra))
        windowed_grid_vis = self.create_windowed_grid_vis(base_grid.get_center())
        # self.play(FadeIn(windowed_grid_vis))
        
        # Bring back the explanation
        # self.play(FadeIn(explanation_intra))
        
        return windowed_grid_vis, code_group, explanation_intra

    def show_im2col_transformation(self, position):
        """Animates the im2col transformation using MergeTransforms."""
        windowed_grid = self.create_windowed_grid_vis(position, animate=False)
        self.play(FadeIn(windowed_grid))

        code_group, _ = self.create_code_block([
            "# Define transforms to flatten dimensions",
            "merge_windows = make_merge_transform([4, 4])",
            "merge_patch = make_merge_transform([3, 3])",
            "im2col_desc = transform_tensor_descriptor(",
            "    windows_desc,",
            "    transforms=[merge_windows, merge_patch],",
            "    lower_dimension_hidden_idss=[[0, 1], [2, 3]],",
            "    upper_dimension_hidden_idss=[[0], [1]]",
            ")",
            "# Final lengths: [16, 9]"
        ])
        code_group.to_edge(RIGHT, buff=0.8).shift(UP * 0.5)
        self.play(FadeIn(code_group))

        im2col_matrix = self.create_im2col_matrix(position, animate=False)
        
        # Animate first 3 rows one by one for clarity
        self.play(
            Transform(windowed_grid[0][0], im2col_matrix[0]),
            run_time=1.5
        )
        self.wait(0.5)
        
        self.play(
            Transform(windowed_grid[0][1], im2col_matrix[1]),
            run_time=1.5
        )
        self.wait(0.5)
        
        self.play(
            Transform(windowed_grid[0][2], im2col_matrix[2]),
            run_time=1.5
        )
        self.wait(1)
        
        # Now animate the rest with lag for visual appeal
        self.play(
            LaggedStart(*[
                Transform(windowed_grid[i][j], im2col_matrix[i*4+j])
                for i in range(4) for j in range(4)
                if i*4+j >= 3  # Skip the first 3 we already animated
            ], lag_ratio=0.08, run_time=4)
        )
        self.wait(1)
        self.remove(windowed_grid)
        # self.add(im2col_matrix)
        return im2col_matrix, code_group
        
    def show_tensor_view_creation(self, im2col_matrix, buffer_view):
        """Shows the final step of creating a TensorView."""
        code_group, _ = self.create_code_block([
            "# Final step: Create the TensorView",
            "im2col_view = make_tensor_view(",
            "    data=buffer_view,",
            "    tensor_desc=im2col_desc",
            ")"
        ])
        code_group.to_edge(RIGHT, buff=0.8).shift(UP * 1.5)
        self.play(FadeIn(code_group))

        explanation = Text(
            "The final view is a zero-copy interpretation\nof the original linear data.",
            font_size=self.EXPLANATION_FONT_SIZE, color=GRAY
        ).next_to(code_group, DOWN, buff=0.4)
        self.play(Write(explanation))

        arrow = Arrow(buffer_view.get_center(), im2col_matrix.get_center(), color=YELLOW, buff=0.5)
        self.play(GrowArrow(arrow))
        self.play(
            Indicate(buffer_view, color=self.BUFFER_COLOR),
            Indicate(im2col_matrix, color=self.WINDOW_COLOR),
            run_time=2
        )
        self.play(FadeOut(arrow))

    def create_tiled_grid_vis(self, position):
        """Creates the full 3x3 grid of 2x2 tiles."""
        data = np.arange(1, 37).reshape(6, 6)
        tiled_grid = VGroup()
        for ti in range(3):
            tile_row = VGroup()
            for tj in range(3):
                tile = VGroup()
                for i in range(2):
                    row = VGroup(*[
                        VGroup(
                            Rectangle(width=0.5, height=0.5, stroke_color=WHITE, stroke_width=1, fill_color=self.TILE_COLOR, fill_opacity=0.4),
                            Text(str(data[ti*2+i][tj*2+j]), font_size=14)
                        ) for j in range(2)
                    ]).arrange(RIGHT, buff=0)
                    tile.add(row)
                tile.arrange(DOWN, buff=0)
                tile_row.add(tile)
            tiled_grid.add(tile_row.arrange(RIGHT, buff=0.1))
        tiled_grid.arrange(DOWN, buff=0.1).move_to(position)
        return tiled_grid


    def create_windowed_grid_vis(self, position, animate=True):
        """Creates the full 4x4 grid of 3x3 overlapping windows."""
        data = np.arange(1, 37).reshape(6, 6)
        window_grid = VGroup()
        for wi in range(4):
            win_row = VGroup()
            for wj in range(4):
                window = VGroup()
                for i in range(3):
                    row = VGroup(*[
                        VGroup(
                            Rectangle(width=0.35, height=0.35, stroke_color=self.WINDOW_COLOR, stroke_width=1, fill_color=self.WINDOW_COLOR, fill_opacity=0.2),
                            Text(str(data[wi+i][wj+j]), font_size=10)
                        ) for j in range(3)
                    ]).arrange(RIGHT, buff=0.05)
                    window.add(row)
                window.arrange(DOWN, buff=0.05)
                win_row.add(window)
            window_grid.add(win_row.arrange(RIGHT, buff=0.1))
        window_grid.arrange(DOWN, buff=0.1).move_to(position).scale(0.8)
        if animate:
            self.play(FadeIn(window_grid))
        return window_grid

    def create_im2col_matrix(self, position, animate=True):
        """Creates the 16x9 im2col matrix visualization."""
        data = np.arange(1, 37).reshape(6, 6)
        matrix = VGroup()
        for i in range(4):
            for j in range(4):
                window_data = data[i:i+3, j:j+3].flatten()
                row = VGroup(*[
                    VGroup(
                        Rectangle(width=0.3, height=0.3, stroke_width=0.5, color=GRAY),
                        Text(str(val), font_size=9)
                    ) for val in window_data
                ]).arrange(RIGHT, buff=0.05)
                matrix.add(row)
        
        matrix.arrange(DOWN, buff=0.05).move_to(position).scale(0.9)
        if animate:
            self.play(FadeIn(matrix))
        return matrix

    def explain_strides(self, part1_lines, part2_lines, code_group):
        """Creates a VGroup with detailed stride explanations."""
        part1 = VGroup(*[Text(line, font_size=self.EXPLANATION_FONT_SIZE, color=GRAY) for line in part1_lines]).arrange(DOWN, aligned_edge=LEFT)
        parts = [part1]
        if part2_lines:
            part2 = VGroup(*[Text(line, font_size=self.EXPLANATION_FONT_SIZE, color=GRAY) for line in part2_lines]).arrange(DOWN, aligned_edge=LEFT)
            parts.append(part2)
        
        return VGroup(*parts).arrange(DOWN, buff=0.4, aligned_edge=LEFT).next_to(code_group, DOWN, buff=0.4, aligned_edge=LEFT)

    def animate_stride_movement(self, grid, strides_parts, v_idx, h_idx, jump, color, v_text, h_text, tile_size=2, is_intra=False):
        """Animates stride movements visually and highlights the corresponding code."""
        origin_i, origin_j = (0, 0)
        
        target_group = VGroup(*[grid[i][j] for i in range(tile_size) for j in range(tile_size)]) if not is_intra else grid[origin_i][origin_j]
        
        # --- Vertical Stride ---
        tile_v = SurroundingRectangle(target_group, buff=0.05, color=color, stroke_width=3)
        v_highlight = SurroundingRectangle(strides_parts[v_idx], buff=0.05, color=YELLOW)
        self.play(Create(tile_v), Create(v_highlight))
        
        v_arrow = Arrow(grid[origin_i][origin_j].get_center(), grid[origin_i+jump][origin_j].get_center(), buff=0.2, color=self.STRIDE_COLOR)
        v_label = Text(v_text, font_size=14, color=self.STRIDE_COLOR).next_to(grid, UP, buff=0.2)
        v_target_group = VGroup(*[grid[origin_i+jump+i][origin_j+j] for i in range(tile_size if not is_intra else 1) for j in range(tile_size if not is_intra else 1)])
        
        self.play(GrowArrow(v_arrow), Write(v_label))
        self.play(tile_v.animate.move_to(v_target_group.get_center()))
        self.play(FadeOut(v_arrow), FadeOut(v_label), FadeOut(tile_v), FadeOut(v_highlight))
        self.wait(0.5)

        # --- Horizontal Stride ---
        tile_h = SurroundingRectangle(target_group, buff=0.05, color=color, stroke_width=3)
        h_highlight = SurroundingRectangle(strides_parts[h_idx], buff=0.05, color=YELLOW)
        self.play(Create(tile_h), Create(h_highlight))
        
        h_arrow = Arrow(grid[origin_i][origin_j].get_center(), grid[origin_i][origin_j+jump].get_center(), buff=0.2, color=self.STRIDE_COLOR)
        h_label = Text(h_text, font_size=14, color=self.STRIDE_COLOR).next_to(grid, UP, buff=0.2)
        h_target_group = VGroup(*[grid[origin_i+i][origin_j+jump+j] for i in range(tile_size if not is_intra else 1) for j in range(tile_size if not is_intra else 1)])

        self.play(GrowArrow(h_arrow), Write(h_label))
        self.play(tile_h.animate.move_to(h_target_group.get_center()))
        
        self.play(FadeOut(h_arrow), FadeOut(h_label), FadeOut(tile_h), FadeOut(h_highlight))
        self.wait(1)

# To render, run in terminal:
# manim -pql convolution_tensor_descriptor_animation.py ConvolutionTensorDescriptorAnimation