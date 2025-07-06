from manim import *
import numpy as np
import sys
import os

# Add project root to sys.path to allow importing pytensor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytensor.space_filling_curve import SpaceFillingCurve


class DetailedCalculationDemo(Scene):
    """
    Show one detailed calculation step-by-step with large text.
    """
    
    def construct(self):
        # Title
        title = Text("Space-Filling Curve: Detailed Calculation", font_size=28)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Configuration
        tensor_lengths = [3, 4]
        dim_access_order = [0, 1] 
        scalars_per_access = [1, 1]
        access_idx = 5  # Show calculation for access index 5
        
        # Create space-filling curve
        sfc = SpaceFillingCurve(
            tensor_lengths=tensor_lengths,
            dim_access_order=dim_access_order,
            scalars_per_access=scalars_per_access,
            snake_curved=False
        )
        
        # Show configuration
        config = VGroup(
            Text(f"tensor_lengths = {tensor_lengths}", font_size=18),
            Text(f"dim_access_order = {dim_access_order}", font_size=18),
            Text(f"scalars_per_access = {scalars_per_access}", font_size=18),
            Text(f"access_lengths = {sfc.access_lengths}", font_size=18),
            Text(f"ordered_access_lengths = {sfc.ordered_access_lengths}", font_size=18),
        )
        config.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        config.move_to(UP * 1.5)
        
        self.play(Write(config))
        self.wait(2)
        
        # Show the detailed calculation
        self.show_detailed_calculation(sfc, access_idx)
    
    def show_detailed_calculation(self, sfc, access_idx):
        """Show one calculation in great detail."""
        
        # Clear screen except title and config
        # (config stays visible)
        
        # Step 1: Show what we're calculating
        input_text = Text(f"Calculating get_index({access_idx})", font_size=24, color=YELLOW)
        input_text.move_to(DOWN * 0.5)
        self.play(Write(input_text))
        self.wait(1)
        
        # Step 2: Show the algorithm
        step1_text = Text("Step 1: Calculate ordered indices using division", font_size=20, color=BLUE)
        step1_text.move_to(DOWN * 1.2)
        self.play(Transform(input_text, step1_text))
        self.wait(1)
        
        # Show the division process step by step
        ordered_indices = []
        remaining = access_idx
        
        remaining_text = Text(f"remaining = {access_idx}", font_size=18)
        remaining_text.move_to(DOWN * 1.8)
        self.play(Write(remaining_text))
        self.wait(1)
        
        # Process each dimension
        for i in range(len(sfc.ordered_access_lengths) - 1, -1, -1):
            length = sfc.ordered_access_lengths[i]
            idx = remaining % length
            ordered_indices.insert(0, idx)
            
            # Show the calculation
            calc_text = Text(f"dim {i}: {remaining} % {length} = {idx}, remaining = {remaining // length}", font_size=18)
            calc_text.move_to(DOWN * (2.2 + (len(sfc.ordered_access_lengths) - 1 - i) * 0.4))
            self.play(Write(calc_text))
            self.wait(1)
            
            remaining //= length
        
        # Show the result of step 1
        result1_text = Text(f"ordered_indices = {ordered_indices}", font_size=20, color=GREEN)
        result1_text.move_to(DOWN * 3.2)
        self.play(Write(result1_text))
        self.wait(2)
        
        # Clear for step 2
        self.play(FadeOut(VGroup(*[mob for mob in self.mobjects if mob.get_center()[1] < 0])))
        
        # Step 2: Map back to original dimensions
        step2_text = Text("Step 2: Map to original dimensions", font_size=20, color=BLUE)
        step2_text.move_to(DOWN * 0.5)
        self.play(Write(step2_text))
        self.wait(1)
        
        # Show the mapping
        indices = [0] * sfc.ndim
        for i, ordered_idx in enumerate(ordered_indices):
            original_dim = sfc.dim_access_order[i]
            indices[original_dim] = ordered_idx * sfc.scalars_per_access[original_dim]
            
            map_text = Text(f"indices[{original_dim}] = {ordered_idx} * {sfc.scalars_per_access[original_dim]} = {indices[original_dim]}", font_size=18)
            map_text.move_to(DOWN * (1.0 + i * 0.4))
            self.play(Write(map_text))
            self.wait(1)
        
        # Final result
        final_result = Text(f"RESULT: get_index({access_idx}) = {indices}", font_size=24, color=RED)
        final_result.move_to(DOWN * 2.5)
        self.play(Write(final_result))
        self.wait(3)


class SimpleSpaceFillingCurveDemo(Scene):
    """
    Simple demonstration showing get_index calculations step by step.
    """
    
    def construct(self):
        # Title
        title = Text("Space-Filling Curve: get_index() Calculations", font_size=24)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Simple configuration
        tensor_lengths = [3, 4]
        dim_access_order = [0, 1] 
        scalars_per_access = [1, 1]
        
        # Create space-filling curve
        sfc = SpaceFillingCurve(
            tensor_lengths=tensor_lengths,
            dim_access_order=dim_access_order,
            scalars_per_access=scalars_per_access,
            snake_curved=False
        )
        
        # Show configuration
        config = VGroup(
            Text(f"tensor_lengths = {tensor_lengths}", font_size=16),
            Text(f"dim_access_order = {dim_access_order}", font_size=16),
            Text(f"scalars_per_access = {scalars_per_access}", font_size=16),
            Text(f"access_lengths = {sfc.access_lengths}", font_size=16),
            Text(f"ordered_access_lengths = {sfc.ordered_access_lengths}", font_size=16),
        )
        config.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        config.move_to(UP * 1.5)
        
        self.play(Write(config))
        self.wait(2)
        
        # Initialize calc_group to None
        self.calc_group = None
        
        # Show calculations for a few access indices
        for access_idx in [0, 1, 2, 3, 4, 5]:
            self.show_calculation_step(sfc, access_idx)
            self.wait(2)
    
    def show_calculation_step(self, sfc, access_idx):
        """Show one calculation step clearly."""
        
        # Clear previous calculation if it exists
        if self.calc_group is not None:
            self.play(FadeOut(self.calc_group), run_time=0.5)
        
        # Create calculation display
        calc_lines = []
        
        # Step 1: Show input
        calc_lines.append(f"get_index({access_idx}):")
        calc_lines.append("")
        
        # Step 2: Calculate ordered indices
        ordered_indices = []
        remaining = access_idx
        
        calc_lines.append("Step 1: Calculate ordered indices")
        calc_lines.append(f"remaining = {access_idx}")
        
        # Show division process
        for i in range(len(sfc.ordered_access_lengths) - 1, -1, -1):
            length = sfc.ordered_access_lengths[i]
            idx = remaining % length
            ordered_indices.insert(0, idx)
            calc_lines.append(f"  dim {i}: {remaining} % {length} = {idx}, remaining = {remaining // length}")
            remaining //= length
        
        calc_lines.append(f"ordered_indices = {ordered_indices}")
        calc_lines.append("")
        
        # Step 3: Map back to original dimensions
        calc_lines.append("Step 2: Map to original dimensions")
        indices = [0] * sfc.ndim
        for i, ordered_idx in enumerate(ordered_indices):
            original_dim = sfc.dim_access_order[i]
            indices[original_dim] = ordered_idx * sfc.scalars_per_access[original_dim]
            calc_lines.append(f"  indices[{original_dim}] = {ordered_idx} * {sfc.scalars_per_access[original_dim]} = {indices[original_dim]}")
        
        calc_lines.append("")
        calc_lines.append(f"RESULT: {indices}")
        
        # Create text objects
        calc_texts = VGroup()
        for line in calc_lines:
            if line == "":
                calc_texts.add(Text(" ", font_size=14))  # Empty line
            elif "RESULT:" in line:
                calc_texts.add(Text(line, font_size=16, color=RED))
            elif "Step" in line:
                calc_texts.add(Text(line, font_size=14, color=BLUE))
            elif "get_index" in line:
                calc_texts.add(Text(line, font_size=16, color=YELLOW))
            else:
                calc_texts.add(Text(line, font_size=14))
        
        calc_texts.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        calc_texts.move_to(DOWN * 1)
        
        # Update the calc_group reference
        self.calc_group = calc_texts
        self.play(Write(calc_texts), run_time=1)


class CompareAccessOrders(Scene):
    """
    Compare two different access orders side by side.
    """
    
    def construct(self):
        title = Text("Comparing Different Access Orders", font_size=24)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Two configurations
        config1 = {'tensor_lengths': [3, 4], 'dim_access_order': [0, 1], 'scalars_per_access': [1, 1]}
        config2 = {'tensor_lengths': [3, 4], 'dim_access_order': [1, 0], 'scalars_per_access': [1, 1]}
        
        sfc1 = SpaceFillingCurve(**config1, snake_curved=False)
        sfc2 = SpaceFillingCurve(**config2, snake_curved=False)
        
        # Left side
        left_title = Text("Access Order [0, 1]", font_size=18, color=BLUE)
        left_title.move_to(LEFT * 4 + UP * 2)
        
        # Right side
        right_title = Text("Access Order [1, 0]", font_size=18, color=RED)
        right_title.move_to(RIGHT * 4 + UP * 2)
        
        self.play(Write(left_title), Write(right_title))
        
        # Show results
        left_results = []
        right_results = []
        
        for i in range(12):  # 3*4 = 12 total
            left_idx = sfc1.get_index(i)
            right_idx = sfc2.get_index(i)
            left_results.append(f"{i}: {left_idx}")
            right_results.append(f"{i}: {right_idx}")
        
        left_text = Text("\n".join(left_results), font_size=12)
        left_text.move_to(LEFT * 4 + DOWN * 0.5)
        
        right_text = Text("\n".join(right_results), font_size=12)
        right_text.move_to(RIGHT * 4 + DOWN * 0.5)
        
        self.play(Write(left_text), Write(right_text), run_time=2)
        self.wait(3)


class ShowVectorization(Scene):
    """
    Show effect of scalars_per_access.
    """
    
    def construct(self):
        title = Text("Effect of scalars_per_access", font_size=24)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Two configurations
        config1 = {'tensor_lengths': [4, 3], 'dim_access_order': [0, 1], 'scalars_per_access': [1, 1]}
        config2 = {'tensor_lengths': [4, 3], 'dim_access_order': [0, 1], 'scalars_per_access': [2, 1]}
        
        sfc1 = SpaceFillingCurve(**config1, snake_curved=False)
        sfc2 = SpaceFillingCurve(**config2, snake_curved=False)
        
        # Left side
        left_title = Text("scalars_per_access = [1, 1]", font_size=16, color=BLUE)
        left_title.move_to(LEFT * 4 + UP * 2.5)
        
        left_info = Text(f"access_lengths = {sfc1.access_lengths}\ntotal_accesses = {sfc1.get_num_of_access()}", font_size=12)
        left_info.next_to(left_title, DOWN, buff=0.3)
        
        # Right side
        right_title = Text("scalars_per_access = [2, 1]", font_size=16, color=RED)
        right_title.move_to(RIGHT * 4 + UP * 2.5)
        
        right_info = Text(f"access_lengths = {sfc2.access_lengths}\ntotal_accesses = {sfc2.get_num_of_access()}", font_size=12)
        right_info.next_to(right_title, DOWN, buff=0.3)
        
        self.play(Write(left_title), Write(right_title))
        self.play(Write(left_info), Write(right_info))
        self.wait(1)
        
        # Show results
        left_results = []
        right_results = []
        
        for i in range(min(8, sfc1.get_num_of_access())):
            left_idx = sfc1.get_index(i)
            left_results.append(f"{i}: {left_idx}")
        
        for i in range(min(8, sfc2.get_num_of_access())):
            right_idx = sfc2.get_index(i)
            right_results.append(f"{i}: {right_idx}")
        
        left_text = Text("\n".join(left_results), font_size=12)
        left_text.next_to(left_info, DOWN, buff=0.5)
        
        right_text = Text("\n".join(right_results), font_size=12)
        right_text.next_to(right_info, DOWN, buff=0.5)
        
        self.play(Write(left_text), Write(right_text), run_time=2)
        self.wait(3) 