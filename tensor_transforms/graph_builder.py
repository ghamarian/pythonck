#!/usr/bin/env python3
"""
Unified graph builder for tensor transformations.
Builds both Lower→Upper and Upper→Lower transformation graphs.
"""

import graphviz
import sympy as sp
from typing import Dict, List, Any, Tuple, Optional, Set
import pytensor.tensor_descriptor as pytd


class HiddenDimensionTracker:
    """Tracks active dimensions in hidden dimension space."""

    def __init__(self):
        self.active_nodes = {}  # hidden_dim_idx -> node_id
        self.node_formulas = {}  # node_id -> sympy expression
        self.node_counter = 0

    def create_node(self, hidden_idx: int, stage: int, transform_idx: int, formula: sp.Expr = None) -> str:
        """Create a new node at the given hidden dimension index."""
        node_id = f"s{stage}_t{transform_idx}_d{hidden_idx}"
        self.active_nodes[hidden_idx] = node_id
        if formula is not None:
            self.node_formulas[node_id] = formula
        return node_id

    def get_node(self, hidden_idx: int) -> Optional[str]:
        """Get the node ID at the given hidden dimension index."""
        return self.active_nodes.get(hidden_idx)

    def get_formula(self, node_id: str) -> Optional[sp.Expr]:
        """Get the formula for a node."""
        return self.node_formulas.get(node_id)

    def update_active(self, hidden_indices: List[int], node_ids: List[str]):
        """Update active nodes for the given indices."""
        for idx, node_id in zip(hidden_indices, node_ids):
            self.active_nodes[idx] = node_id


class TransformStage:
    """Represents a stage in the transformation pipeline."""

    def __init__(self, stage_num: int):
        self.stage_num = stage_num
        self.transforms = []  # List of (transform, lower_ids, upper_ids, node_infos)

    def add_transform(self, transform, lower_ids, upper_ids, node_infos):
        """Add a transform with its node information."""
        self.transforms.append((transform, lower_ids, upper_ids, node_infos))


def get_transform_info(descriptor):
    """Extract transform information from a descriptor."""
    transforms = descriptor.get_transforms()
    lower_idss = descriptor.get_lower_dimension_hidden_idss()
    upper_idss = descriptor.get_upper_dimension_hidden_idss()
    return list(zip(transforms, lower_idss, upper_idss))


def apply_transform_forward(transform, input_symbols: List[sp.Expr]) -> List[sp.Expr]:
    """Apply a transform in the forward direction (lower -> upper)."""
    if hasattr(transform, 'sympy_calculate_upper'):
        try:
            result = transform.sympy_calculate_upper(input_symbols)
            return result if result else [sp.Symbol(f"y{i}") for i in range(len(input_symbols))]
        except Exception as e:
            print(f"Forward transform failed: {e}")
    # Fallback
    return [sp.Symbol(f"y{i}") for i in range(len(input_symbols))]


def apply_transform_inverse(transform, input_symbols: List[sp.Expr]) -> List[sp.Expr]:
    """Apply a transform in the inverse direction (upper -> lower)."""
    if hasattr(transform, 'sympy_calculate_lower'):
        try:
            result = transform.sympy_calculate_lower(input_symbols)
            return result if result else [sp.Symbol(f"x{i}") for i in range(len(input_symbols))]
        except Exception as e:
            print(f"Inverse transform failed: {e}")
    # Fallback
    return [sp.Symbol(f"x{i}") for i in range(len(input_symbols))]


def parse_descriptors(descriptors, variables, original_code=None):
    """Parse descriptors and return pytensor descriptors with proper names."""
    from tensor_transforms import TensorTransformParser
    import re

    parser = TensorTransformParser()
    pytensor_descriptors = []
    descriptor_registry = {}

    # Try to extract names from original code if provided
    original_names = {}
    if original_code:
        pattern = re.compile(
            r'constexpr\s+auto\s+(\w+)\s*=\s*(transform_tensor_descriptor|make_naive_tensor_descriptor_packed|make_naive_tensor_descriptor)\s*\(',
            re.MULTILINE | re.DOTALL
        )
        descriptor_index = 0
        for match in pattern.finditer(original_code):
            original_names[descriptor_index] = match.group(1)
            descriptor_index += 1

    for i, desc_str in enumerate(descriptors):
        # Get the name for this descriptor
        if i in original_names:
            name = original_names[i]
        else:
            # Fall back to inferring from the descriptor itself
            name_match = re.search(r'constexpr\s+auto\s+(\w+)\s*=', desc_str)
            if name_match:
                name = name_match.group(1)
            else:
                # Try to infer from reference
                ref_match = re.search(r'transform_tensor_descriptor\s*\(\s*(\w+)', desc_str)
                if ref_match:
                    ref_name = ref_match.group(1)
                    name = f'{ref_name}_t{i}'
                else:
                    name = f'desc_{i}'

        try:
            parser.descriptor_registry = descriptor_registry
            tensor_desc = parser.create_pytensor_descriptor(desc_str, variables)
            pytensor_descriptors.append(tensor_desc)

            # Register with inferred name
            descriptor_registry[name] = tensor_desc
            # Also register with generic name
            descriptor_registry[f'desc_{i}'] = tensor_desc

        except Exception as e:
            print(f"Failed to create descriptor {i}: {e}")
            continue

    return pytensor_descriptors


def build_lower_to_upper_graph(descriptors, variables, original_code=None):
    """
    Build a graph showing Lower → Upper transformations as specified in C++ code.

    This shows the FORWARD direction through all transforms, starting from
    the lower (physical) representation and ending at the upper (logical) representation.
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="ortho")
    dot.attr('node', shape='box', style='rounded,filled')

    # Title
    vars_display = ", ".join(f"{k}={v}" for k, v in sorted(variables.items())[:3])
    if len(variables) > 3:
        vars_display += "..."
    dot.node("title", f"Lower → Upper Graph\\n{vars_display}",
             shape="note", style="filled", fillcolor="lightyellow")

    if not descriptors:
        return dot

    # Parse descriptors
    pytensor_descriptors = parse_descriptors(descriptors, variables, original_code)

    if not pytensor_descriptors:
        return dot

    tracker = HiddenDimensionTracker()

    # Determine the starting point based on the first descriptor
    first_desc = pytensor_descriptors[0]
    first_desc_str = descriptors[0].strip()

    # Create initial nodes (lower representation)
    with dot.subgraph(name='cluster_start') as cluster:
        cluster.attr(style='filled', fillcolor='#e8e8ff', label='Lower (Start)')

        if "make_naive_tensor_descriptor_packed" in first_desc_str:
            # Packed: starts with storage at dimension 0
            node_id = "start_storage"
            cluster.node(node_id, "storage", fillcolor="#ccccff")
            tracker.active_nodes[0] = node_id
            tracker.node_formulas[node_id] = sp.Symbol("storage")
        elif "make_naive_tensor_descriptor" in first_desc_str:
            # Regular naive: starts with linear address
            node_id = "start_linear"
            cluster.node(node_id, "linear_addr", fillcolor="#ccccff")
            tracker.active_nodes[0] = node_id
            tracker.node_formulas[node_id] = sp.Symbol("linear_addr")
        else:
            # Transform descriptor or other: determine from analysis
            transform_info = get_transform_info(first_desc)
            if transform_info:
                # Find all input indices needed
                all_input_indices = set()
                for _, lower_ids, _ in transform_info:
                    all_input_indices.update(lower_ids)

                for idx in sorted(all_input_indices):
                    node_id = f"start_d{idx}"
                    cluster.node(node_id, f"in{idx}", fillcolor="#ccccff")
                    tracker.active_nodes[idx] = node_id
                    tracker.node_formulas[node_id] = sp.Symbol(f"in{idx}")

    # Process each descriptor in forward order
    stage_num = 0
    processed_transform_count = 0

    for desc_idx, tensor_desc in enumerate(pytensor_descriptors):
        transform_info = get_transform_info(tensor_desc)
        all_transforms_count = len(tensor_desc.get_transforms())

        # For multi-descriptor cases, only process NEW transforms
        if desc_idx > 0:
            prev_desc = pytensor_descriptors[desc_idx - 1]
            prev_count = len(prev_desc.get_transforms())
            transform_info = transform_info[prev_count:]

        if not transform_info:
            continue

        processed_transform_count = all_transforms_count
        stage_num += 1

        # Create stage cluster
        with dot.subgraph(name=f'cluster_stage_{stage_num}') as cluster:
            cluster.attr(style='filled',
                        fillcolor=f'#{"ffeeee" if stage_num % 2 else "eeffee"}',
                        label=f'Stage {stage_num}')

            # Apply transforms in forward order
            for t_idx, (transform, lower_ids, upper_ids) in enumerate(transform_info):
                transform_name = transform.__class__.__name__.replace('Transform', '')

                # Collect input symbols
                input_symbols = []
                input_nodes = []
                for idx in lower_ids:
                    node_id = tracker.get_node(idx)
                    if node_id:
                        input_nodes.append(node_id)
                        formula = tracker.get_formula(node_id)
                        input_symbols.append(formula if formula else sp.Symbol(f"x{idx}"))
                    else:
                        input_symbols.append(sp.Symbol(f"x{idx}"))

                # Apply transform in forward direction
                output_formulas = apply_transform_forward(transform, input_symbols)

                # Create output nodes
                for j, out_idx in enumerate(upper_ids):
                    node_id = tracker.create_node(out_idx, stage_num, t_idx)

                    if j < len(output_formulas):
                        formula = output_formulas[j]
                    else:
                        formula = sp.Symbol(f"y{out_idx}")

                    tracker.node_formulas[node_id] = formula

                    # Simplify and format
                    safe_vars = {k: v for k, v in variables.items()
                                if isinstance(v, (int, float, complex, sp.Basic))}
                    simplified = sp.simplify(formula.subs(safe_vars))
                    label = str(simplified)

                    cluster.node(node_id, label, fillcolor="#c0ffc0")

                    # Create edges from inputs
                    for input_node in input_nodes:
                        dot.edge(input_node, node_id, label=transform_name)

    # Create final output nodes (upper representation)
    final_desc = pytensor_descriptors[-1]

    with dot.subgraph(name='cluster_end') as cluster:
        cluster.attr(style='filled', fillcolor='#e8ffe8', label='Upper (End)')

        top_hidden_ids = final_desc.get_top_dimension_hidden_ids()

        for i, hidden_idx in enumerate(top_hidden_ids):
            node_id = tracker.get_node(hidden_idx)
            if node_id:
                final_node_id = f"upper_d{i}"
                cluster.node(final_node_id, f"d{i}", fillcolor="#66ff66")
                dot.edge(node_id, final_node_id, color="green", style="bold")

    return dot


def build_upper_to_lower_graph(descriptors, variables, original_code=None):
    """
    Build Upper→Lower graph by reversing the Lower→Upper construction.
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR", splines="ortho")
    dot.attr('node', shape='box', style='rounded,filled')

    # Title
    vars_display = ", ".join(f"{k}={v}" for k, v in sorted(variables.items())[:3])
    if len(variables) > 3:
        vars_display += "..."
    dot.node("title", f"Upper → Lower Graph\\n{vars_display}",
             shape="note", style="filled", fillcolor="lightblue")

    if not descriptors:
        return dot

    # Parse descriptors
    pytensor_descriptors = parse_descriptors(descriptors, variables, original_code)

    if not pytensor_descriptors:
        return dot

    # Extract transformation stages from Lower→Upper construction
    stages = []
    tracker = HiddenDimensionTracker()

    # Initialize starting nodes
    first_desc = pytensor_descriptors[0]
    first_desc_str = descriptors[0].strip()
    initial_dims = []

    if "make_naive_tensor_descriptor_packed" in first_desc_str:
        initial_dims = [0]
    elif "make_naive_tensor_descriptor" in first_desc_str:
        initial_dims = [0]
    else:
        transform_info = get_transform_info(first_desc)
        if transform_info:
            all_input_indices = set()
            for _, lower_ids, _ in transform_info:
                all_input_indices.update(lower_ids)
            initial_dims = sorted(all_input_indices)

    # Process each descriptor to extract stages
    stage_num = 0
    processed_transform_count = 0

    for desc_idx, tensor_desc in enumerate(pytensor_descriptors):
        transform_info = get_transform_info(tensor_desc)

        # For multi-descriptor cases, only process NEW transforms
        if processed_transform_count > 0:
            transform_info = transform_info[processed_transform_count:]

        if not transform_info:
            continue

        processed_transform_count += len(transform_info)
        stage_num += 1

        stage = TransformStage(stage_num)

        # Apply transforms in forward order
        for t_idx, (transform, lower_ids, upper_ids) in enumerate(transform_info):
            node_infos = []
            for j, out_idx in enumerate(upper_ids):
                node_infos.append({
                    'hidden_idx': out_idx,
                    'node_id': f"s{stage_num}_t{t_idx}_d{out_idx}",
                    'formula': None,
                    'input_nodes': []
                })

            stage.add_transform(transform, lower_ids, upper_ids, node_infos)

        stages.append(stage)

    # Build Upper→Lower graph using the extracted information
    new_tracker = HiddenDimensionTracker()

    # Create starting nodes (upper representation)
    final_desc = pytensor_descriptors[-1]
    top_hidden_ids = final_desc.get_top_dimension_hidden_ids()

    with dot.subgraph(name='cluster_start') as cluster:
        cluster.attr(style='filled', fillcolor='#ffe8e8', label='Upper (Start)')
        for i, hidden_idx in enumerate(top_hidden_ids):
            node_id = f"start_d{i}"
            cluster.node(node_id, f"d{i}", fillcolor="#ffcccc")
            new_tracker.active_nodes[hidden_idx] = node_id
            new_tracker.node_formulas[node_id] = sp.Symbol(f"d{i}")

    # Process stages in REVERSE order
    reverse_stage_num = 0

    for stage in reversed(stages):
        reverse_stage_num += 1

        with dot.subgraph(name=f'cluster_stage_{reverse_stage_num}') as cluster:
            cluster.attr(style='filled',
                        fillcolor=f'#{"eeeeff" if reverse_stage_num % 2 else "ffffe0"}',
                        label=f'Stage {reverse_stage_num}')

            # Process transforms in REVERSE order within the stage
            for transform, lower_ids, upper_ids, node_infos in reversed(stage.transforms):
                transform_name = transform.__class__.__name__.replace('Transform', '')

                # Collect input symbols and nodes
                input_symbols = []
                input_nodes = []

                for idx in upper_ids:
                    node_id = new_tracker.get_node(idx)
                    if node_id:
                        input_nodes.append(node_id)
                        formula = new_tracker.get_formula(node_id)
                        input_symbols.append(formula if formula else sp.Symbol(f"d{idx}"))
                    else:
                        input_symbols.append(sp.Symbol(f"d{idx}"))

                # Apply transform in inverse direction
                output_formulas = apply_transform_inverse(transform, input_symbols)

                # Create output nodes
                for j, out_idx in enumerate(lower_ids):
                    # Find the original transform index
                    original_t_idx = 0
                    for info in node_infos:
                        if info['node_id'].startswith(f"s{stage.stage_num}_t"):
                            parts = info['node_id'].split('_')
                            original_t_idx = int(parts[1][1:])
                            break

                    node_id = new_tracker.create_node(out_idx, reverse_stage_num, original_t_idx)

                    if j < len(output_formulas):
                        formula = output_formulas[j]
                    else:
                        formula = sp.Symbol(f"x{out_idx}")

                    new_tracker.node_formulas[node_id] = formula

                    # Simplify and format
                    safe_vars = {k: v for k, v in variables.items()
                                if isinstance(v, (int, float, complex, sp.Basic))}
                    simplified = sp.simplify(formula.subs(safe_vars))
                    label = str(simplified)

                    cluster.node(node_id, label, fillcolor="#ffc0c0")

                    # Create edges from ALL input nodes
                    for input_node in input_nodes:
                        dot.edge(input_node, node_id, label=f"{transform_name}⁻¹")

    # Create final output nodes (lower representation)
    with dot.subgraph(name='cluster_end') as cluster:
        cluster.attr(style='filled', fillcolor='#e8e8ff', label='Lower (End)')

        if "make_naive_tensor_descriptor_packed" in first_desc_str:
            # Single storage output
            node_id = new_tracker.get_node(0)
            if node_id:
                final_node_id = "lower_storage"
                cluster.node(final_node_id, "storage", fillcolor="#6666ff")
                dot.edge(node_id, final_node_id, color="blue", style="bold")
        elif "make_naive_tensor_descriptor" in first_desc_str:
            # Single linear address output
            node_id = new_tracker.get_node(0)
            if node_id:
                final_node_id = "lower_linear"
                cluster.node(final_node_id, "linear_addr", fillcolor="#6666ff")
                dot.edge(node_id, final_node_id, color="blue", style="bold")
        else:
            # Use only the initial dimensions
            for i, idx in enumerate(initial_dims):
                node_id = new_tracker.get_node(idx)
                if node_id:
                    final_node_id = f"lower_d{i}"
                    cluster.node(final_node_id, f"out{i}", fillcolor="#6666ff")
                    dot.edge(node_id, final_node_id, color="blue", style="bold")

    return dot
