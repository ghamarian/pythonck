"""
Pedantic tile distribution module for the Composable Kernels visualizer.

This module implements the tile distribution functionality by closely
following the C++ implementation in tile_distribution.hpp and
tile_distribution_encoding.hpp.
"""

import typing
import numpy as np
import json

# Helper to mimic ck_tile::remove_cvref_t (essentially, just use the type)
def remove_cvref_t(t):
    return t

# Helper to mimic ck_tile::number<N> (we just use the integer N directly in Python)
def Number(val: int) -> int:
    return val

# Helper to mimic ck_tile::sequence<Is...> (Python list)
# In C++, sequences can also be of types. Here we assume sequences of values.
def Sequence(*args: int) -> typing.List[int]:
    return list(args)

# Helper to mimic ck_tile::tuple<Ts...> (Python tuple or list)
# In C++, tuples can hold heterogeneous types. Python tuples do too.
def Tuple(*args) -> typing.Union[typing.Tuple[typing.Any, ...], typing.List[typing.Any]]: # Using List for HsLengthss etc.
    return list(args) # Using list for mutable sequences of sequences

# Helper to mimic ck_tile::array (Python list with fixed size expectation or numpy array)
# For now, Python lists will be used. Size checks can be added.
def Array(size: int, initial_value: typing.Any = 0) -> typing.List[typing.Any]:
    if isinstance(initial_value, list) and len(initial_value) == size:
        return initial_value
    return [initial_value] * size

# C++: tile_distribution_encoding.hpp
class TileDistributionEncodingPedantic:
    """
    Python equivalent of ck_tile::tile_distribution_encoding.
    """
    def __init__(self, encoding_dict: typing.Dict[str, typing.Any], variables: typing.Dict[str, int] = None):
        """
        Initialize from a dictionary-based encoding (from parser) and variables.
        
        Args:
            encoding_dict: Dictionary containing the parsed tile_distribution_encoding
                           (e.g., from a JSON or visualizer input).
            variables: Dictionary mapping variable names (like "Nr_y") to their values.
        """
        self.variables = variables or {}

        # C++: tile_distribution_encoding.hpp:L21-L26 (template arguments)
        # These are resolved from the input dictionary and variables.
        self.RsLengths: typing.List[int] = self._resolve_values(encoding_dict.get('RsLengths', []))
        self.PsLengths: typing.List[int] = self._resolve_values(encoding_dict.get('PsLengths', []))
        self.HsLengthss: typing.List[typing.List[int]] = [
            self._resolve_values(h_list) 
            for h_list in encoding_dict.get('HsLengthss', [])
        ]
        self.Ps2RHssMajor: typing.List[typing.List[int]] = [
            self._resolve_values(p_list)
            for p_list in encoding_dict.get('Ps2RHssMajor', [])
        ]
        self.Ps2RHssMinor: typing.List[typing.List[int]] = [
            self._resolve_values(p_list)
            for p_list in encoding_dict.get('Ps2RHssMinor', [])
        ]
        self.Ys2RHsMajor: typing.List[int] = self._resolve_values(encoding_dict.get('Ys2RHsMajor', []))
        self.Ys2RHsMinor: typing.List[int] = self._resolve_values(encoding_dict.get('Ys2RHsMinor', []))

        # C++: tile_distribution_encoding.hpp:L28-L29 (static_asserts)
        assert len(self.Ps2RHssMajor) == len(self.Ps2RHssMinor), "Ps2RHssMajor and Ps2RHssMinor must have the same size"
        assert len(self.Ys2RHsMajor) == len(self.Ys2RHsMinor), "Ys2RHsMajor and Ys2RHsMinor must have the same size"

        # C++: tile_distribution_encoding.hpp:L31-L34 (NDimX, NDimP, NDimY, NDimR)
        self.NDimX: int = len(self.HsLengthss)
        self.NDimP: int = len(self.Ps2RHssMajor)
        self.NDimY: int = len(self.Ys2RHsMajor)
        self.NDimR: int = len(self.RsLengths)

        # Correct NDimP calculation based on C++ which uses Ps2RHssMajor::size()
        # NDimP was: len(self.PsLengths)
        if not self.PsLengths and self.NDimP > 0:
            print(f"INFO: PsLengths not explicitly provided. Inferring NDimP={self.NDimP} from Ps2RHssMajor. Setting PsLengths to placeholders.")
            self.PsLengths = [1] * self.NDimP
        elif len(self.PsLengths) != self.NDimP and self.NDimP > 0 : # If PsLengths was provided but mismatched NDimP from Ps2RHssMajor
             print(f"WARNING: Mismatch between PsLengths size ({len(self.PsLengths)}) and NDimP from Ps2RHssMajor ({self.NDimP}). Using NDimP from Ps2RHssMajor.")
             # Optionally, resize or warn further. For now, NDimP from Ps2RHssMajor takes precedence for dimension count.
             # If PsLengths is shorter, it might lead to issues later if accessed directly up to NDimP.
             # If it's longer, it's also strange. For now, just a warning.
             # A more robust approach might be to ensure PsLengths is also resized to NDimP with placeholders if mismatch.

        # C++: tile_distribution_encoding.hpp:L37-L42 (constexpr static members holding the sequences)
        # In Python, these are just the instance attributes themselves.
        # self.rs_lengths_ = self.RsLengths (already assigned)
        # self.hs_lengthss_ = self.HsLengthss
        # ...and so on.

        # Placeholder for the 'detail' struct computations
        self.detail = self._compute_detail()
        
        # Add source_code attribute, similar to tiler.py
        self.source_code = None # Will be set via a method

    def _resolve_values(self, values: typing.List[typing.Any]) -> typing.List[int]:
        """
        Resolve variable names (strings) in a list to their integer values.
        If a value is already an int, it's used directly.
        Uses self.variables for lookup. Defaults to 4 if var not found.
        """
        # This is similar to _resolve_values in the original tiler.py
        resolved = []
        for val in values:
            if isinstance(val, str):
                resolved.append(self.variables.get(val, 4)) # Default to 4 if var not found
            elif isinstance(val, (int, float)):
                resolved.append(int(val))
            else:
                # This case should ideally not happen with valid input
                print(f"Warning: Unexpected value type '{type(val)}' in _resolve_values, using 4.")
                resolved.append(4) 
        return resolved

    def _compute_detail(self) -> typing.Dict[str, typing.Any]:
        """
        Compute the values that would be in the 'detail' nested struct
        in the C++ version.
        C++: tile_distribution_encoding.hpp:L46 (struct detail)
        """
        detail_data = {}

        # C++: tile_distribution_encoding.hpp:L49 (ndim_rh_major_)
        detail_data['ndim_rh_major_'] = self.NDimX + 1 
        
        # C++: tile_distribution_encoding.hpp:L50 (ndim_span_major_)
        detail_data['ndim_span_major_'] = self.NDimX

        # C++: tile_distribution_encoding.hpp:L53 (ndims_rhs_minor_)
        ndims_rhs_minor_list = []
        ndims_rhs_minor_list.append(len(self.RsLengths))
        for h_idx in range(self.NDimX):
            ndims_rhs_minor_list.append(len(self.HsLengthss[h_idx]))
        detail_data['ndims_rhs_minor_'] = ndims_rhs_minor_list
        
        # C++: tile_distribution_encoding.hpp:L65 (max_ndim_rh_minor_)
        if not detail_data['ndims_rhs_minor_']:
             detail_data['max_ndim_rh_minor_'] = 0
        else:
            detail_data['max_ndim_rh_minor_'] = max(detail_data['ndims_rhs_minor_']) if detail_data['ndims_rhs_minor_'] else 0

        # C++: tile_distribution_encoding.hpp:L69 (rhs_lengthss_)
        rhs_lengthss_list = [self.RsLengths] + self.HsLengthss
        detail_data['rhs_lengthss_'] = rhs_lengthss_list
        
        # ys_lengths_ (C++: tile_distribution_encoding.hpp:L72)
        ys_lengths_tmp = Array(self.NDimY, -1)
        for i in range(self.NDimY):
            rh_major = self.Ys2RHsMajor[i]
            rh_minor = self.Ys2RHsMinor[i]
            if (0 <= rh_major < len(detail_data['rhs_lengthss_']) and
                0 <= rh_minor < len(detail_data['rhs_lengthss_'][rh_major])):
                ys_lengths_tmp[i] = detail_data['rhs_lengthss_'][rh_major][rh_minor]
        detail_data['ys_lengths_'] = ys_lengths_tmp
        
        # rhs_major_minor_to_ys_ (C++: tile_distribution_encoding.hpp:L84)
        max_rh_minor = detail_data['max_ndim_rh_minor_']
        num_rh_major = self.NDimX + 1
        rhs_major_minor_to_ys_tmp = [[-1] * max_rh_minor for _ in range(num_rh_major)]
        for i in range(self.NDimY):
            rh_major = self.Ys2RHsMajor[i]
            rh_minor = self.Ys2RHsMinor[i]
            if 0 <= rh_major < num_rh_major and 0 <= rh_minor < max_rh_minor:
                 rhs_major_minor_to_ys_tmp[rh_major][rh_minor] = i
        detail_data['rhs_major_minor_to_ys_'] = rhs_major_minor_to_ys_tmp

        # C++: tile_distribution_encoding.hpp:L101 (ndims_span_minor_)
        ndims_span_minor_list = [0] * self.NDimX
        for i in range(self.NDimY):
            span_major = self.Ys2RHsMajor[i] - 1
            if 0 <= span_major < self.NDimX:
                ndims_span_minor_list[span_major] += 1
        detail_data['ndims_span_minor_'] = ndims_span_minor_list

        # C++: tile_distribution_encoding.hpp:L113 (max_ndim_span_minor_)
        if not detail_data['ndims_span_minor_']:
            detail_data['max_ndim_span_minor_'] = 0
        else:
            detail_data['max_ndim_span_minor_'] = max(detail_data['ndims_span_minor_']) if detail_data['ndims_span_minor_'] else 0
            
        # C++: tile_distribution_encoding.hpp:L117 (rhs_major_minor_to_span_minor_)
        rhs_major_minor_to_span_minor_tmp = [[-1] * detail_data['max_ndim_rh_minor_'] for _ in range(detail_data['ndim_rh_major_'])]
        for rh_major_idx in range(detail_data['ndim_rh_major_']):
            current_rh_max_minor_len = detail_data['ndims_rhs_minor_'][rh_major_idx]
            count_ndim_span_minor = 0
            for rh_minor_idx in range(current_rh_max_minor_len):
                idim_y = detail_data['rhs_major_minor_to_ys_'][rh_major_idx][rh_minor_idx]
                if idim_y >= 0:
                    rhs_major_minor_to_span_minor_tmp[rh_major_idx][rh_minor_idx] = count_ndim_span_minor
                    count_ndim_span_minor += 1
        detail_data['rhs_major_minor_to_span_minor_'] = rhs_major_minor_to_span_minor_tmp

        # C++: tile_distribution_encoding.hpp:L137 (ys_to_span_major_)
        ys_to_span_major_list = [-1] * self.NDimY
        for i in range(self.NDimY):
            ys_to_span_major_list[i] = self.Ys2RHsMajor[i] - 1
        detail_data['ys_to_span_major_'] = ys_to_span_major_list
        
        # C++: tile_distribution_encoding.hpp:L141 (ys_to_span_minor_)
        ys_to_span_minor_list = [-1] * self.NDimY
        for i in range(self.NDimY):
            rh_major = self.Ys2RHsMajor[i]
            rh_minor = self.Ys2RHsMinor[i]
            if (0 <= rh_major < len(detail_data['rhs_major_minor_to_span_minor_']) and
                0 <= rh_minor < len(detail_data['rhs_major_minor_to_span_minor_'][rh_major])): # Python handles this multi-line due to parenthesis
                ys_to_span_minor_list[i] = detail_data['rhs_major_minor_to_span_minor_'][rh_major][rh_minor]
        detail_data['ys_to_span_minor_'] = ys_to_span_minor_list

        # C++: tile_distribution_encoding.hpp:L147 (distributed_spans_lengthss_)
        max_span_minor_val = detail_data['max_ndim_span_minor_']
        num_span_major_val = detail_data['ndim_span_major_'] 
        distributed_spans_lengthss_tmp = [[-1] * max_span_minor_val for _ in range(num_span_major_val)]
        for i in range(self.NDimY):
            rh_major = self.Ys2RHsMajor[i]
            rh_minor = self.Ys2RHsMinor[i]
            h_sequence_idx = rh_major - 1
            if 0 <= h_sequence_idx < self.NDimX:
                if 0 <= rh_minor < len(self.HsLengthss[h_sequence_idx]):
                    h_length = self.HsLengthss[h_sequence_idx][rh_minor]
                    span_major = h_sequence_idx
                    span_minor = detail_data['rhs_major_minor_to_span_minor_'][rh_major][rh_minor] # Error check: rh_major can be 0 for R, span_minor from R map?
                    if span_minor != -1: # Ensure span_minor is valid
                        if 0 <= span_major < num_span_major_val and 0 <= span_minor < max_span_minor_val:
                            distributed_spans_lengthss_tmp[span_major][span_minor] = h_length
        detail_data['distributed_spans_lengthss_'] = distributed_spans_lengthss_tmp
        
        # C++: tile_distribution_encoding.hpp:L166 (ndims_distributed_spans_minor_)
        detail_data['ndims_distributed_spans_minor_'] = detail_data['ndims_span_minor_']

        # C++: tile_distribution_encoding.hpp:L178 (does_p_own_r_)
        does_p_own_r_tmp = [[False] * self.NDimR for _ in range(self.NDimP)]
        if self.NDimR > 0:
            for idim_p in range(self.NDimP):
                current_p_major_map = self.Ps2RHssMajor[idim_p]
                current_p_minor_map = self.Ps2RHssMinor[idim_p]
                ndim_low = len(current_p_major_map)
                for idim_low_idx in range(ndim_low):
                    rh_major_for_p = current_p_major_map[idim_low_idx]
                    rh_minor_for_p = current_p_minor_map[idim_low_idx]
                    if rh_major_for_p == 0:
                        if 0 <= rh_minor_for_p < self.NDimR:
                             does_p_own_r_tmp[idim_p][rh_minor_for_p] = True
        detail_data['does_p_own_r_'] = does_p_own_r_tmp

        # C++: tile_distribution_encoding.hpp:L200 (ps_over_rs_derivative_)
        ps_over_rs_derivative_tmp = [[0] * self.NDimR for _ in range(self.NDimP)]
        if self.NDimR > 0:
            for idim_p in range(self.NDimP):
                current_p_major_map = self.Ps2RHssMajor[idim_p]
                current_p_minor_map = self.Ps2RHssMinor[idim_p]
                ndim_low = len(current_p_major_map)
                p_over_rh_derivative = 1
                for idim_low_idx in reversed(range(ndim_low)):
                    rh_major_for_p = current_p_major_map[idim_low_idx]
                    rh_minor_for_p = current_p_minor_map[idim_low_idx]
                    rh_length = -1
                    if (0 <= rh_major_for_p < len(detail_data['rhs_lengthss_']) and
                        0 <= rh_minor_for_p < len(detail_data['rhs_lengthss_'][rh_major_for_p])):
                        rh_length = detail_data['rhs_lengthss_'][rh_major_for_p][rh_minor_for_p]
                    else:
                        print(f"Warning: Invalid (rh_major, rh_minor) access in ps_over_rs_derivative_ for P{idim_p}, low_dim{idim_low_idx}")

                    if rh_major_for_p == 0: # If P maps to an R-dimension
                         if 0 <= rh_minor_for_p < self.NDimR: # Check rh_minor_for_p is a valid R index
                            ps_over_rs_derivative_tmp[idim_p][rh_minor_for_p] = p_over_rh_derivative
                    
                    if rh_length > 0:
                        p_over_rh_derivative *= rh_length
                    elif rh_length == 0:
                         print(f"Warning: rh_length is 0 in ps_over_rs_derivative_ calc for P{idim_p}, low_dim{idim_low_idx}. Derivative might be ill-defined.")
                         p_over_rh_derivative = 0 
        detail_data['ps_over_rs_derivative_'] = ps_over_rs_derivative_tmp
        
        # C++: tile_distribution_encoding.hpp:L225 (get_h_dim_lengths_prefix_sum)
        uniformed_h_dim_lengths = [len(h_seq) for h_seq in self.HsLengthss]
        h_dim_lengths_prefix_sum_list = [0] * (len(uniformed_h_dim_lengths) + 1)
        current_sum = 0
        for i, length in enumerate(uniformed_h_dim_lengths):
            h_dim_lengths_prefix_sum_list[i+1] = current_sum + length
            current_sum += length
        detail_data['h_dim_lengths_prefix_sum_'] = h_dim_lengths_prefix_sum_list

        # C++: tile_distribution_encoding.hpp:L243 (get_uniformed_idx_y_to_h)
        global_rh_component_offsets = [0] * (1 + self.NDimX) 
        current_offset = 0
        # Offset for R dimensions (if any)
        # Ys2RHsMajor = 0 means R. RsLengths is self.DstrEncode.RsLengths.
        # Number of components in R is self.NDimR (len(self.RsLengths))
        # So H0 starts after NDimR components.
        global_rh_component_offsets[0] = 0 # Base for R components
        current_offset = self.NDimR 
        for h_idx in range(self.NDimX): # For H0, H1, ...
            global_rh_component_offsets[h_idx + 1] = current_offset
            current_offset += len(self.HsLengthss[h_idx])
        
        uniformed_idx_y_to_h_list = [-1] * self.NDimY
        for i in range(self.NDimY):
            ys_major_val = self.Ys2RHsMajor[i] # 0 for R, 1 for H0, 2 for H1...
            ys_minor_val = self.Ys2RHsMinor[i] # index within R or H_k
            base_offset_for_major = global_rh_component_offsets[ys_major_val]
            uniformed_idx_y_to_h_list[i] = base_offset_for_major + ys_minor_val
        detail_data['uniformed_idx_y_to_h_'] = uniformed_idx_y_to_h_list

        # C++: tile_distribution_encoding.hpp:L253 (get_sorted_y_info)
        idx_seq = detail_data['uniformed_idx_y_to_h_']
        sorted_dims_val = []
        sorted_to_unsorted_map_val = [] # Stores original indices of the first occurrence of each unique sorted value
        histogram_counts = []
        sorted_y_prefix_sum_val = [0]

        if idx_seq:
            indexed_values = sorted([(val, orig_idx) for orig_idx, val in enumerate(idx_seq)])
            if indexed_values:
                last_val = None
                for val, orig_idx in indexed_values:
                    if val != last_val:
                        sorted_dims_val.append(val)
                        sorted_to_unsorted_map_val.append(orig_idx)
                        last_val = val
                
                # Calculate histogram for sorted_dims_val based on original idx_seq
                histogram_counts = [0] * len(sorted_dims_val)
                value_to_sorted_idx_map = {val: i for i, val in enumerate(sorted_dims_val)}
                for val_from_original_idx_seq in idx_seq:
                    if val_from_original_idx_seq in value_to_sorted_idx_map: # Should always be true if idx_seq is not empty
                        histogram_counts[value_to_sorted_idx_map[val_from_original_idx_seq]] +=1
                
                # Prefix sum of histogram_counts
                current_hist_sum = 0
                for count in histogram_counts: # Build up sorted_y_prefix_sum_val
                    current_hist_sum += count
                    sorted_y_prefix_sum_val.append(current_hist_sum)
        
        detail_data['sorted_y_info_'] = {
            'sorted_dims': sorted_dims_val,
            'sorted2unsorted_map': sorted_to_unsorted_map_val, 
            'histogram': histogram_counts,
            'prefix_sum': sorted_y_prefix_sum_val 
        }

        # C++: tile_distribution_encoding_detail.hpp:L343 (is_ys_from_r_span_)
        # A Y-dimension is from an R-span if its Ys2RHsMajor maps to the R-component major index (0).
        is_ys_from_r_span_list = [False] * self.NDimY
        for i in range(self.NDimY):
            if self.Ys2RHsMajor[i] == 0: # 0 is the major index for R-dimensions in rhs_lengthss_
                is_ys_from_r_span_list[i] = True
        detail_data['is_ys_from_r_span_'] = is_ys_from_r_span_list

        return detail_data

    def print_encoding(self):
        """
        Prints the encoding information, similar to the C++ print method.
        C++: tile_distribution_encoding.hpp:L309
        C++: detail struct print: tile_distribution_encoding.hpp:L268
        """
        print("tile_distribution_encoding{")
        print(f"  NDimX: {self.NDimX}, NDimP: {self.NDimP}, NDimY: {self.NDimY}, NDimR: {self.NDimR},")
        print(f"  RsLengths: {self.RsLengths},")
        print(f"  HsLengthss: {self.HsLengthss},")
        print(f"  Ps2RHssMajor: {self.Ps2RHssMajor},")
        print(f"  Ps2RHssMinor: {self.Ps2RHssMinor},")
        print(f"  Ys2RHsMajor: {self.Ys2RHsMajor},")
        print(f"  Ys2RHsMinor: {self.Ys2RHsMinor},")
        print("  detail: {")
        detail_keys_ordered = [
            'ndim_rh_major_', 'ndim_span_major_', 'ndims_rhs_minor_', 'max_ndim_rh_minor_',
            'rhs_lengthss_', 'ys_lengths_', 'rhs_major_minor_to_ys_',
            'ndims_span_minor_', 'max_ndim_span_minor_', 'rhs_major_minor_to_span_minor_',
            'ys_to_span_major_', 'ys_to_span_minor_', 'distributed_spans_lengthss_',
            'ndims_distributed_spans_minor_', 'does_p_own_r_', 'ps_over_rs_derivative_',
            'h_dim_lengths_prefix_sum_', 'uniformed_idx_y_to_h_', 'sorted_y_info_',
            'is_ys_from_r_span_'
        ]
        for key in detail_keys_ordered:
            if key in self.detail:
                value = self.detail[key]
                if isinstance(value, list) and value and isinstance(value[0], list) and key != 'rhs_major_minor_to_ys_' and key != 'distributed_spans_lengthss_' and key != 'does_p_own_r_' and key != 'ps_over_rs_derivative_':
                     # Generic list of lists (like HsLengthss if it were in detail)
                    print(f"    {key}: [")
                    for sublist in value:
                        print(f"      {sublist}")
                    print("    ]")
                elif key in ['rhs_major_minor_to_ys_', 'distributed_spans_lengthss_', 'does_p_own_r_', 'ps_over_rs_derivative_']:
                    # Specific formatting for these tables
                    print(f"    {key}:")
                    for row_idx, row_val in enumerate(value):
                        print(f"      Row {row_idx}: {row_val}")
                elif key == 'sorted_y_info_' and isinstance(value, dict):
                    print(f"    {key}: {{")
                    for sub_key, sub_val in value.items():
                        print(f"      {sub_key}: {sub_val}")
                    print("    }")
                else:
                    print(f"    {key}: {value}")
            else:
                print(f"    {key}: <Not Implemented Yet>")
        print("  }") # End of detail
        print("}") # End of tile_distribution_encoding

    def set_source_code(self, code: str):
        """
        Set the source code for this tile distribution to show in the visualization.
        """
        self.source_code = code

    # More methods will be added here, especially for the 'detail' struct calculations.


class TileDistributionPedantic:
    """
    Python equivalent of ck_tile::tile_distribution.
    This class will use TileDistributionEncodingPedantic.
    """
    def __init__(self, encoding_dict: typing.Dict[str, typing.Any], variables: typing.Dict[str, int] = None):
        self.encoding_input_dict = encoding_dict # Keep original for hints
        self.variables_input = variables or {}
        
        self.DstrEncode = TileDistributionEncodingPedantic(encoding_dict, variables)
        
        self.NDimX = self.DstrEncode.NDimX
        self.NDimPs = self.DstrEncode.NDimP
        self.NDimYs = self.DstrEncode.NDimY
        self.NDimRs = self.DstrEncode.NDimR

        # Call the adaptor creation method (assuming it will be named _create_adaptor_encodings_json_style)
        ps_ys_to_xs_adaptor_json, ys_to_d_adaptor_json, d_length, rh_map = \
            self._create_adaptor_encodings_json_style() # This will replace the placeholder call

        self.PsYs2XsAdaptor: typing.Dict = ps_ys_to_xs_adaptor_json
        self.Ys2DDescriptor: typing.Dict = ys_to_d_adaptor_json
        
        # Store d_length directly in Ys2DDescriptor, which is expected to be the JSON dict itself
        self.Ys2DDescriptor['d_length'] = d_length

        # Store the rh_map in DstrDetail for use by other methods like calculate_rs_index_from_ps_index
        self.DstrDetail: typing.Dict[str, typing.Any] = {'rh_map': rh_map}
        
        self.tile_shape: typing.Sequence[int] = self._calculate_tile_shape()
        self.thread_mapping: typing.Dict[str, int] = self._calculate_thread_mapping()
        self.source_code: typing.Optional[str] = None
        self.tile_name: typing.Optional[str] = None

    # MAKE SURE THE OLD _create_adaptor_encodings_json_style_placeholder IS REMOVED

    # START OF THE NEW _create_adaptor_encodings_json_style METHOD
    def _create_adaptor_encodings_json_style(self) -> typing.Tuple[typing.Dict, typing.Dict, int, typing.Dict]:
        """
        Pedantically creates JSON-style adaptor encodings based on self.DstrEncode.
        Mimics the logic of C++ `detail::make_adaptor_encoding_for_tile_distribution`
        from `tile_distribution.hpp`.

        Returns:
            ps_ys_to_xs_adaptor_json: JSON structure for PsYs to Xs adaptor.
            ys_to_d_adaptor_json: JSON structure for Ys to D adaptor.
            d_length_calculated: Calculated length of the D-space.
            rh_map: Map from (rh_major, rh_minor) to hidden_id and length.
        """
        RsLengths_ = self.DstrEncode.RsLengths
        HsLengthss_ = self.DstrEncode.HsLengthss
        Ps2RHssMajor_ = self.DstrEncode.Ps2RHssMajor
        Ps2RHssMinor_ = self.DstrEncode.Ps2RHssMinor
        Ys2RHsMajor_ = self.DstrEncode.Ys2RHsMajor
        Ys2RHsMinor_ = self.DstrEncode.Ys2RHsMinor

        NDimXs = self.DstrEncode.NDimX
        NDimPs = self.DstrEncode.NDimP
        NDimYs = self.DstrEncode.NDimY

        psys_to_xs_transforms = []
        rh_map = {}  # Stores {(rh_major, rh_minor): {'id': hidden_id, 'length': length}}
                     # rh_major: 0 for R, 1 for H0, 2 for H1, ... (i.e., HsLengthss index + 1)
        
        next_available_hidden_id = 0  # Counter for all new hidden dimension IDs
        effective_psys_top_dim_ids = [] # Stores actual hidden IDs for PsYs adaptor's top interface

        # 1. Replicate Transform (for RsLengths)
        ndim_r_minor = len(RsLengths_)
        if ndim_r_minor > 0:
            r_minor_lengths = list(RsLengths_)
            dst_dim_ids_for_replicate = []
            for i in range(ndim_r_minor):
                hid = next_available_hidden_id
                dst_dim_ids_for_replicate.append(hid)
                rh_map[(0, i)] = {'id': hid, 'length': r_minor_lengths[i]}
                next_available_hidden_id += 1
            psys_to_xs_transforms.append({
                "Name": "Replicate", "MetaData": r_minor_lengths,
                "SrcDimIds": [], "DstDimIds": dst_dim_ids_for_replicate
            })

        # 2. Unmerge Transforms (for HsLengthss)
        for idim_x in range(NDimXs):
            h_minor_lengths = list(HsLengthss_[idim_x])
            ndim_h_minor = len(h_minor_lengths)
            if ndim_h_minor > 0:
                dst_dim_ids_for_unmerge = []
                for i in range(ndim_h_minor):
                    hid = next_available_hidden_id
                    dst_dim_ids_for_unmerge.append(hid)
                    rh_map[(idim_x + 1, i)] = {'id': hid, 'length': h_minor_lengths[i]} # rh_major for H0 is 1, ...
                    next_available_hidden_id += 1
                psys_to_xs_transforms.append({
                    "Name": "Unmerge", "MetaData": h_minor_lengths,
                    "SrcDimIds": [idim_x], "DstDimIds": dst_dim_ids_for_unmerge # Src is X0, X1...
                })

        # 3. Merge Transforms (for Ps dimensions)
        p_merged_hidden_ids = [] # Stores the final hidden ID for each P-dim after merge
        for iDimP in range(NDimPs):
            p_final_hidden_id = next_available_hidden_id # This P will merge into this new ID
            next_available_hidden_id += 1
            p_merged_hidden_ids.append(p_final_hidden_id)

            p2RHsMajor_current_P = Ps2RHssMajor_[iDimP]
            p2RHsMinor_current_P = Ps2RHssMinor_[iDimP]
            ndim_low_for_p_merge = len(p2RHsMajor_current_P)
            src_dims_for_p_merge = []
            src_lengths_for_p_merge_metadata = []

            for k in range(ndim_low_for_p_merge):
                rh_major_val = p2RHsMajor_current_P[k]
                rh_minor_val = p2RHsMinor_current_P[k]
                entry = rh_map.get((rh_major_val, rh_minor_val))
                if entry:
                    src_dims_for_p_merge.append(entry['id'])
                    src_lengths_for_p_merge_metadata.append(entry['length'])
                else:
                    raise ValueError(f"P-Merge Error: RH component ({rh_major_val},{rh_minor_val}) for P{iDimP} not in rh_map.")
            psys_to_xs_transforms.append({
                "Name": "Merge", "MetaData": src_lengths_for_p_merge_metadata,
                "SrcDimIds": src_dims_for_p_merge, "DstDimIds": [p_final_hidden_id]
            })
        
        effective_psys_top_dim_ids.extend(p_merged_hidden_ids) # Add P-dim effective IDs
        
        ys_hidden_ids_for_top_view = [] # Collect hidden IDs for YS-dims
        for iDimY in range(NDimYs):
            rh_major_val = Ys2RHsMajor_[iDimY]
            rh_minor_val = Ys2RHsMinor_[iDimY]
            entry = rh_map.get((rh_major_val, rh_minor_val))
            if entry:
                ys_hidden_ids_for_top_view.append(entry['id'])
            else:
                raise ValueError(f"PsYsAdaptor TopView YS Error: RH component ({rh_major_val},{rh_minor_val}) for Y{iDimY} not in rh_map.")
        effective_psys_top_dim_ids.extend(ys_hidden_ids_for_top_view) # Add YS-dim effective IDs

        # 4. Construct PsYs2XsAdaptor JSON
        # BottomView: Original X dimensions (IDs 0 to NDimX-1)
        psys_bottom_id_to_name = {i: f"X{i}" for i in range(NDimXs)}
        psys_bottom_name_lengths = {}
        for i in range(NDimXs):
            length = 1
            # Ensure product of empty list is 1, or if HsLengthss_[i] is empty, length of X dim is 1.
            if HsLengthss_[i]: 
                length = int(np.prod(HsLengthss_[i]))
            else: # If HsLengthss_[i] is empty list e.g. X_i has 0 components
                length = 1 # An X dimension with no H components could be seen as length 1.
                           # C++ container_reduce with multiplies and initial 1 would yield 1 for empty sequence.
            psys_bottom_name_lengths[f"X{i}"] = length
            
        # TopView: Uses the effective_psys_top_dim_ids. Map these hidden IDs to logical names P0, Y0...
        psys_top_id_to_name = {} # Maps hidden_id -> "P0", "Y0", etc.
        psys_top_name_lengths = {} # Maps "P0", "Y0" -> length
        for i, hid in enumerate(effective_psys_top_dim_ids):
            if i < NDimPs: # This is a P-dimension
                name = f"P{i}"
                psys_top_id_to_name[hid] = name
                psys_top_name_lengths[name] = self.DstrEncode.PsLengths[i] if i < len(self.DstrEncode.PsLengths) else 1
            else: # This is a YS-dimension
                name = f"Y{i - NDimPs}"
                psys_top_id_to_name[hid] = name
                ys_idx_in_list = i - NDimPs # Original index of this YS dimension
                orig_rh_major = Ys2RHsMajor_[ys_idx_in_list]
                orig_rh_minor = Ys2RHsMinor_[ys_idx_in_list]
                entry = rh_map.get((orig_rh_major, orig_rh_minor)) # Get its length from rh_map
                psys_top_name_lengths[name] = entry['length'] if entry else 1

        ps_ys_to_xs_adaptor_json = {
            "BottomView": {"BottomDimensionIdToName": psys_bottom_id_to_name, "BottomDimensionNameLengths": psys_bottom_name_lengths},
            "TopView": {"TopDimensionIdToName": psys_top_id_to_name, "TopDimensionNameLengths": psys_top_name_lengths, "_effective_display_order_ids_": effective_psys_top_dim_ids},
            "Transformations": psys_to_xs_transforms
        }

        # 5. Construct Ys2DDescriptor JSON
        ys_to_d_transforms = []
        y_lengths_for_d_unmerge_metadata = []
        d_length_calculated = 1

        if NDimYs > 0:
            for i in range(NDimYs): # Iterate through logical Y0, Y1...
                rh_major_val = Ys2RHsMajor_[i]
                rh_minor_val = Ys2RHsMinor_[i]
                entry = rh_map.get((rh_major_val, rh_minor_val))
                y_len_current_dim = 1 # Default
                if entry: 
                    y_len_current_dim = entry['length']
                else: 
                    raise ValueError(f"Ys2D Error: YS component ({rh_major_val},{rh_minor_val}) for Y{i} not in rh_map.")
                y_lengths_for_d_unmerge_metadata.append(y_len_current_dim)
                if y_len_current_dim == 0: # If any Y length is 0, d_length becomes 0
                    d_length_calculated = 0 
                if d_length_calculated != 0 : # Avoid multiplying by zero if already zero
                    d_length_calculated *= y_len_current_dim
            
            ys_to_d_transforms.append({
                "Name": "Unmerge", "MetaData": y_lengths_for_d_unmerge_metadata,
                "SrcDimIds": [0], "DstDimIds": list(range(NDimYs)) # Dst IDs are logical 0..NDimYs-1 for Y0,Y1..
            })
        else: # NDimYs == 0
            d_length_calculated = 1
            ys_to_d_transforms.append({"Name": "Unmerge", "MetaData": [], "SrcDimIds": [0], "DstDimIds": []})

        ys_to_d_bottom_id_to_name = {"0": "D_flat"}
        ys_to_d_bottom_name_lengths = {"D_flat": d_length_calculated}
        ys_to_d_top_id_to_name = {i: f"Y{i}" for i in range(NDimYs)} 
        ys_to_d_top_name_lengths = {f"Y{i}": y_lengths_for_d_unmerge_metadata[i] for i in range(NDimYs)}

        ys_to_d_adaptor_json = {
            "BottomView": {"BottomDimensionIdToName": ys_to_d_bottom_id_to_name, "BottomDimensionNameLengths": ys_to_d_bottom_name_lengths},
            "TopView": {"TopDimensionIdToName": ys_to_d_top_id_to_name, "TopDimensionNameLengths": ys_to_d_top_name_lengths},
            "Transformations": ys_to_d_transforms
        }
        
        return ps_ys_to_xs_adaptor_json, ys_to_d_adaptor_json, d_length_calculated, rh_map
    # END OF THE NEW _create_adaptor_encodings_json_style METHOD

    def _calculate_tile_shape(self) -> typing.Sequence[int]:
        # Mimics C++: tile_distribution::get_lengths() but sums them up for a simple shape
        # For a more representative shape, might need to sum specific H dimensions (e.g., Hs[0] for one part, Hs[1] for another)
        # Or, more closely, use self.PsYs2XsAdaptor.BottomDimensionNameLengths
        # This is a very basic tile shape representation.
        
        # Using RsLengths for one dimension and sum of first H sequence for another
        # This is a heuristic. A more detailed shape might be needed for specific visualizations.
        r_sum = sum(self.DstrEncode.RsLengths) if self.DstrEncode.RsLengths else 1
        
        h0_sum = 0
        if self.DstrEncode.HsLengthss and self.DstrEncode.HsLengthss[0]:
            h0_sum = sum(self.DstrEncode.HsLengthss[0])
        else:
            h0_sum = 1 # Default if no H dims or H0 is empty
            
        return [max(1, r_sum), max(1, h0_sum)] # Ensure non-zero dimensions

    def _calculate_thread_mapping(self) -> typing.Dict[str, int]:
        """
        Calculate a simple placeholder mapping of threads to tile elements,
        similar to tiler.py for basic visualizer compatibility.
        """
        mapping = {}
        # Use self.tile_shape which is [sum(RsLengths), sum(HsLengthss[0])]
        rows, cols = self.tile_shape # In __init__, tile_shape is already computed.
        
        # Ensure rows and cols are at least 1 for iteration if tile_shape could be [0,x] or [x,0]
        rows_iter = max(1, rows)
        cols_iter = max(1, cols)

        thread_idx = 0
        for i in range(rows_iter):
            for j in range(cols_iter):
                mapping[f"{i},{j}"] = thread_idx
                thread_idx += 1
        return mapping

    def _apply_tensor_adaptor_transformations(self, input_coords: typing.Sequence[int], adaptor_encoding: dict) -> typing.Sequence[int]:
        """
        Applies a series of tensor transformations based on the provided adaptor encoding.
        """
        current_coords = list(input_coords)
        
        # top_tensor_view_dims is a list of [id, lower_inclusive, upper_exclusive]
        # For this forward transformation, current_coords correspond to adaptor_encoding['top_tensor_view']['dims']
        
        # Map initial input_coords by their dimension ID from top_tensor_view
        known_dim_values: typing.Dict[typing.Any, int] = {}
        top_dims_desc = adaptor_encoding['top_tensor_view']['dims']

        if not top_dims_desc and not current_coords: # Handle scalar case (0-dim input)
             pass # No initial known_dim_values needed if transforms handle creation of bottom dims
        elif len(current_coords) != len(top_dims_desc):
            raise ValueError(
                f"Input coordinates length ({len(current_coords)}) does not match "
                f"top_tensor_view dimension count ({len(top_dims_desc)}). Coords: {current_coords}, TopView: {top_dims_desc}"
            )
        else:
            for i, dim_desc in enumerate(top_dims_desc):
                known_dim_values[dim_desc[0]] = current_coords[i]

        for transform_def in adaptor_encoding['transforms']:
            transform_type = transform_def['type']
            params = transform_def['parameters']
            top_ids = transform_def['top_ids_']          # Dimension IDs input to this transform step
            bottom_ids = transform_def['bottom_ids_']    # Dimension IDs output by this transform step

            if transform_type == "PassThrough":
                # val_at_bottom_ids[0] = val_at_top_ids[0]
                if not top_ids: # e.g. scalar to scalar from empty top_ids
                    if not bottom_ids: raise ValueError("PassThrough: top_ids and bottom_ids cannot both be empty")
                    # This case implies creating a bottom dim from nothing specific in top_ids
                    # Assume a default value or that it's handled by adaptor structure (e.g. fixed size 1 dim)
                    # This requires careful definition in adaptor for 0-dim inputs
                    if bottom_ids[0] not in known_dim_values: # If bottom_id not already set by a previous branching transform
                         known_dim_values[bottom_ids[0]] = 0 # Placeholder for scalar to scalar, refine if needed
                elif top_ids[0] not in known_dim_values:
                    raise ValueError(f"PassThrough: Dim '{top_ids[0]}' not found in known_dim_values.")
                else:
                    val_from_top = known_dim_values[top_ids[0]]
                    known_dim_values[bottom_ids[0]] = val_from_top
            
            elif transform_type == "Merge":
                # merged_val (at bottom_ids[0]) = val_major (at top_ids[0]) * len_minor (at top_ids[1])
                major_val_dim_id = top_ids[0]
                minor_val_dim_id = top_ids[1]
                
                if major_val_dim_id not in known_dim_values or minor_val_dim_id not in known_dim_values:
                    raise ValueError(f"Merge: Dim '{major_val_dim_id}' or '{minor_val_dim_id}' not found.")
                
                val_major = known_dim_values[major_val_dim_id]
                val_minor = known_dim_values[minor_val_dim_id]
                len_minor = params['lengths_of_inputs'][1] # Length of the second (minor) input dimension
                
                known_dim_values[bottom_ids[0]] = val_major * len_minor + val_minor
            
            elif transform_type == "Embed":
                # val_padded (at bottom_ids[0]) = val_original (at top_ids[0]) + pre_padding
                original_val_dim_id = top_ids[0]
                if original_val_dim_id not in known_dim_values:
                    raise ValueError(f"Embed: Dim '{original_val_dim_id}' not found.")
                
                val_original = known_dim_values[original_val_dim_id]
                pre_padding = params['pre_padding']
                known_dim_values[bottom_ids[0]] = val_original + pre_padding
                
            elif transform_type == "Split":
                # val_major (bottom_ids[0]), val_minor (bottom_ids[1]) from val_original (top_ids[0])
                original_val_dim_id = top_ids[0]
                if original_val_dim_id not in known_dim_values:
                    raise ValueError(f"Split: Dim '{original_val_dim_id}' not found.")
                
                val_original = known_dim_values[original_val_dim_id]
                split_lengths = params['lengths'] # [len_major_output, len_minor_output]
                # val_minor = val_original % len_minor_output
                # val_major = val_original // len_minor_output
                len_minor_output = split_lengths[1]
                
                known_dim_values[bottom_ids[1]] = val_original % len_minor_output # Minor
                known_dim_values[bottom_ids[0]] = val_original // len_minor_output # Major
            else:
                raise NotImplementedError(f"Transform type '{transform_type}' is not implemented.")

        # Extract final coordinates based on bottom_tensor_view
        final_coords_desc = adaptor_encoding['bottom_tensor_view']['dims']
        final_coords = [0] * len(final_coords_desc)
        if not final_coords_desc and not known_dim_values: # Scalar output from scalar input
            return [] 
            
        for i, dim_desc in enumerate(final_coords_desc):
            dim_id = dim_desc[0]
            if dim_id not in known_dim_values:
                raise ValueError(f"Final coordinate for dim_id '{dim_id}' not found. Known: {known_dim_values.keys()}")
            final_coords[i] = known_dim_values[dim_id]
            
        return final_coords

    def _invert_tensor_adaptor_transformations(self, target_coords: typing.Sequence[int], adaptor_encoding: dict) -> typing.Sequence[int]:
        """
        Applies the inverse of a series of tensor transformations.
        It computes the 'top_tensor_view' coordinates from the 'bottom_tensor_view' coordinates.
        """
        known_dim_values: typing.Dict[typing.Any, int] = {}
        bottom_dims_desc = adaptor_encoding['bottom_tensor_view']['dims']

        if not bottom_dims_desc and not target_coords: # Scalar case (0-dim target)
            pass
        elif len(target_coords) != len(bottom_dims_desc):
            raise ValueError(
                f"Target coordinates length ({len(target_coords)}) does not match "
                f"bottom_tensor_view dimension count ({len(bottom_dims_desc)}).")
        else:
            for i, dim_desc in enumerate(bottom_dims_desc):
                known_dim_values[dim_desc[0]] = target_coords[i]

        for transform_def in reversed(adaptor_encoding['transforms']):
            transform_type = transform_def['type']
            params = transform_def['parameters']
            fw_top_ids = transform_def['top_ids_']      # Dimension IDs input to forward transform
            fw_bottom_ids = transform_def['bottom_ids_'] # Dimension IDs output by forward transform

            if transform_type == "PassThrough":
                if not fw_bottom_ids : # scalar to scalar handled by empty list
                     if not fw_top_ids: raise ValueError("PassThrough inv: fw_bottom_ids and fw_top_ids cannot both be empty")
                     if fw_top_ids[0] not in known_dim_values: # If created by transform
                          known_dim_values[fw_top_ids[0]] = 0 # Placeholder, assumes created from nothing specific
                elif fw_bottom_ids[0] not in known_dim_values:
                    raise ValueError(f"PassThrough inverse: Dim '{fw_bottom_ids[0]}' not found in known_dim_values.")
                else:
                    val_from_bottom = known_dim_values[fw_bottom_ids[0]]
                    known_dim_values[fw_top_ids[0]] = val_from_bottom
            
            elif transform_type == "Merge":
                merged_val_dim_id = fw_bottom_ids[0]
                if merged_val_dim_id not in known_dim_values:
                    raise ValueError(f"Merge inverse: Dim '{merged_val_dim_id}' not found for merged value.")
                merged_val = known_dim_values[merged_val_dim_id]
                
                major_input_dim_id = fw_top_ids[0]
                minor_input_dim_id = fw_top_ids[1]
                len_minor = params['lengths_of_inputs'][1]
                
                known_dim_values[minor_input_dim_id] = merged_val % len_minor
                known_dim_values[major_input_dim_id] = merged_val // len_minor
            
            elif transform_type == "Embed":
                padded_val_dim_id = fw_bottom_ids[0]
                if padded_val_dim_id not in known_dim_values:
                    raise ValueError(f"Embed inverse: Dim '{padded_val_dim_id}' not found for padded value.")
                padded_val = known_dim_values[padded_val_dim_id]
                
                original_dim_id = fw_top_ids[0]
                pre_padding = params['pre_padding']
                known_dim_values[original_dim_id] = padded_val - pre_padding
                
            elif transform_type == "Split":
                major_output_dim_id = fw_bottom_ids[0]
                minor_output_dim_id = fw_bottom_ids[1]

                if major_output_dim_id not in known_dim_values or minor_output_dim_id not in known_dim_values:
                    raise ValueError(f"Split inverse: Dim '{major_output_dim_id}' or '{minor_output_dim_id}' not found.")
                val_major = known_dim_values[major_output_dim_id]
                val_minor = known_dim_values[minor_output_dim_id]
                
                original_dim_id = fw_top_ids[0]
                split_lengths = params['lengths'] 
                len_minor_output = split_lengths[1]
                
                known_dim_values[original_dim_id] = val_major * len_minor_output + val_minor
            else:
                raise NotImplementedError(f"Inverse for transform type '{transform_type}' is not implemented.")

        top_dims_desc_final = adaptor_encoding['top_tensor_view']['dims']
        inverted_coords = [0] * len(top_dims_desc_final)
        if not top_dims_desc_final and not known_dim_values : # scalar input resulted in scalar output
            return []

        for i, dim_desc in enumerate(top_dims_desc_final):
            dim_id = dim_desc[0]
            if dim_id not in known_dim_values:
                 raise ValueError(
                    f"Inverted coordinate for dim_id '{dim_id}' not found in known_dim_values. "
                    f"Available keys: {list(known_dim_values.keys())}")
            inverted_coords[i] = known_dim_values[dim_id]
            
        return inverted_coords

    def calculate_rs_index_from_ps_index(self, ps_idx: typing.Sequence[int]) -> typing.Sequence[int]:
        """
        Calculates the R-space multi-index from a P-space multi-index.
        Corresponds to TileDistribution::CalculateRsIndexFromPsIndex in C++.
        ps_idx elements are multiplied by corresponding elements in ps_over_rs_derivative_
        and summed up for each R dimension.
        """
        if len(ps_idx) != self.NDimPs:
            raise ValueError(f"Length of ps_idx ({len(ps_idx)}) must match NDimPs ({self.NDimPs}).")
        
        if self.NDimRs == 0:
            return [] # No R-dimensions, so R-index is empty

        rs_idx = [0] * self.NDimRs
        ps_over_rs_derivative = self.DstrEncode.detail['ps_over_rs_derivative_']

        for r_dim_idx in range(self.NDimRs):
            accumulated_r_val = 0
            for p_dim_idx in range(self.NDimPs):
                # ps_over_rs_derivative_ has shape [NDimP][NDimR]
                derivative_val = ps_over_rs_derivative[p_dim_idx][r_dim_idx]
                accumulated_r_val += ps_idx[p_dim_idx] * derivative_val
            rs_idx[r_dim_idx] = accumulated_r_val
        return rs_idx

    def calculate_index(self, idx_p: typing.Sequence[int], idx_y: typing.Sequence[int]) -> int:
        """
        Calculates the final flattened index based on P (tile) and Y (within-tile) coordinates.
        Mimics C++: tile_distribution.hpp, calculate_index function.
        """
        # This method needs to use the PsYs2XsAdaptor and Ys2DAdaptor.
        # The logic involves:
        # 1. Using PsYs2XsAdaptor to transform (idx_p, idx_y) into xs_coords (bottom-view of this adaptor)
        #    This is like CalculateBottomIndex(ps_ys_coords) -> xs_coords
        # 2. Flatten xs_coords into a scalar offset (flattened_x_offset).
        # 3. Get product_of_x_lengths (total elements in an X-tile).
        # 4. Using Ys2DAdaptor to transform idx_y into d_scalar_idx (bottom-view of this adaptor, which is D)
        #    This is like CalculateBottomIndex(ys_coords) -> d_scalar_idx
        # 5. final_index = d_scalar_idx * product_of_x_lengths + flattened_x_offset

        if self.NDimPs != len(idx_p):
            raise ValueError(f"Length of idx_p ({len(idx_p)}) does not match NDimPs ({self.NDimPs})")
        if self.NDimYs != len(idx_y):
            raise ValueError(f"Length of idx_y ({len(idx_y)}) does not match NDimYs ({self.NDimYs})")

        # Step 1 & 2: Calculate xs_coords and flattened_x_offset
        xs_coords = self._calculate_xs_from_ps_ys(idx_p, idx_y, self.PsYs2XsAdaptor)
        
        # Flatten xs_coords
        # x_lengths are from PsYs2XsAdaptor['BottomView']['BottomDimensionNameLengths']
        # Order by key "0", "1", ...
        x_lengths_map = self.PsYs2XsAdaptor['BottomView']['BottomDimensionNameLengths']
        
        flattened_x_offset = 0
        product_of_x_lengths = 1
        
        if self.NDimX > 0 :
            # sorted_x_logical_indices = sorted([int(k) for k in x_lengths_map.keys()]) # OLD, Incorrect: keys are "X0", "X1"
            # We need the logical indices 0, 1, ... which are keys in BottomDimensionIdToName
            sorted_x_logical_indices = sorted([int(k) for k in self.PsYs2XsAdaptor['BottomView']['BottomDimensionIdToName'].keys()])
            
            current_stride = 1
            for i in range(len(sorted_x_logical_indices) -1, -1, -1):
                x_logical_idx = sorted_x_logical_indices[i]
                x_len_name = self.PsYs2XsAdaptor['BottomView']['BottomDimensionIdToName'][x_logical_idx] # CORRECTED: use int key
                x_len = x_lengths_map[x_len_name]
                
                flattened_x_offset += xs_coords[x_logical_idx] * current_stride
                current_stride *= x_len
            product_of_x_lengths = current_stride # After loop, current_stride is total product
        elif self.NDimX == 0: # No X dimensions (e.g. only R-dims)
             flattened_x_offset = 0 # No X-contribution to offset
             product_of_x_lengths = 1 # Multiplicative identity


        # Step 3: Calculate d_scalar_idx
        d_scalar_idx = self._calculate_d_from_ys(idx_y, self.Ys2DDescriptor)

        # Step 4: Calculate final_index
        final_index = d_scalar_idx * product_of_x_lengths + flattened_x_offset
        
        return final_index

    def get_lengths(self) -> typing.Sequence[int]:
        """
        Returns the lengths of the problem space dimensions (PsLengths).
        Corresponds to TileDistribution::GetLengths() in C++.
        """
        return self.DstrEncode.PsLengths

    def get_ys_lengths(self) -> typing.Sequence[int]:
        """
        Returns the lengths of the Y-space dimensions.
        Corresponds to TileDistribution::GetYsLengths() in C++ which returns dstr_encoder_detail_.ys_lengths_.
        """
        return self.DstrEncode.detail['ys_lengths_']

    def get_rs_lengths(self) -> typing.Sequence[int]:
        """
        Returns the lengths of the R-space dimensions.
        Corresponds to TileDistribution::GetRsLengths() in C++ which returns dstr_encoder_.rs_lengths_.
        """
        return self.DstrEncode.RsLengths

    def get_distributed_spans(self) -> typing.Sequence[typing.Sequence[typing.Tuple[int, int]]]:
        """
        Returns the distributed spans lengths.
        Corresponds to TileDistribution::GetDistributedSpans() which returns detail_.distributed_spans_lengthss_.
        The structure is a sequence (for each YS dim) of sequences (for each P dim) of tuples (span_idx, length).
        """
        return self.DstrEncode.detail['distributed_spans_lengthss_']

    def get_y_indices_from_distributed_indices(self, d_scalar_idx: int) -> typing.Sequence[int]:
        """
        Calculates the Y coordinates (within-tile indices) from a scalar distributed index 'd'.
        Mimics C++: tile_distribution.hpp, get_y_indices_from_distributed_indices function.
        This uses the Ys2DAdaptor in reverse (CalculateTopIndex).
        """
        # The Ys2DAdaptor transforms Ys (top view) to D (bottom view, scalar).
        # We need to go from D (scalar) to Ys. This is CalculateTopIndex.
        
        ys_coords = self._calculate_ys_from_d(d_scalar_idx, self.Ys2DDescriptor)

        if len(ys_coords) != self.NDimYs:
            # This could happen if NDimYs is 0 and ys_coords is empty, which is fine.
            # Or if there's a mismatch.
            if not (self.NDimYs == 0 and not ys_coords):
                 raise ValueError(f"Calculated ys_coords length ({len(ys_coords)}) does not match NDimYs ({self.NDimYs})")
        
        return ys_coords

    def print_distribution(self):
        """
        Prints the details of the tile distribution, including the encoding and adaptor structures.
        Mimics parts of the C++ TileDistribution::print() method.
        """
        print("=================================================================")
        print(f"TileDistributionPedantic: {getattr(self, 'tile_name', 'N/A')}")
        print("-----------------------------------------------------------------")
        print("### Tile Distribution Encoding Details: ###")
        self.DstrEncode.print_encoding()
        print("-----------------------------------------------------------------")
        print("### PsYs-to-Xs Adaptor (self.PsYs2XsAdaptor): ###")
        print(json.dumps(self.PsYs2XsAdaptor, indent=2, default=str))
        print("-----------------------------------------------------------------")
        print("### Ys-to-D Descriptor (self.Ys2DDescriptor): ###")
        print(json.dumps(self.Ys2DDescriptor, indent=2, default=str))
        print("-----------------------------------------------------------------")
        print("### DstrDetail (e.g., rh_major_minor_to_adaptor_hidden_idss_): ###")
        print(json.dumps(self.DstrDetail, indent=2, default=str))
        print("-----------------------------------------------------------------")
        print("Call calculate_index(idx_p, idx_y) for specific mappings.")
        print("=================================================================")

    def get_occupancy(self) -> float:
        """
        Calculates the occupancy based on the flat_tile_data_map.
        Occupancy is the ratio of validly mapped cells in a representative P-tile
        to the total cells in that P-tile's R-H shape.
        Error markers (-99, -999) in the map are considered unmapped.
        """
        flat_data_map = self.get_flat_tile_data_map()
        if not flat_data_map: return 0.0
        first_p_tile_coords = next(iter(flat_data_map))
        tile_layout = flat_data_map[first_p_tile_coords]
        tile_r, tile_h = self.tile_shape
        total_cells_in_tile = tile_r * tile_h
        if total_cells_in_tile == 0: return 0.0
        mapped_elements = sum(1 for data_idx in tile_layout.values() if data_idx not in [-99, -999])
        return mapped_elements / total_cells_in_tile if total_cells_in_tile > 0 else 0.0
    
    def get_utilization(self) -> float:
        """
        Calculates utilization based on occupancy and an efficiency factor.
        Mimics the approach from the original tiler.py.
        """
        occupancy = self.get_occupancy()
        efficiency_factor = 0.95 
        return occupancy * efficiency_factor

    def set_tile_name(self, name: str):
        """
        Set the tile name for this tile distribution.
        """
        if not hasattr(self.DstrEncode, 'encoding_input_dict'): 
            self.DstrEncode.encoding_input_dict = {} 
        self.DstrEncode.encoding_input_dict['_tile_name'] = name

    def set_source_code(self, code: str):
        """
        Set the source code for this tile distribution to show in the visualization.
        """
        self.DstrEncode.set_source_code(code)

    def _get_compatible_dimensions_dict(self) -> typing.Dict[str, typing.List[int]]:
        """
        Creates a 'dimensions' dictionary similar to tiler.py's output,
        using pedantically available information where possible.
        P-dim and Y-dim lengths are simplified here as they are complex to derive
        solely from the current encoding for this specific flat list output.
        X-dims are based on the pedantic tile_shape.
        """
        p_dims_repr = [self.NDimPs] if self.NDimPs > 0 else [1]
        y_dims_repr = [self.NDimYs] if self.NDimYs > 0 else [1]
        x_dims_repr = self.tile_shape 
        return {'p_dims': p_dims_repr, 'y_dims': y_dims_repr, 'x_dims': x_dims_repr}

    def get_flat_tile_data_map(self) -> typing.Dict[typing.Tuple[int, ...], typing.Dict[typing.Tuple[int, int], int]]:
        """
        Generates a detailed data map: (P-coords) -> {(R,H)-cell -> data_index}.
        This was the previous implementation of get_visualization_data.
        """
        visualization_data: typing.Dict[typing.Tuple[int, ...], typing.Dict[typing.Tuple[int, int], int]] = {}
        if not self.DstrEncode.PsLengths and self.NDimPs > 0 : return {}
        
        p_coords_ranges = [range(length) for length in self.DstrEncode.PsLengths] if self.NDimPs > 0 else [range(1)]
        num_p_dims_iter = self.NDimPs if self.NDimPs > 0 else 1
        current_p_indices = [0] * num_p_dims_iter

        while True:
            p_coords = tuple(current_p_indices) if self.NDimPs > 0 else tuple()
            tile_data: typing.Dict[typing.Tuple[int, int], int] = {}
            tile_shape_r, tile_shape_h = self.tile_shape
            # if tile_shape_r <= 0 or tile_shape_h <= 0: pass # This condition doesn't make sense here

            for r_in_tile in range(tile_shape_r if tile_shape_r > 0 else 1): # Iterate at least once if shape is 0
                for h_in_tile in range(tile_shape_h if tile_shape_h > 0 else 1): # Iterate at least once if shape is 0
                    ys_coords = [-1] * self.NDimYs 
                    # Simplified YS construction for visualization: try to map R,H to available YS dims
                    # This part is highly heuristic and not strictly pedantic if YS dims aren't just R,H
                    available_ys_indices = list(range(self.NDimYs))
                    
                    if available_ys_indices: # Assign R to first available YS
                        ys_coords[available_ys_indices.pop(0)] = r_in_tile
                    if available_ys_indices: # Assign H to next available YS
                        ys_coords[available_ys_indices.pop(0)] = h_in_tile
                    
                    # Fill remaining YS with 0 (placeholder)
                    for i in range(self.NDimYs):
                        if ys_coords[i] == -1: ys_coords[i] = 0
                    
                    final_idx = -99 # Default error
                    try:
                        final_idx = self.calculate_index(p_coords, ys_coords if self.NDimYs > 0 else [])
                    except Exception as e:
                        # print(f"Error in calculate_index for p={p_coords}, y={ys_coords}: {e}")
                        final_idx = -999                             
                    tile_data[(r_in_tile, h_in_tile)] = final_idx
            visualization_data[p_coords] = tile_data
            
            if num_p_dims_iter == 0 or (self.NDimPs == 0 and num_p_dims_iter == 1 and not self.DstrEncode.PsLengths): # Ensure single iteration for NDimP=0
                 break

            increment_done = False
            for i in range(num_p_dims_iter - 1, -1, -1):
                current_p_indices[i] += 1
                if current_p_indices[i] < p_coords_ranges[i].stop:
                    increment_done = True
                    break
                else:
                    current_p_indices[i] = 0 
            if not increment_done: break 
        return visualization_data

    def get_visualization_data(self) -> typing.Dict[str, typing.Any]:
        """
        Get data structured for compatibility with app.py visualizer.
        This includes hierarchical structure, occupancy, utilization, etc.
        """
        hierarchical_structure = self.calculate_hierarchical_tile_structure()
        return {
            'tile_shape': self.tile_shape, 
            'thread_mapping': self.thread_mapping,
            'dimensions': self._get_compatible_dimensions_dict(),
            'occupancy': self.get_occupancy(), 
            'utilization': self.get_utilization(), 
            'hierarchical_structure': hierarchical_structure, 
            'source_code': self.DstrEncode.source_code,
        }

    def calculate_hierarchical_tile_structure(self) -> typing.Dict[str, typing.Any]:
        """
        Calculates the hierarchical structure of tiles for visualization.
        Configurable via 'visualization_hints' in the input encoding_dict.
        Hints:
            'thread_per_warp_p_indices': List of P-dim index(es) (e.g., [idx_m] or [idx_m, idx_n])
            'warp_per_block_p_indices': List of P-dim index(es) (e.g., [idx_0, idx_1, ...])
            'vector_dim_ys_index': YS-dim index for vector length.
            'repeat_factor_ys_index': YS-dim index for repeat factor.
        """
        # DEBUG PRINT - Check if function is entered and see basic params
        print(f"DEBUG: ENTERING calculate_hierarchical_tile_structure")
        print(f"DEBUG:   NDimPs = {self.NDimPs}")
        print(f"DEBUG:   NDimYs = {self.NDimYs}")
        print(f"DEBUG:   visualization_hints = {self.encoding_input_dict.get('visualization_hints', {})}")
        print(f"DEBUG:   DstrEncode.variables = {self.DstrEncode.variables}")
        print(f"DEBUG:   DstrEncode.HsLengthss = {self.DstrEncode.HsLengthss}")

        # Ensure detail is computed if not already
        if not self.DstrEncode.detail:
            self.DstrEncode.detail = self.DstrEncode._compute_detail()

        hints = self.encoding_input_dict.get("visualization_hints", {})

        # Initialize with defaults that can be overridden by hints
        hierarchical_info = {
            'BlockSize': [], 'ThreadPerWarp': [1, 1], 'WarpPerBlock': [1, 1], 
            'VectorDimensions': [1], 'Repeat': [1, 1], 'ThreadBlocks': {},
            'TileName': self.encoding_input_dict.get('_tile_name', "Pedantic Tile Distribution"),
            'DimensionValues': [],
            'VectorDimensionYSIndex': -1 # NEW: To store which YS dim is the vector
        }

        # Populate DimensionValues from HsLengthss (raw, for annotation)
        # Also resolve variable names to values for HsLengthss for internal use
        resolved_hs_lengthss = []
        for h_seq_orig in self.DstrEncode.HsLengthss:
            resolved_seq = []
            for val_or_var_name in h_seq_orig:
                if isinstance(val_or_var_name, int):
                    resolved_seq.append(val_or_var_name)
                    hierarchical_info['DimensionValues'].append(val_or_var_name)
                elif isinstance(val_or_var_name, str):
                    resolved_val = self.DstrEncode.variables.get(val_or_var_name, 1) # Default to 1 if var not found for safety
                    resolved_seq.append(resolved_val)
                    hierarchical_info['DimensionValues'].append(resolved_val)
                else: # Should not happen
                    resolved_seq.append(1)
                    hierarchical_info['DimensionValues'].append(1)
            resolved_hs_lengthss.append(resolved_seq)

        # --- Helper function to get lengths for given P-dimension indices ---
        # (get_lengths_for_p_indices is defined here or accessible)
        # Note: This was an inner function, moved out for clarity or ensure it's method of class
        # For this edit, assuming it's an accessible method: self._get_lengths_for_p_dim_component

        # Helper for product
        def product(iterable):
            res = 1
            for x in iterable:
                if x <= 0 : return 0 # Product with zero or negative is problematic for tile dims
                res *= x
            return res if iterable else 1


        # --- Process VectorDimensions --- 
        vec_ys_idx_hint = hints.get('vector_dim_ys_index')
        if vec_ys_idx_hint is not None and 0 <= vec_ys_idx_hint < self.NDimYs:
            # Use explicit hint if provided
            if vec_ys_idx_hint < len(self.DstrEncode.detail.get('ys_lengths_', [])):
                 val = self.DstrEncode.detail['ys_lengths_'][vec_ys_idx_hint]
                 hierarchical_info['VectorDimensions'] = [val if val > 0 else 1]
                 hierarchical_info['VectorDimensionYSIndex'] = vec_ys_idx_hint if (val if val > 0 else 1) > 1 else -1 # Store hint index if vector is non-scalar
        elif self.NDimYs > 0:
            # IMPROVED INFERENCE: Find Ys that map to the highest/last position in H sequences
            # Group vector dimensions by which H sequence they belong to (for M/N dimensions)
            vector_dimensions_by_h = {}  # Maps H-idx to list of (value, y_idx)
            
            # Find Y dimensions that map to the last position of an H sequence
            for y_idx in range(self.NDimYs):
                if y_idx < len(self.DstrEncode.Ys2RHsMajor) and y_idx < len(self.DstrEncode.Ys2RHsMinor):
                    y_major = self.DstrEncode.Ys2RHsMajor[y_idx]
                    y_minor = self.DstrEncode.Ys2RHsMinor[y_idx]
                    
                    # If Y maps to H sequence (major > 0)
                    if y_major > 0 and y_major <= len(self.DstrEncode.HsLengthss):
                        h_idx = y_major - 1  # Convert to 0-based index
                        h_seq = self.DstrEncode.HsLengthss[h_idx]
                        
                        # Check if it maps to the last element of the H sequence
                        if y_minor == len(h_seq) - 1:
                            val = h_seq[y_minor]
                            if val > 0:
                                if h_idx not in vector_dimensions_by_h:
                                    vector_dimensions_by_h[h_idx] = []
                                vector_dimensions_by_h[h_idx].append((val, y_idx))
            
            # Use the found vector dimensions, or fall back to prior heuristic
            if vector_dimensions_by_h:
                # Process each H sequence's vector dimensions
                all_vector_values = []
                all_vector_y_indices = []
                
                # Sort by H-index to have consistent ordering (H0 first, then H1, etc.)
                for h_idx in sorted(vector_dimensions_by_h.keys()):
                    # For each H, combine its vector dimensions (usually just one per H)
                    h_vectors = vector_dimensions_by_h[h_idx]
                    h_combined_value = 1
                    for val, y_idx in h_vectors:
                        h_combined_value *= val
                        all_vector_y_indices.append(y_idx)
                    
                    all_vector_values.append(h_combined_value)
                
                # Store multi-dimensional vector information
                hierarchical_info['VectorDimensions'] = all_vector_values
                hierarchical_info['VectorDimensionYSIndex'] = all_vector_y_indices[0] if all_vector_y_indices else -1
                # Still calculate combined value for compatibility
                combined_vector_value = 1
                for val in all_vector_values:
                    combined_vector_value *= val
                hierarchical_info['VectorK'] = combined_vector_value  # Total vector elements
                
                print(f"DEBUG: Found vector dimensions from Ys mapping to last H positions by H sequence: {vector_dimensions_by_h}")
                print(f"DEBUG: Vector dimensions: {all_vector_values}")
            elif self.DstrEncode.detail.get('ys_lengths_'):
                # Fall back to old heuristic - use last Y if its length is typical for vectors
                last_ys_len = self.DstrEncode.detail['ys_lengths_'][-1]
                if 1 < last_ys_len <= 16:
                    hierarchical_info['VectorDimensions'] = [last_ys_len]
                    hierarchical_info['VectorDimensionYSIndex'] = self.NDimYs - 1 # Index of the last YS dim
            # else, default to [1] is already set, VectorDimensionYSIndex remains -1

        # --- Process ThreadPerWarp ---
        tpw_p_indices_hint = hints.get('thread_per_warp_p_indices')
        
        # Check for explicit override first
        tpw_override = hints.get('thread_per_warp_override')
        if tpw_override and isinstance(tpw_override, list):
            # Use explicit override values
            if len(tpw_override) >= 2:
                hierarchical_info['ThreadPerWarp'] = [tpw_override[0], tpw_override[1]]
            elif len(tpw_override) == 1:
                hierarchical_info['ThreadPerWarp'] = [tpw_override[0], 1]
            print(f"DEBUG: Using ThreadPerWarp override: {hierarchical_info['ThreadPerWarp']}")
        elif tpw_p_indices_hint is not None:
            current_hint_val = tpw_p_indices_hint
            if not isinstance(current_hint_val, list):
                current_hint_val = [current_hint_val] # Ensure it's a list

            if len(current_hint_val) == 1:
                m_lengths = self._get_lengths_for_p_dim_component(current_hint_val[0])
                hierarchical_info['ThreadPerWarp'] = [product(m_lengths) if m_lengths else 1, 1]
            elif len(current_hint_val) >= 2:
                m_lengths = self._get_lengths_for_p_dim_component(current_hint_val[0])
                n_lengths = self._get_lengths_for_p_dim_component(current_hint_val[1])
                hierarchical_info['ThreadPerWarp'] = [product(m_lengths) if m_lengths else 1, product(n_lengths) if n_lengths else 1]
            # else: uses default [1,1] initialized above
        elif self.NDimPs >= 1: # INFERENCE for ThreadPerWarp
            # Default to [1,1] initially
            tpw_m_len = 1
            tpw_n_len = 1

            if self.NDimPs >= 2: # Use P1 for ThreadPerWarp
                p1_contrib_lens = self._get_lengths_for_p_dim_component(1) # Get all lengths P1 maps to
                if p1_contrib_lens:
                    tpw_m_len = p1_contrib_lens[0] # First component for M-dim
                    if len(p1_contrib_lens) > 1:
                        tpw_n_len = p1_contrib_lens[1] # Second component for N-dim
                    else:
                        tpw_n_len = 1 # If P1 maps to only one, N-dim is 1
                else:
                    # P1 exists but has no valid mappings - keep ThreadPerWarp at [1,1]
                    print(f"DEBUG: P1 exists but no component lengths are mapped to it. Setting ThreadPerWarp=[1,1].")
                    tpw_m_len = 1
                    tpw_n_len = 1
                # If p1_contrib_lens is empty, tpw_m_len/tpw_n_len remain 1
            elif self.NDimPs == 1: # Only P0 exists, special care needed
                # By default, in Composable Kernels when there's only one P dimension (P0),
                # it's more often used for WarpPerBlock, not ThreadPerWarp, especially when
                # mapping to R dimensions or the first components in H-sequences.
                # For examples like:
                # - sequence<MWarp>, tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>
                # - sequence<WarpPerBlock_N>, tuple<sequence<Repeat_M, WarpPerBlock_M>, sequence<Repeat_K>>
                # ThreadPerWarp should default to [1,1]
                
                # Check Encoding for Key Indications:
                # 1. When only P0 exists and maps to R (major=0) or first H component (minor=0/1)
                # 2. When the mapped dimensions have names like "Warp", "WarpPerBlock", etc.
                # 3. When the IndexMapping shows P0 mapping to elements with "Warp" in the name
                
                # Explicit check - look at the pattern of P0 mappings
                p0_maps_to_warp = False
                
                # Check if this looks like a WarpPerBlock mapping pattern
                if self.DstrEncode.Ps2RHssMajor and self.DstrEncode.Ps2RHssMinor:
                    p0_major_mappings = self.DstrEncode.Ps2RHssMajor[0] if 0 < len(self.DstrEncode.Ps2RHssMajor) else []
                    p0_minor_mappings = self.DstrEncode.Ps2RHssMinor[0] if 0 < len(self.DstrEncode.Ps2RHssMinor) else []
                    
                    # Check if P0 maps primarily to R-dimensions or early H components
                    r_mappings_count = sum(1 for major in p0_major_mappings if major == 0)
                    first_h_component_mappings = sum(1 for major, minor in zip(p0_major_mappings, p0_minor_mappings) 
                                                 if major > 0 and (minor == 0 or minor == 1))
                    
                    if r_mappings_count > 0 or first_h_component_mappings > 0:
                        # This looks like a WarpPerBlock mapping
                        p0_maps_to_warp = True
                
                # For the specific case where P0 maps to components related to WarpPerBlock
                # or we only have a "warp level" example without thread-level distribution
                # (detected from P0's mapping pattern), set ThreadPerWarp to [1,1]
                if p0_maps_to_warp:
                    print(f"DEBUG: NDimPs=1 (only P0) and mapping pattern suggests WarpPerBlock. Setting ThreadPerWarp=[1,1]")
                    tpw_m_len = 1
                    tpw_n_len = 1
                else:
                    # Traditional inference from P0 (rare case, typically P1 does ThreadPerWarp)
                    p0_contrib_lens = self._get_lengths_for_p_dim_component(0)
                    if p0_contrib_lens:
                        tpw_m_len = p0_contrib_lens[0]
                        if len(p0_contrib_lens) > 1:
                            tpw_n_len = p0_contrib_lens[1]
                        else:
                            tpw_n_len = 1
            
            # If we have a thread_per_warp_p_indices hint with an empty list, explicitly set [1,1]
            if hints.get('thread_per_warp_p_indices') == []:
                print(f"DEBUG: Explicit empty thread_per_warp_p_indices hint. Setting ThreadPerWarp=[1,1]")
                tpw_m_len = 1
                tpw_n_len = 1
                
            hierarchical_info['ThreadPerWarp'] = [max(1,tpw_m_len), max(1,tpw_n_len)]
            # Default [1,1] is already set if NDimPs = 0 or components are not found
            # For NDimPs < 2 (i.e. only P0 or no P's), if the user expects TPW from P0, 
            # they might need a hint if this P1-centric logic isn't desired.
            # This matches tiler.py's P1-driven TPW.

        # --- Process WarpPerBlock ---
        wpb_p_indices_hint = hints.get('warp_per_block_p_indices')
        wpb_m_len = 1
        wpb_n_len = 1 # Often WPB is [M_warps, 1]
        
        # Check for explicit override first
        wpb_override = hints.get('warp_per_block_override')
        if wpb_override and isinstance(wpb_override, list):
            # Use explicit override values
            if len(wpb_override) >= 2:
                hierarchical_info['WarpPerBlock'] = [wpb_override[0], wpb_override[1]]
            elif len(wpb_override) == 1:
                hierarchical_info['WarpPerBlock'] = [wpb_override[0], 1]
            print(f"DEBUG: Using WarpPerBlock override: {hierarchical_info['WarpPerBlock']}")
        elif wpb_p_indices_hint is not None:
            current_hint_val = wpb_p_indices_hint
            if not isinstance(current_hint_val, list):
                current_hint_val = [current_hint_val]
            
            wpb_combined_lengths_m = []
            wpb_combined_lengths_n = [] 

            if len(current_hint_val) == 1: 
                 wpb_combined_lengths_m.extend(self._get_lengths_for_p_dim_component(current_hint_val[0]))
            elif len(current_hint_val) >= 2: 
                 wpb_combined_lengths_m.extend(self._get_lengths_for_p_dim_component(current_hint_val[0]))
                 wpb_combined_lengths_n.extend(self._get_lengths_for_p_dim_component(current_hint_val[1]))

            if wpb_combined_lengths_m: wpb_m_len = product(wpb_combined_lengths_m)
            if wpb_combined_lengths_n: wpb_n_len = product(wpb_combined_lengths_n)
            hierarchical_info['WarpPerBlock'] = [max(1,wpb_m_len), max(1,wpb_n_len)]
            if not wpb_combined_lengths_m and not wpb_combined_lengths_n:
                hierarchical_info['WarpPerBlock'] = [1, 1]

        elif self.NDimPs >= 1: # INFERENCE for WarpPerBlock using P0 (aligns with tiler.py)
            p0_contrib_lens = self._get_lengths_for_p_dim_component(0)
            if p0_contrib_lens:
                wpb_m_len = p0_contrib_lens[0] # First component for M-dim
                if len(p0_contrib_lens) > 1:
                    wpb_n_len = p0_contrib_lens[1] # Second component for N-dim (less common for WPB)
                else:
                    wpb_n_len = 1 # If P0 maps to only one, N-dim is 1 for WPB
            # If p0_contrib_lens is empty, wpb_m_len/n_len remain 1
            hierarchical_info['WarpPerBlock'] = [max(1,wpb_m_len), max(1,wpb_n_len)]
        # else: default [1,1] already set for WarpPerBlock (from initialization if NDimPs = 0)

        # If WPB is still [1,1] after hints/inference, and we have some P-dims,
        # apply a more common default for visualization (e.g., 4 warps in M-dim).
        # This is a heuristic for better visual representation when true inference is hard.
        if hierarchical_info['WarpPerBlock'] == [1,1] and self.NDimPs >= 1:
            # Check if ThreadPerWarp looks reasonable (e.g., not just [1,1])
            tpw_m, tpw_n = hierarchical_info['ThreadPerWarp']
            if tpw_m * tpw_n > 1: # If TPW is not trivial
                 print(f"INFO: WPB resulted in [1,1]. Applying common default [4,1] for visualization as NDimPs={self.NDimPs} >= 1 and TPW is non-trivial ({tpw_m}x{tpw_n}).")
                 hierarchical_info['WarpPerBlock'] = [4, 1]


        # --- Process Repeat --- 
        rep_ys_idx_hint = hints.get('repeat_factor_ys_index')
        # Default repeat is [1,1] (set at initialization)
        if rep_ys_idx_hint is not None and 0 <= rep_ys_idx_hint < self.NDimYs:
            # Use explicit hint if provided
            if rep_ys_idx_hint < len(self.DstrEncode.detail.get('ys_lengths_', [])):
                val = self.DstrEncode.detail['ys_lengths_'][rep_ys_idx_hint]
                # Current hint logic applies hint value to M-dimension of Repeat, N-dim remains 1.
                hierarchical_info['Repeat'] = [max(1, val), 1]
        else:
            # IMPROVED INFERENCE: Find Ys that map to the first position in H sequences
            repeat_y_indices = []
            repeat_values = []
            
            for y_idx in range(self.NDimYs):
                if y_idx < len(self.DstrEncode.Ys2RHsMajor) and y_idx < len(self.DstrEncode.Ys2RHsMinor):
                    y_major = self.DstrEncode.Ys2RHsMajor[y_idx]
                    y_minor = self.DstrEncode.Ys2RHsMinor[y_idx]
                    
                    # If Y maps to H sequence (major > 0)
                    if y_major > 0 and y_major <= len(self.DstrEncode.HsLengthss):
                        h_idx = y_major - 1  # Convert to 0-based index
                        
                        # Check if it maps to the first element of the H sequence
                        if y_minor == 0:
                            h_seq = self.DstrEncode.HsLengthss[h_idx]
                            val = h_seq[y_minor]
                            if val > 0:
                                repeat_y_indices.append(y_idx)
                                repeat_values.append(val)
            
            # Use found repeat factors if available
            if repeat_y_indices:
                # For now, use up to 2 repeat factors for M and N dimensions
                repeat_m = 1
                repeat_n = 1
                
                # Sort by index to ensure consistent assignment
                sorted_repeats = sorted(zip(repeat_y_indices, repeat_values))
                
                if len(sorted_repeats) >= 1:
                    repeat_m = sorted_repeats[0][1]
                if len(sorted_repeats) >= 2:
                    repeat_n = sorted_repeats[1][1]
                
                hierarchical_info['Repeat'] = [max(1, repeat_m), max(1, repeat_n)]
                print(f"DEBUG: Found repeat factors from Ys mapping to first H positions: {sorted_repeats}")
            # Otherwise, it stays at default [1,1]
        
        # Calculate BlockSize from derived/default TPW and WPB
        tpw_m, tpw_n = hierarchical_info['ThreadPerWarp']
        wpb_m, wpb_n = hierarchical_info['WarpPerBlock']
        
        # Assuming hierarchical_info['Repeat'] is like [val], applying to M-dimension of repeat by default
        # repeat_val_m = hierarchical_info['Repeat'][0] if hierarchical_info['Repeat'] and len(hierarchical_info['Repeat']) > 0 else 1
        # repeat_val_n = 1 # Default N-dimension repeat to 1, unless Repeat becomes a 2-element list in future
        repeat_val_m, repeat_val_n = hierarchical_info['Repeat'] # Now Repeat is [m,n]
        
        hierarchical_info['BlockSize'] = [
            tpw_m * wpb_m * repeat_val_m,
            tpw_n * wpb_n * repeat_val_n
        ]

        # ThreadBlocks visualization (Placeholder: generic grid, not pedantic P-coords per thread)
        thread_blocks_viz = {}
        # product() helper should be available from earlier in the function
        num_warps_to_iterate = product(hierarchical_info['WarpPerBlock'])
        # The max(1, num_warps_to_iterate) handles the zero case later
        
        threads_m_in_warp = hierarchical_info['ThreadPerWarp'][0]
        threads_n_in_warp = hierarchical_info['ThreadPerWarp'][1]

        # Ensure threads_m_in_warp and threads_n_in_warp are at least 1
        threads_m_in_warp = max(1, threads_m_in_warp)
        threads_n_in_warp = max(1, threads_n_in_warp)
        num_warps_to_iterate = max(1, num_warps_to_iterate)

        current_global_vis_thread_id = 0
        for warp_iter_idx in range(num_warps_to_iterate):
            warp_key = f"Warp{warp_iter_idx}"
            thread_blocks_viz[warp_key] = {}
            for tm_idx in range(threads_m_in_warp):
                for tn_idx in range(threads_n_in_warp):
                    thread_blocks_viz[warp_key][f"T{current_global_vis_thread_id}"] = {
                        "position": [tm_idx, tn_idx], "global_id": current_global_vis_thread_id,
                    }
                    current_global_vis_thread_id += 1
        hierarchical_info['ThreadBlocks'] = thread_blocks_viz
        
        # Final check for VectorK from tiler.py, if needed by visualizer
        # VectorK should already be calculated for multi-dimensional vectors
        if 'VectorK' not in hierarchical_info and hierarchical_info['VectorDimensions']:
            if len(hierarchical_info['VectorDimensions']) == 1:
                # Single dimension case
                hierarchical_info['VectorK'] = hierarchical_info['VectorDimensions'][0]
            else:
                # Multi-dimensional case
                combined_value = 1
                for val in hierarchical_info['VectorDimensions']:
                    combined_value *= val
                hierarchical_info['VectorK'] = combined_value

        return hierarchical_info

    # This was the previous get_lengths_for_p_indices, let's rename it for clarity
    def _get_lengths_for_p_dim_component(self, p_dim_index: int) -> typing.List[int]:
        """
        For a given P-dimension index, find all R/H component lengths it maps to.
        Example: P0 maps to H0[0] (length L0) and H1[1] (length L1) -> returns [L0, L1]
        """
        # DEBUG PRINT
        print(f"DEBUG: _get_lengths_for_p_dim_component(p_dim_index={p_dim_index})")
        print(f"DEBUG:   self.NDimPs={self.NDimPs}")
        print(f"DEBUG:   self.DstrEncode.Ps2RHssMajor={self.DstrEncode.Ps2RHssMajor}")
        print(f"DEBUG:   self.DstrEncode.Ps2RHssMinor={self.DstrEncode.Ps2RHssMinor}")
        print(f"DEBUG:   self.DstrEncode.RsLengths={self.DstrEncode.RsLengths}")
        print(f"DEBUG:   self.DstrEncode.HsLengthss={self.DstrEncode.HsLengthss}")

        dims = []
        if not (0 <= p_dim_index < self.NDimPs):
            # print(f"Warning: P-dimension index {p_dim_index} out of bounds for NDimPs={self.NDimPs}")
            print(f"DEBUG:   P-dim index {p_dim_index} out of bounds. Returning [].") # DEBUG
            return [] # Return empty if P-dim index is invalid

        # Ensure mapping arrays are not empty and p_dim_index is valid for them
        if not self.DstrEncode.Ps2RHssMajor or p_dim_index >= len(self.DstrEncode.Ps2RHssMajor):
            print(f"DEBUG:   Ps2RHssMajor empty or p_dim_index too large. Returning [].") # DEBUG
            return []
        
        current_p_major_map = self.DstrEncode.Ps2RHssMajor[p_dim_index]
        current_p_minor_map = self.DstrEncode.Ps2RHssMinor[p_dim_index]
        print(f"DEBUG:   For P{p_dim_index}: major_map={current_p_major_map}, minor_map={current_p_minor_map}") # DEBUG
        
        for idim_low_idx in range(len(current_p_major_map)):
            rh_major = current_p_major_map[idim_low_idx]
            rh_minor = current_p_minor_map[idim_low_idx]
            print(f"DEBUG:     Mapping {idim_low_idx}: rh_major={rh_major}, rh_minor={rh_minor}") # DEBUG

            length = -1
            if rh_major == 0:  # Maps to R-space
                if 0 <= rh_minor < self.NDimRs and rh_minor < len(self.DstrEncode.RsLengths):
                    length = self.DstrEncode.RsLengths[rh_minor]
                    print(f"DEBUG:       Maps to R-space. R[{rh_minor}], length={length}") # DEBUG
            elif rh_major > 0:  # Maps to H-space (1-based index for H from Ps2RHssMajor)
                h_sequence_idx = rh_major - 1 # 0-based index for HsLengthss
                if (0 <= h_sequence_idx < self.NDimX and
                        h_sequence_idx < len(self.DstrEncode.HsLengthss) and
                        0 <= rh_minor < len(self.DstrEncode.HsLengthss[h_sequence_idx])):
                    length = self.DstrEncode.HsLengthss[h_sequence_idx][rh_minor]
                    print(f"DEBUG:       Maps to H-space. H[{h_sequence_idx}][{rh_minor}], length={length}") # DEBUG
                else:
                    print(f"DEBUG:       Maps to H-space. Invalid H index: h_seq_idx={h_sequence_idx}, rh_minor={rh_minor}") # DEBUG
            
            if length > 0 : # Only append valid, positive lengths
                dims.append(length)
            else:
                print(f"DEBUG:       Length not > 0 (was {length}). Not appending.") # DEBUG
                # print(f"Warning: P-dim {p_dim_index} mapping to invalid R/H component: rh_major={rh_major}, rh_minor={rh_minor}")

        print(f"DEBUG:   _get_lengths_for_p_dim_component for P{p_dim_index} returning: {dims}") # DEBUG
        return dims

    # Additional methods from ck_tile::tile_distribution to be implemented:
    # - get_lengths() -> C++: tile_distribution.hpp:L107
    # - calculate_rs_index_from_ps_index(ps_idx) -> C++: tile_distribution.hpp:L132
    # - calculate_index(ps_idx) -> C++: tile_distribution.hpp:L163
    # - get_distributed_spans() -> C++: tile_distribution.hpp:L174
    # - get_y_indices_from_distributed_indices(d_indices) -> C++: tile_distribution.hpp:L190
    
    # Helper for get_lane_id() and get_warp_id() - these are context-dependent in C++ (GPU hardware)
    # For simulation, we might need to iterate through them or pass them as parameters.
    # For now, if calculate_index uses them, it needs a way to get them.

    # C++: tile_distribution.hpp:L294 (CalculateBottomIndex/CalculateTopIndex for Ys2DAdaptor)
    # For now, this directly calls the more generic _apply_tensor_adaptor_transformations
    # or _invert_tensor_adaptor_transformations.
    # These will be replaced by more specific logic if the adaptor format from
    # _create_adaptor_encodings_json_style is different.

    def _calculate_xs_from_ps_ys(self, ps_coords: typing.Sequence[int], ys_coords: typing.Sequence[int], adaptor: dict) -> typing.Sequence[int]:
        """
        Calculates X (bottom) coordinates from P and Y (top) coordinates using the PsYs2XsAdaptor.
        This implements the "CalculateBottomIndex" logic for the PsYs2XsAdaptor.
        """
        all_hid_coords = {} # Stores coordinates of elemental hidden dimensions (R-hidden, H-hidden)

        # Effective top dimension IDs and their names/lengths
        top_dim_ids_ordered = adaptor['TopView']['_effective_display_order_ids_']
        top_dim_id_to_name = adaptor['TopView']['TopDimensionIdToName']
        # top_dim_name_to_len = adaptor['TopView']['TopDimensionNameLengths']

        # Map input ps_coords and ys_coords to their respective hidden IDs
        current_ps_idx = 0
        current_ys_idx = 0
        for top_hid in top_dim_ids_ordered:
            dim_name = top_dim_id_to_name.get(str(top_hid)) # JSON keys are strings
            if dim_name is None and isinstance(top_hid, int): # try int key if str failed
                 dim_name = top_dim_id_to_name.get(top_hid)


            if dim_name and dim_name.startswith("P"):
                if current_ps_idx < len(ps_coords):
                    all_hid_coords[top_hid] = ps_coords[current_ps_idx]
                    current_ps_idx += 1
                else:
                    raise ValueError(f"Mismatch between ps_coords length and P-dimensions in adaptor TopView for hid {top_hid}")
            elif dim_name and dim_name.startswith("Y"):
                if current_ys_idx < len(ys_coords):
                    all_hid_coords[top_hid] = ys_coords[current_ys_idx]
                    current_ys_idx += 1
                else:
                    raise ValueError(f"Mismatch between ys_coords length and Y-dimensions in adaptor TopView for hid {top_hid}")
            # If dim_name is None or doesn't start with P/Y, it might be an issue or an implicitly handled R-dim.
            # For Ys that are R-dims, their hids are directly used.

        # Populate all_hid_coords from ys_coords based on Y->RH mappings in DstrEncode
        # Ys2RHsMajor/Minor map logical Y index to RH major/minor.
        # DstrDetail['rh_map'] maps (rh_major, rh_minor) to hidden_id and length.
        # The adaptor's TopView for Y already uses these "elemental" hidden IDs.
        for i, y_coord_val in enumerate(ys_coords):
            y_rh_major = self.DstrEncode.Ys2RHsMajor[i]
            y_rh_minor = self.DstrEncode.Ys2RHsMinor[i]
            y_hid = self.DstrDetail['rh_map'][(y_rh_major, y_rh_minor)]['id']
            all_hid_coords[y_hid] = y_coord_val
            # Also ensure Y-dims listed in TopView are covered
            # (Handled by the loop over top_dim_ids_ordered if Y hids are correctly there)


        # Invert "Merge" transforms for P-dimensions to get underlying H/R hidden coordinates
        # P-dimensions in TopView are already hidden IDs resulting from a merge.
        for p_logical_idx in range(self.NDimPs): # Iterate through logical P dimensions
            # Find the p_hid for this logical P dimension from TopView
            # This assumes P0, P1... names map to the order of ps_coords
            p_hid_to_find = -1
            p_name_to_find = f"P{p_logical_idx}"
            
            for hid_key, name_val in adaptor['TopView']['TopDimensionIdToName'].items():
                if name_val == p_name_to_find:
                    p_hid_to_find = int(hid_key) # hid_key might be str from JSON
                    break
            
            if p_hid_to_find == -1 or p_hid_to_find not in all_hid_coords:
                # This P-dim might not be in top_dim_ids_ordered or ps_coords was short
                # Or this P-dim is not part of the TopView of THIS adaptor, which would be an issue.
                # For now, assume ps_coords directly provides values for P-hids in TopView
                if p_hid_to_find != -1 and p_hid_to_find in all_hid_coords:
                     pass # Value already set from ps_coords
                elif self.NDimPs > 0 and len(ps_coords) == self.NDimPs:
                    # Fallback: try to find its merge transform by DstDimIds
                    # This is getting complex, relying on ps_coords matching the TopView P-dims is better.
                    pass
                else:
                    continue # Skip if P-dim not resolvable here


            p_coord = all_hid_coords[p_hid_to_find]

            # Find the Merge transform that produced this p_hid
            merge_transform = None
            for trans in adaptor['Transformations']:
                if trans['Name'] == 'Merge' and trans['DstDimIds'] == [p_hid_to_find]:
                    merge_transform = trans
                    break
            
            if merge_transform:
                src_hids_for_p = merge_transform['SrcDimIds']
                lengths_for_p_merge = merge_transform['MetaData']
                
                temp_p_val = p_coord
                for i in range(len(lengths_for_p_merge) - 1, -1, -1):
                    component_coord = temp_p_val % lengths_for_p_merge[i]
                    all_hid_coords[src_hids_for_p[i]] = component_coord
                    temp_p_val //= lengths_for_p_merge[i]
                if temp_p_val != 0: # Should be 0 if p_coord was valid for these lengths
                    # print(f"Warning: P-coord decomposition for P_hid {p_hid_to_find} had remainder {temp_p_val}")
                    pass


        # Now all_hid_coords should contain values for all elemental H and R hidden dimensions

        # Apply "Unmerge" transforms (forward) to compose X coordinates
        # X dimensions are in adaptor['BottomView']['BottomDimensionIdToName']
        # e.g., {0: "X0", 1: "X1"}
        num_x_dims = len(adaptor['BottomView']['BottomDimensionNameLengths'])
        xs_coords = Array(num_x_dims, 0)
        
        # Order of X dimension calculation matters if BottomDimensionIdToName uses integer keys
        # that imply order (0, 1, ... NDimX-1)
        x_dim_indices_sorted = sorted([int(k) for k in adaptor['BottomView']['BottomDimensionIdToName'].keys()])


        for x_logical_idx in x_dim_indices_sorted: # Assumes X0, X1, ...
            # Find the Unmerge transform for this X dimension
            # The 'SrcDimIds' of an Unmerge transform refers to the X dimension's logical index.
            unmerge_transform = None
            for trans in adaptor['Transformations']:
                # Match X's logical index with SrcDimIds[0] of an Unmerge transform
                if trans['Name'] == 'Unmerge' and trans['SrcDimIds'] == [x_logical_idx]:
                    unmerge_transform = trans
                    break
            
            if unmerge_transform:
                dst_hids_from_x = unmerge_transform['DstDimIds']
                lengths_for_x_unmerge = unmerge_transform['MetaData']
                
                x_val_component = 0
                current_stride = 1
                # The C++ Unmerge (Split) means:
                # X_flat = H0*L1*L2 + H1*L2 + H2
                # So, to get X_flat from components H0, H1, H2 and lengths L0, L1, L2
                # (where MetaData is [L0,L1,L2] for DstDimIds [H0_hid, H1_hid, H2_hid])
                # X_flat = 0
                # for i = 0 to N-1: X_flat = X_flat * L_i + H_i_coord

                val_for_x = 0
                for i in range(len(dst_hids_from_x)):
                    hid_component = dst_hids_from_x[i]
                    len_component = lengths_for_x_unmerge[i]
                    coord_component = all_hid_coords.get(hid_component)
                    if coord_component is None:
                        raise ValueError(f"Missing coordinate for hidden dim {hid_component} needed for X{x_logical_idx}")
                    val_for_x = val_for_x * len_component + coord_component
                xs_coords[x_logical_idx] = val_for_x
            else:
                # This could happen if an X dim is not formed by Unmerge, e.g. direct passthrough
                # For now, assume all Xs are from Unmerges from H-dims in this adaptor.
                # Or if X is R-dim (not typical for PsYs2XsAdaptor)
                if self.NDimX == 0 and self.NDimRs > 0 and x_logical_idx < self.NDimRs:
                     # If no H-dims (NDimX=0), then X-dims might be R-dims.
                     # This case is not fully handled by the current adaptor structure's transforms.
                     # The PsYs2Xs adaptor primarily handles Hs->Xs and Ps->Hs/Rs.
                     # If X = R, its coordinate should come from a Y that maps to R.
                     # Let's assume the structure implies this.
                     # If X_logical_idx corresponds to an R-dim, its value should be in all_hid_coords.
                     # R-dims have rh_major=0. rh_map[(0, x_logical_idx)]['id']
                    r_hid = self.DstrDetail['rh_map'].get((0, x_logical_idx), {}).get('id')
                    if r_hid is not None and r_hid in all_hid_coords:
                        xs_coords[x_logical_idx] = all_hid_coords[r_hid]
                    else:
                        # print(f"Warning: No Unmerge transform found for X{x_logical_idx}, and not directly an R-dim.")
                        # Default to 0 or handle as error
                        xs_coords[x_logical_idx] = 0 # Placeholder
                else:
                    # print(f"Warning: No Unmerge transform found for X{x_logical_idx}")
                    xs_coords[x_logical_idx] = 0 # Or raise error


        return xs_coords

    def _calculate_d_from_ys(self, ys_coords: typing.Sequence[int], adaptor_descriptor: dict) -> int:
        """
        Calculates the scalar D index from Y coordinates using the Ys2DDescriptor.
        This implements "CalculateBottomIndex" for the Ys2DAdaptor (merging Ys into D).
        """
        adaptor = adaptor_descriptor # CORRECTED
        
        # Ys2DAdaptor usually has one "Unmerge" transform: D_flat -> (LogicalY0, LogicalY1, ...)
        # To calculate D from Ys, we do the inverse: Merge Ys into D.
        # MetaData of the Unmerge transform gives the lengths of the logical Y dimensions.
        if not adaptor['Transformations']:
             if not ys_coords: return 0 # No Ys, D is 0
             if len(ys_coords) == 1: return ys_coords[0] # single Y, D is that Y
             raise ValueError("Ys2DAdaptor has no transformations but multiple Ys provided.")

        unmerge_transform = None
        for trans in adaptor['Transformations']:
            if trans['Name'] == 'Unmerge': # Expecting one such transform defining D's relation to Ys
                unmerge_transform = trans
                break
        
        if not unmerge_transform:
            # Fallback for simple pass-through if no explicit unmerge
            if len(ys_coords) == 1 and adaptor_descriptor['d_length'] == self.DstrEncode.detail['ys_lengths_'][0]:
                 return ys_coords[0]
            elif not ys_coords:
                 return 0
            raise ValueError("Ys2DAdaptor is missing the expected 'Unmerge' transformation.")

        ys_lengths_from_meta = unmerge_transform['MetaData']
        
        if len(ys_coords) != len(ys_lengths_from_meta):
            raise ValueError(f"Mismatch between number of Y coordinates ({len(ys_coords)}) and Y lengths in Ys2DAdaptor metadata ({len(ys_lengths_from_meta)})")

        d_scalar_idx = 0
        for i in range(len(ys_coords)):
            d_scalar_idx = d_scalar_idx * ys_lengths_from_meta[i] + ys_coords[i]
            
        return d_scalar_idx

    def _calculate_ys_from_d(self, d_scalar_idx: int, adaptor_descriptor: dict) -> typing.Sequence[int]:
        """
        Calculates Y coordinates from the scalar D index using the Ys2DDescriptor.
        This implements "CalculateTopIndex" for the Ys2DAdaptor (unmerging D into Ys).
        """
        adaptor = adaptor_descriptor # CORRECTED
        
        unmerge_transform = None
        for trans in adaptor['Transformations']:
            if trans['Name'] == 'Unmerge':
                unmerge_transform = trans
                break
        
        if not unmerge_transform:
            # Fallback for simple pass-through
            if adaptor_descriptor['d_length'] > 0 and self.NDimYs == 1: # Assuming single Y if no Unmerge
                 # Check if d_length matches the single Y_length for consistency
                 if self.DstrEncode.detail['ys_lengths_'] and adaptor_descriptor['d_length'] == self.DstrEncode.detail['ys_lengths_'][0]:
                    return [d_scalar_idx]
                 elif not self.DstrEncode.detail['ys_lengths_'] : # NDimY might be 0 if ys_lengths is empty
                    return []

            elif self.NDimYs == 0:
                return []

            raise ValueError("Ys2DAdaptor is missing the expected 'Unmerge' transformation for decomposition.")

        ys_lengths_from_meta = unmerge_transform['MetaData']
        num_ys_dims = len(ys_lengths_from_meta)
        ys_coords = Array(num_ys_dims, 0)
        
        temp_d = d_scalar_idx
        for i in range(num_ys_dims - 1, -1, -1):
            if ys_lengths_from_meta[i] <= 0: # Avoid division by zero or negative length
                raise ValueError(f"Invalid length {ys_lengths_from_meta[i]} for Y dimension {i} in Ys2DAdaptor.")
            ys_coords[i] = temp_d % ys_lengths_from_meta[i]
            temp_d //= ys_lengths_from_meta[i]
        
        if temp_d != 0:
            # This implies d_scalar_idx was out of bounds for the given Y lengths
            # print(f"Warning: d_scalar_idx {d_scalar_idx} resulted in non-zero remainder {temp_d} after Y decomposition.")
            pass # Or raise an error depending on strictness

        return ys_coords

    # C++: tile_distribution.hpp:L219
    def calculate_rs_index_from_ps_index(self, p_indices: typing.Sequence[int]) -> typing.Sequence[int]:
        """
        Calculates R-space indices from P-space indices.
        Corresponds to TileDistribution::CalculateRsIndexFromPsIndex in C++.
        Each R-coordinate is a sum over P-dimensions of (p_coord * p_to_r_derivative).
        """
        if self.NDimRs == 0:
            return []
        if len(p_indices) != self.NDimPs:
            raise ValueError(f"Length of p_indices ({len(p_indices)}) must match NDimPs ({self.NDimPs}).")

        rs_idx = Array(self.NDimRs, 0)
        ps_over_rs_derivative = self.DstrEncode.detail['ps_over_rs_derivative_']

        for r_dim_idx in range(self.NDimRs):
            accumulated_value = 0
            for p_dim_idx in range(self.NDimPs):
                # ps_over_rs_derivative_ has shape [NDimP][NDimR]
                derivative_val = ps_over_rs_derivative[p_dim_idx][r_dim_idx]
                accumulated_value += p_indices[p_dim_idx] * derivative_val
            rs_idx[r_dim_idx] = accumulated_value
        
        return rs_idx

    # C++: tile_distribution.hpp:L247
    def get_y_indices_from_p_indices(self, p_indices: typing.Sequence[int]) -> typing.Sequence[int]:
        """
        Calculates Y-space indices from P-space indices.
        Corresponds to TileDistribution::GetYsIndicesFromPsIndices.
        """
        r_indices = self.calculate_rs_index_from_ps_index(p_indices)
        detail = self.DstrEncode.detail
        ys_lengths = detail['ys_lengths_']
        y_indices = [0] * len(ys_lengths)

        # C++ logic:
        # const auto ys_to_span_major = dstr_encoder_detail_.ys_to_span_major_;
        # const auto ys_to_span_minor = dstr_encoder_detail_.ys_to_span_minor_;
        # for(index_t y_dim_idx = 0; y_dim_idx < GetNumOfYsDimensions(); ++y_dim_idx)
        # {
        #     const auto span_major_idx = ys_to_span_major[y_dim_idx];
        #     const auto span_minor_idx = ys_to_span_minor[y_dim_idx];
        #
        #     if(dstr_encoder_detail_.is_ys_from_r_span_[y_dim_idx])
        #     {
        #         // Ys from R-spans
        #         y_idx_result(y_dim_idx) = r_idx[span_minor_idx];
        #     }
        #     else
        #     {
        #         // Ys from H-spans
        #         // y_idx_result(y_dim_idx) = hs_idx[span_major_idx][span_minor_idx];
        #         // For this function, we only care about P->R->Y. H-spans are not involved via P.
        #         // If a Y is not from R, its value is not determined by P-indices directly.
        #         // The C++ version seems to only populate Ys that come from R-spans in this function.
        #         // Let's assume 0 for Ys not derived from R-spans for now, or check if this is the correct interpretation.
        #         // Based on `is_ys_from_r_span_`, if it's false, the Y doesn't come from R,
        #         // so p_indices don't determine it. How it's then set needs clarification if not 0.
        #         // The example in C++ only assigns if (is_ys_from_r_span_).
        #         y_indices[y_dim_idx] = 0; # Default for Ys not from R-spans
        #     }
        # }

        # Need 'is_ys_from_r_span_' from detail, which is not yet calculated.
        # Let's assume it exists for now or find its equivalent.
        # 'ys_to_span_major_' and 'ys_to_span_minor_' point to R/H dimensions.
        # If ys_to_span_major_[y_dim_idx] == 0, it's an R-span.
        # The minor index then refers to the specific R dimension.

        if 'ys_to_span_major_' not in detail or 'ys_to_span_minor_' not in detail or 'is_ys_from_r_span_' not in detail:
            print("WARNING: Essential detail fields missing ('ys_to_span_major_', 'ys_to_span_minor_', or 'is_ys_from_r_span_'). Cannot accurately compute Y from P.")
            return y_indices # Return zeros if essential info is missing

        for y_dim_idx in range(len(ys_lengths)):
            # span_major_idx = detail['ys_to_span_major_'][y_dim_idx] # Not directly needed if using is_ys_from_r_span_
            span_minor_idx = detail['ys_to_span_minor_'][y_dim_idx] # This is Ys2RHsMinor if Ys2RHsMajor indicates R-span
                                                                 # Or it's the minor index into the specific H-span if Ys2RHsMajor indicates H-span.
                                                                 # For R-spans, Ys2RHsMinor is the R-dim index.
            
            is_from_r_span = detail['is_ys_from_r_span_'][y_dim_idx]

            if is_from_r_span:
                # If Y is from R, its Ys2RHsMinor directly gives the R-dimension index.
                r_dim_source_idx = self.DstrEncode.Ys2RHsMinor[y_dim_idx]
                if r_dim_source_idx < len(r_indices):
                    y_indices[y_dim_idx] = r_indices[r_dim_source_idx]
                else:
                    print(f"WARNING: R-dimension source index {r_dim_source_idx} (from Ys2RHsMinor for Y{y_dim_idx}) out of bounds for r_indices (len {len(r_indices)}).")
                    y_indices[y_dim_idx] = 0
            else:
                # Ys from H-spans are not determined by P-indices in this path.
                y_indices[y_dim_idx] = 0 # Default for Ys not from R-spans

        return y_indices

    def get_distributed_spans(self) -> typing.Sequence[typing.Sequence[typing.Tuple[int, int]]]:
        """
        Returns the calculated distributed spans for Y dimensions, showing (offset, length) for each minor span.
        Corresponds to TileDistribution::GetDistributedSpans() in C++ which returns detail_.distributed_spans_lengthss_.
        """
        return self.DstrEncode.detail['distributed_spans_lengthss_']

    # --- Placeholder/TODO methods from C++ --- #
    def get_tile_shape_ys(self) -> typing.Sequence[int]:
        """
        Returns the shape of the Y-dimensions within a tile.
        Corresponds to TileDistribution::GetTileShapeYs() in C++, which effectively
        returns dstr_encoder_detail_.ys_lengths_ (as per C++ source comments).
        """
        # The C++ implementation has a static_assert for NDimY > 0.
        # If NDimY is 0, ys_lengths_ will be empty, so returning it is fine.
        if not hasattr(self.DstrEncode, 'detail') or 'ys_lengths_' not in self.DstrEncode.detail:
            # This case should ideally not happen if __init__ ran correctly.
            if self.NDimYs == 0:
                return []
            else:
                # This indicates an initialization problem or premature call.
                # print("Warning: get_tile_shape_ys called before detail.ys_lengths_ is available.")
                return [1] * self.NDimYs # Fallback, though likely incorrect.
        return self.DstrEncode.detail['ys_lengths_']

    def get_lane_id(self) -> int:
        """
        Returns the lane ID within a warp. Placeholder, always returns 0.
        Corresponds to TileDistribution::GetLaneId() in C++.
        """
        return 0

    def get_warp_id(self) -> int:
        """
        Returns the warp ID. Placeholder, always returns 0.
        Corresponds to TileDistribution::GetWarpId() in C++.
        """
        return 0


# Example usage (similar to tiler.py)
if __name__ == "__main__":
    # User's specific example:
    example_encoding_dict = {
        "RsLengths": [1],
        "HsLengthss": [
            ["M0", "M1", "M2"],      # H0
            ["K0", "K1"]             # H1
        ],
        "Ps2RHssMajor": [ # P0 maps to H0[1], P1 maps to H0[2] and H1[0]
            [1],                     # P0 Major: H0
            [1, 2]                   # P1 Major: H0, H1
        ],
        "Ps2RHssMinor": [
            [1],                     # P0 Minor: H0[1] (M1)
            [2, 0]                   # P1 Minor: H0[2] (M2), H1[0] (K0)
        ],
        "Ys2RHsMajor": [1, 2],       # Y0 -> H0, Y1 -> H1
        "Ys2RHsMinor": [0, 1],       # Y0 -> H0[0] (M0), Y1 -> H1[1] (K1)
        "PsLengths": [1,1] # Example, assuming NDimP=2 from Ps2RHssMajor
    }
    
    example_variables = {
        "M0": 4, "M1": 4, "M2": 16,
        "K0": 4, "K1": 8
    }

    print("--- Pedantic TileDistribution with User's Example ---")
    pedantic_tile_dist = TileDistributionPedantic(example_encoding_dict, example_variables)
    
    # print("\n--- Encoding Details ---")
    # pedantic_tile_dist.DstrEncode.print_encoding()

    # print("\n--- Distribution Details ---")
    # pedantic_tile_dist.print_distribution()

    print("\n--- Visualization Data (Pedantic) ---")
    viz_data_pedantic = pedantic_tile_dist.get_visualization_data()
    # print("Tile Shape:", viz_data_pedantic['tile_shape'])
    # print("Occupancy:", viz_data_pedantic['occupancy'])
    # print("Utilization:", viz_data_pedantic['utilization'])
    print("Hierarchical Structure (from get_visualization_data):")
    hier_struct = viz_data_pedantic.get('hierarchical_structure', {})
    for key, value in hier_struct.items():
        if key == 'ThreadBlocks':
            print(f"  {key}: {{...omitted for brevity...}}")
        else:
            print(f"  {key}: {value}")

    # Directly call calculate_hierarchical_tile_structure for its debug prints
    print("\n--- Direct call to calculate_hierarchical_tile_structure ---")
    direct_hier_struct = pedantic_tile_dist.calculate_hierarchical_tile_structure()
    print("Hierarchical Structure (from direct call):")
    for key, value in direct_hier_struct.items():
        if key == 'ThreadBlocks':
            print(f"  {key}: {{...omitted for brevity...}}")
        else:
            print(f"  {key}: {value}")


    # --- Specific tests for TileDistributionEncodingPedantic.detail (from before) ---
    # detail = pedantic_tile_dist.DstrEncode.detail # Using the user's example
    # print(f"\nCalculated detail.ys_lengths_: {detail['ys_lengths_']}")
    # Expected YsLengths: Y0->H0[0](M0=4), Y1->H1[1](K1=8) -> [4,8]
    # assert detail['ys_lengths_'] == [4, 8] 
    # print(f"Calculated detail.rhs_major_minor_to_ys_: (MaxMinor: {detail['max_ndim_rh_minor_']})")
    # H0 has 3 components (M0,M1,M2), H1 has 2 (K0,K1). MaxMinor should be 3.
    # R has 1 component.
    # num_rh_major_ = NDimX+1 = 2+1 = 3
    # expected_rhs_major_minor_to_ys = [[-1]*3 for _ in range(3)] # MaxMinor is 3, NDimRHMajor is 3
    # Y0 (idx 0) maps to H0[0] (rh_major=1, rh_minor=0)
    # expected_rhs_major_minor_to_ys[1][0] = 0 
    # Y1 (idx 1) maps to H1[1] (rh_major=2, rh_minor=1)
    # expected_rhs_major_minor_to_ys[2][1] = 1
    # assert detail['rhs_major_minor_to_ys_'] == expected_rhs_major_minor_to_ys
    # print("\nInitial detail structure tests passed for user example.")