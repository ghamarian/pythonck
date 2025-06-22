import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensor_transforms import TensorTransformParser
from tensor_transform_app import build_transformation_graph_from_pytensor


def test_edge_structure():
    """Verify that hierarchical merge has the correct edge structure."""
    print("\n" + "="*80)
    print("TEST: Edge Structure Verification")
    print("="*80)
    
    # Your original nested merge case
    descriptors = ["""
    transform_tensor_descriptor(
        input_desc,
        make_tuple(
            make_merge_transform(
                make_tuple(
                    make_pass_through_transform(number<A>{}),
                    make_merge_transform(make_tuple(number<B>{}, number<C>{}))
                )
            ),
            make_pass_through_transform(number<D>{})
        ),
        make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
        make_tuple(sequence<0>{}, sequence<1>{})
    )
    """]
    
    variables = {"A": 2, "B": 4, "C": 8, "D": 16, "input_desc": 1}
    
    # Build the graph
    graph = build_transformation_graph_from_pytensor(descriptors, variables)
    graph_source = graph.source
    
    print("Analyzing edge structure in the graph...")
    
    # Look for the key edges we expect
    edges = []
    lines = graph_source.split('\n')
    for line in lines:
        if '->' in line and 'edge' not in line:  # Skip edge attribute lines
            edges.append(line.strip())
    
    print(f"Found {len(edges)} edges:")
    for edge in edges:
        print(f"  {edge}")
    
    # Check for specific expected edges
    expected_patterns = [
        "intermediate",  # Should have intermediate nodes
        "input_d0",      # Should have input d0
        "input_d1",      # Should have input d1  
        "input_d2",      # Should have input d2
        "s1_t0_d0",      # Should have final merge node
    ]
    
    pattern_found = {}
    for pattern in expected_patterns:
        pattern_found[pattern] = pattern in graph_source
    
    print(f"\nPattern Analysis:")
    for pattern, found in pattern_found.items():
        status = "✅" if found else "❌"
        print(f"{status} {pattern}: {found}")
    
    # Check for the crucial intermediate-to-final edge
    intermediate_to_final = False
    for line in lines:
        if 'intermediate' in line and 's1_t0_d0' in line and '->' in line:
            intermediate_to_final = True
            print(f"\n🎯 FOUND intermediate-to-final edge: {line.strip()}")
            break
    
    if intermediate_to_final:
        print("✅ SUCCESS: Intermediate node properly connected to final node!")
    else:
        print("❌ ISSUE: Missing edge from intermediate to final node")
    
    print(f"\n" + "="*40)
    print("EDGE STRUCTURE SUMMARY:")
    print(f"Expected hierarchical structure:")
    print(f"  d1, d2 → intermediate → final")
    print(f"  d0 → final")
    print(f"  d3 → passthrough → out1")
    
    if intermediate_to_final and all(pattern_found.values()):
        print(f"\n🎉 All expected structure elements found!")
        return True
    else:
        print(f"\n⚠️  Some structure elements missing")
        return False


if __name__ == "__main__":
    success = test_edge_structure()
    
    if success:
        print(f"\n🎯 CONCLUSION: The edge structure is correct!")
        print(f"   The graph now shows proper hierarchical connections.")
    else:
        print(f"\n⚠️  There may still be edge connection issues.") 