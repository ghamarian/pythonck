import pytest
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensor_transforms import TensorTransformParser
from tensor_transform_app import build_transformation_graph_from_pytensor


def test_hierarchical_merge_verification():
    """Verify that hierarchical merge produces two separate merge operations."""
    print("\n" + "="*80)
    print("TEST: Hierarchical Merge Verification")
    print("="*80)
    
    # Test the exact case from the user's query
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
    
    print("Graph Analysis:")
    print(f"Total graph length: {len(graph_source)} characters")
    
    # Count different node types
    merge_count = graph_source.count('Merge')
    passthrough_count = graph_source.count('PassThrough')
    edge_count = graph_source.count('->')
    
    # Look for intermediate nodes
    intermediate_count = graph_source.count('intermediate')
    
    print(f"Merge labels found: {merge_count}")
    print(f"PassThrough labels found: {passthrough_count}")
    print(f"Intermediate nodes: {intermediate_count}")
    print(f"Total edges: {edge_count}")
    
    # Check for the key formulas
    has_bc_merge = "8*d1 + d2" in graph_source
    has_final_merge = "32*d0 + 8*d1 + d2" in graph_source
    
    print(f"\nFormula Analysis:")
    print(f"Has B+C intermediate merge (8*d1 + d2): {has_bc_merge}")
    print(f"Has final hierarchical merge (32*d0 + 8*d1 + d2): {has_final_merge}")
    
    # Verify the hierarchical structure
    print(f"\n" + "="*40)
    print("VERIFICATION RESULTS:")
    
    success_criteria = [
        (intermediate_count >= 1, "✅ Has intermediate nodes" if intermediate_count >= 1 else "❌ No intermediate nodes"),
        (has_bc_merge, "✅ B+C merge detected" if has_bc_merge else "❌ B+C merge missing"),
        (has_final_merge, "✅ Hierarchical final merge" if has_final_merge else "❌ Final merge incorrect"),
        (merge_count >= 4, f"✅ Multiple merge operations ({merge_count})" if merge_count >= 4 else f"❌ Too few merges ({merge_count})"),
    ]
    
    all_passed = True
    for passed, message in success_criteria:
        print(message)
        if not passed:
            all_passed = False
    
    print(f"\n" + "="*40)
    if all_passed:
        print("🎉 SUCCESS: Hierarchical merge is working correctly!")
        print("   The graph now shows two separate merge operations:")
        print("   1. B + C → intermediate (8*d1 + d2)")
        print("   2. A + intermediate → final (32*d0 + 8*d1 + d2)")
    else:
        print("❌ ISSUES: Some verification criteria failed")
    
    return all_passed


def test_compare_flat_vs_hierarchical():
    """Compare flat merge vs hierarchical merge to show the difference."""
    print("\n" + "="*80)
    print("TEST: Compare Flat vs Hierarchical Merge")
    print("="*80)
    
    # Flat merge: explicitly merge A, B, C in one operation
    flat_desc = ["""
    transform_tensor_descriptor(
        input_desc,
        make_tuple(
            make_merge_transform(make_tuple(number<A>{}, number<B>{}, number<C>{}))
        ),
        make_tuple(sequence<0, 1, 2>{}),
        make_tuple(sequence<0>{})
    )
    """]
    
    # Hierarchical merge: A + (merge of B, C)
    hierarchical_desc = ["""
    transform_tensor_descriptor(
        input_desc,
        make_tuple(
            make_merge_transform(
                make_tuple(
                    number<A>{},
                    make_merge_transform(make_tuple(number<B>{}, number<C>{}))
                )
            )
        ),
        make_tuple(sequence<0, 1, 2>{}),
        make_tuple(sequence<0>{})
    )
    """]
    
    variables = {"A": 2, "B": 4, "C": 8, "input_desc": 1}
    
    # Build both graphs
    flat_graph = build_transformation_graph_from_pytensor(flat_desc, variables)
    hierarchical_graph = build_transformation_graph_from_pytensor(hierarchical_desc, variables)
    
    # Analyze differences
    flat_source = flat_graph.source
    hierarchical_source = hierarchical_graph.source
    
    flat_merge_count = flat_source.count('Merge')
    hierarchical_merge_count = hierarchical_source.count('Merge')
    
    flat_intermediate_count = flat_source.count('intermediate')
    hierarchical_intermediate_count = hierarchical_source.count('intermediate')
    
    print("COMPARISON RESULTS:")
    print(f"Flat merge operations: {flat_merge_count}")
    print(f"Hierarchical merge operations: {hierarchical_merge_count}")
    print(f"Flat intermediate nodes: {flat_intermediate_count}")
    print(f"Hierarchical intermediate nodes: {hierarchical_intermediate_count}")
    
    # Check formulas
    flat_has_simple = "2*d0 + 4*d1 + d2" in flat_source or "d0 + d1 + d2" in flat_source
    hierarchical_has_intermediate = "8*d1 + d2" in hierarchical_source
    
    print(f"\nFlat formula style: {flat_has_simple}")
    print(f"Hierarchical has intermediate: {hierarchical_has_intermediate}")
    
    if hierarchical_merge_count > flat_merge_count and hierarchical_intermediate_count > 0:
        print("\n✅ SUCCESS: Hierarchical merge produces more complex graph structure!")
    else:
        print("\n❌ ISSUE: Hierarchical merge not sufficiently different from flat")


if __name__ == "__main__":
    success1 = test_hierarchical_merge_verification()
    test_compare_flat_vs_hierarchical()
    
    if success1:
        print(f"\n🎯 CONCLUSION: The nested merge parsing issue has been FIXED!")
        print(f"   Nested structures now properly create hierarchical graphs.")
    else:
        print(f"\n⚠️  There may still be some issues to address.") 