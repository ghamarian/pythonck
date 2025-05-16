# Tile Distribution Encoding Indexing Explanation

## Overview of the Example Encoding

```
tile_distribution_encoding<
    sequence<1>,                            // 0 R
    tuple<sequence<Nr_y, Nr_p, Nw>,         // H
          sequence<Kr_y, Kr_p, Kw, Kv>>,    // H
    tuple<sequence<1, 2>,                   // p major (Maps to H0 and H1)
          sequence<2, 1>>,                  // p minor (Maps to H1 and H0)
    tuple<sequence<1, 1>,                   // p minor (index within H)
          sequence<2, 2>>,                  // p minor (index within H)
    sequence<1, 2, 2>,                      // Y major (Maps to H0, H1, H1)
    sequence<0, 0, 3>>{}                    // y minor (index within H)
```

With variable settings:
```
Nr_y = 4, Nr_p = 4, Nw = 8, Kr_y = 4, Kr_p = 8, Kw = 8, Kv = 4
```

## The Three-Level Hierarchy

The tile distribution encoding uses a three-level hierarchy:

1. **Top Level**: Contains P dimensions and Y dimensions
   - P dimensions represent thread mapping
   - Y dimensions represent spatial output mapping

2. **Middle Level**: Hidden values from sequences (R and H)
   - These store the actual dimension sizes

3. **Bottom Level**: R and H sequences
   - R is the root sequence (typically 1 value)
   - H contains the actual computation dimensions

## Indexing Explained

### Major and Minor Indices

- **Major indices** determine which sequence to use:
  - 0 = R sequence
  - 1 = H0 sequence
  - 2 = H1 sequence
  - etc.

- **Minor indices** determine which position within the sequence:
  - 0 = first element
  - 1 = second element
  - etc.

### For this specific encoding:

1. **R Sequence**: `sequence<1>` 
   - R[0] = 1

2. **H Sequences**:
   - H0: `sequence<Nr_y, Nr_p, Nw>` = [4, 4, 8]
   - H1: `sequence<Kr_y, Kr_p, Kw, Kv>` = [4, 8, 8, 4]

3. **P Mapping**:
   - P0 uses major indices [1, 2] (H0, H1) and minor indices [1, 1]:
     - P0[0] maps to H0[1] = 4 (for H0, element 1)
     - P0[1] maps to H1[1] = 8 (for H1, element 1)
   
   - P1 uses major indices [2, 1] (H1, H0) and minor indices [2, 2]:
     - P1[0] maps to H1[2] = 8 (for H1, element 2)
     - P1[1] maps to H0[2] = 8 (for H0, element 2)

4. **Y Mapping**:
   - Y uses major indices [1, 2, 2] (H0, H1, H1) and minor indices [0, 0, 3]:
     - Y0 maps to H0[0] = 4 (for H0, element 0)
     - Y1 maps to H1[0] = 4 (for H1, element 0)
     - Y2 maps to H1[3] = 4 (for H1, element 3)

## Practical Meaning

In practical terms:

- **P dimensions** control how threads are distributed across the computational grid.
  For this encoding, P0 uses values 4 and 8 for dimensions, while P1 uses 8 and 8.

- **Y dimensions** define the spatial mapping of the output tensors.
  Here, Y0=4, Y1=4, and Y2=4 are the sizes of different output dimensions.

The JSON output representation provides a machine-readable version of this data, but the diagram creates a visual representation that makes the relationships clearer.

## How to Interpret the Diagram

In the visualization diagram:
1. Top boxes show P and Y dimensions
2. Middle boxes show the hidden values (4, 4, 8, 4)
3. Bottom boxes show the R and H dimensions
4. Arrows represent the mapping relationships

This hierarchical structure allows the tile_distribution_encoding to compactly represent complex thread mapping and data layout patterns using its 3-tier indexing system. 