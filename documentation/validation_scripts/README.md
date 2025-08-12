# Validation Scripts for Composable Kernels Documentation

Hey there! Welcome to the validation scripts for our Composable Kernels documentation. 

## What's This All About?

You know how frustrating it is when you're following a tutorial and the code doesn't work? We've all been there! That's exactly why these scripts exist. Every single code example in our documentation gets tested here first, so when you copy-paste code from the docs, it'll work perfectly.

## How We Do It

We follow a simple but rock-solid process:

1. **Write standalone scripts** - Each concept gets its own working example
2. **Test everything** - If it doesn't run, it doesn't go in the docs
3. **Extract the good stuff** - Only working code makes it to the `.qmd` files
4. **Test again** - We make sure the extracted snippets work in isolation too

## Directory Structure

Here's how we've organized everything to match your learning journey:

```
tile_distribution_documentation/validation_scripts/
├── part1_foundation/          ✅ Memory → Tensors (the basics)
├── part2_transforms/          ✅ Coordinate magic (the engine)
├── part3_distribution_api/    ✅ High-level APIs (what you'll use)
├── part4_coordinate_systems/  🚧 P, Y, X, R, D spaces (coordinate mapping)
├── part5_internals/          🚧 Under the hood (encoding, static tensors)
├── part6_thread_mapping/     🚧 Hardware threads coordination
├── part7_advanced_topics/    🚧 Performance, debugging, profiling
├── common/                   ✅ Shared utilities
└── code_examples/            ✅ Common helpers
```

## Running the Scripts

### Quick Start
```bash
# Test a single concept
cd documentation/validation_scripts/part1_foundation/
python 01_buffer_basics.py

# Test an entire part
python test_part1.py

# Test everything
cd ../integration_tests/
python test_integration.py
```

### The Full Workflow
```bash
# Step 1: Test individual scripts in order
cd documentation/validation_scripts/part1_foundation/
python 01_buffer_basics.py
python 02_tensor_view.py

# Step 2: Run part-specific validation
python test_part1.py

# Step 3: Move to next part
cd ../part2_transforms/
python 01_individual_transforms.py
# ... and so on

# Step 4: Run integration tests
cd ../integration_tests/
python test_integration.py
```

## What Makes a Good Validation Script?

Each script should:
- **Run independently** - No complex setup required
- **Show what it does** - Clear print statements explaining each step
- **Handle errors gracefully** - Helpful messages if something goes wrong
- **Be educational** - Comments that teach, not just describe
- **Match the docs** - Code that will appear in the `.qmd` files

## Dependencies

We keep it simple - only use what's already in the main project:
- numpy
- The pytensor package (our main library)
- Standard library modules

## Contributing

Found a bug? Want to add an example? Here's how:

1. **Write your script** following the naming convention
2. **Test it thoroughly** - make sure it runs clean
3. **Add it to the appropriate test file**
4. **Update the documentation** with the working code

## Need Help?

If you're stuck, check out:
- The main project README
- The individual script comments
- The integration tests for complete examples

Remember: if the code doesn't work here, it won't work in the docs either. That's the whole point!

Happy coding! 🚀 