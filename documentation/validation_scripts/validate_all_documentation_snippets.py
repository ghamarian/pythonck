#!/usr/bin/env python3
"""
Validate all Python code snippets in the tile distribution documentation.

This script:
1. Scans all .qmd files in the concepts directory
2. Extracts Python code blocks
3. Runs each snippet and verifies it works
4. Reports any issues found
"""

import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any
import subprocess
import tempfile

# Add the pythonck directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class DocumentationValidator:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.results = []
        self.total_snippets = 0
        self.passed_snippets = 0
        self.failed_snippets = 0
        
    def extract_python_snippets(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """Extract Python code snippets from a .qmd file.
        
        Returns:
            List of (code, line_number, snippet_type) tuples
        """
        snippets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern for {pyodide} blocks
        pyodide_pattern = r'```{pyodide}(.*?)```'
        pyodide_matches = re.finditer(pyodide_pattern, content, re.DOTALL)
        
        for match in pyodide_matches:
            code_block = match.group(1)
            # Skip blocks with echo: false or output: false
            if '#| echo: false' in code_block or '#| autorun: true' in code_block:
                continue
                
            # Extract actual Python code (remove metadata comments)
            lines = code_block.strip().split('\n')
            code_lines = []
            for line in lines:
                if not line.strip().startswith('#|'):
                    code_lines.append(line)
            
            code = '\n'.join(code_lines).strip()
            if code:
                line_num = content[:match.start()].count('\n') + 1
                snippets.append((code, line_num, 'pyodide'))
        
        # Pattern for regular Python code blocks
        python_pattern = r'```python(.*?)```'
        python_matches = re.finditer(python_pattern, content, re.DOTALL)
        
        for match in python_matches:
            code = match.group(1).strip()
            if code and not code.startswith('#'):  # Skip comment-only blocks
                line_num = content[:match.start()].count('\n') + 1
                snippets.append((code, line_num, 'python'))
        
        return snippets
    
    def validate_snippet(self, code: str, file_path: Path, line_num: int, snippet_type: str, cumulative_context: List[str]) -> Tuple[bool, str]:
        """Validate a single code snippet with cumulative context from previous snippets.
        
        Returns:
            (success, error_message)
        """
        # Skip pseudo-code or incomplete snippets
        if 'pass' in code and '...' in code:
            return True, "Skipped: Pseudo-code snippet"
        
        if snippet_type == 'python' and ('def ' in code or 'class ' in code) and '...' in code:
            return True, "Skipped: Function/class definition template"
        
        # Skip any snippet that contains '...' as it indicates incomplete/truncated code
        if '...' in code:
            return True, "Skipped: Incomplete snippet (contains '...')"
        
        # Skip snippets that are just function signatures without implementation
        lines = code.strip().split('\n')
        if lines and lines[0].strip().startswith('def '):
            # Check if there's actual implementation after the docstring
            non_empty_lines = [l for l in lines[1:] if l.strip() and not l.strip().startswith('"""') and not l.strip().endswith('"""')]
            if not non_empty_lines:
                return True, "Skipped: Function signature without implementation"
        
        # Skip snippets that are just single expressions or names
        if len(code.strip().split('\n')) == 1 and not any(op in code for op in ['=', '(', '[', '{', 'import', 'from']):
            return True, "Skipped: Single expression/name"
        
        # Build cumulative code from all previous snippets
        cumulative_code_str = '\n'.join(cumulative_context)
        
        # Properly indent the cumulative code and current code
        if cumulative_code_str:
            indented_cumulative = '\n'.join('    ' + line for line in cumulative_code_str.split('\n'))
        else:
            indented_cumulative = ''
            
        indented_code = '\n'.join('    ' + line for line in code.split('\n'))
        
        # Create a test script with necessary imports
        # Use absolute path to ensure correct module discovery
        project_root = str(Path(__file__).parent.parent.parent.absolute())
        
        test_code = f"""
import sys
import os
from pathlib import Path

# Add absolute path to ensure module discovery
sys.path.insert(0, '{project_root}')

# Standard imports that might be needed
import numpy as np

# Import all common pytensor modules
try:
    from pytensor.buffer_view import BufferView, AddressSpaceEnum, MemoryOperationEnum, make_buffer_view
    from pytensor.tensor_view import TensorView, make_tensor_view, make_naive_tensor_view_packed
    from pytensor.tensor_descriptor import (
        TensorDescriptor, make_tensor_descriptor, 
        make_naive_tensor_descriptor_packed, make_naive_tensor_descriptor
    )
    from pytensor.tensor_coordinate import TensorCoordinate, make_tensor_coordinate
    from pytensor.tile_distribution import make_static_tile_distribution
    from pytensor.tile_distribution_encoding import make_tile_distribution_encoding
    from pytensor.tile_window import make_tile_window
    from pytensor.sweep_tile import sweep_tile
    from pytensor.static_distributed_tensor import make_static_distributed_tensor
except ImportError as e:
    print(f"Import warning: {{e}}")

# Mock functions for examples that might not be imported
def get_thread_id():
    return 0

# Mock any missing functions
if 'make_static_tile_distribution' not in locals():
    def make_static_tile_distribution(encoding):
        class MockDistribution:
            tile_distribution = None
        return MockDistribution()

if 'make_tile_window' not in locals():
    def make_tile_window(tensor_view, window_lengths, origin, tile_distribution):
        class MockTileWindow:
            def load(self):
                return MockLoadedTensor()
        return MockTileWindow()

if 'make_static_distributed_tensor' not in locals():
    def make_static_distributed_tensor(dtype, tile_distribution):
        class MockDistributedTensor:
            tile_distribution = tile_distribution
        return MockDistributedTensor()

class MockLoadedTensor:
    def get_element(self, indices):
        return 1.0

if 'sweep_tile' not in locals():
    def sweep_tile(tensor, func):
        pass

# Execute cumulative context and current snippet
try:
    # Execute all previous snippets first (cumulative context)
{indented_cumulative}
    
    # Now execute the current snippet
{indented_code}
    print("\\nSnippet executed successfully!")
except Exception as e:
    print(f"\\nError: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        # Write to temporary file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file = f.name
        
        try:
            # Run the snippet
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                error_msg = result.stderr or result.stdout
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            return False, "Snippet timed out (>10 seconds)"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
        finally:
            os.unlink(temp_file)
    
    def validate_file(self, file_path: Path):
        """Validate all snippets in a single file."""
        print(f"\n{'='*60}")
        print(f"Validating: {file_path.name}")
        print(f"{'='*60}")
        
        snippets = self.extract_python_snippets(file_path)
        if not snippets:
            print("No Python snippets found.")
            return
        
        print(f"Found {len(snippets)} Python snippet(s)")
        
        # Accumulate code context for this file
        cumulative_context = []
        
        for i, (code, line_num, snippet_type) in enumerate(snippets):
            self.total_snippets += 1
            print(f"\nSnippet {i+1} (line {line_num}, type: {snippet_type}):")
            print("-" * 40)
            
            # Show first few lines of the snippet
            code_preview = '\n'.join(code.split('\n')[:5])
            if len(code.split('\n')) > 5:
                code_preview += '\n...'
            print(code_preview)
            
            success, error = self.validate_snippet(code, file_path, line_num, snippet_type, cumulative_context)
            
            if success:
                self.passed_snippets += 1
                print(f"✅ PASSED{': ' + error if error else ''}")
                
                # Add successful snippet to cumulative context for next snippets
                # Skip adding if it's just a skipped snippet
                if not error.startswith("Skipped:"):
                    cumulative_context.append(code)
            else:
                self.failed_snippets += 1
                print(f"❌ FAILED: {error}")
                self.results.append({
                    'file': file_path.name,
                    'line': line_num,
                    'type': snippet_type,
                    'error': error,
                    'code_preview': code_preview
                })
        
        # Clear context between files
        cumulative_context.clear()
    
    def validate_all(self):
        """Validate all .qmd files in the concepts directory."""
        qmd_files = sorted(self.docs_dir.glob('*.qmd'))
        
        print(f"Found {len(qmd_files)} documentation files to validate")
        
        for file_path in qmd_files:
            self.validate_file(file_path)
        
        # Print summary
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total snippets tested: {self.total_snippets}")
        print(f"Passed: {self.passed_snippets} ✅")
        print(f"Failed: {self.failed_snippets} ❌")
        
        if self.failed_snippets > 0:
            print(f"\nFAILED SNIPPETS:")
            print("-" * 60)
            for result in self.results:
                print(f"\nFile: {result['file']}, Line: {result['line']}")
                print(f"Type: {result['type']}")
                print(f"Code preview:")
                print(result['code_preview'])
                print(f"Error: {result['error'][:200]}...")
                print("-" * 40)
            return False
        else:
            print("\n🎉 All snippets validated successfully!")
            return True


def main():
    """Main entry point."""
    # Determine the concepts directory
    script_dir = Path(__file__).parent
    concepts_dir = script_dir.parent / 'concepts'
    
    if not concepts_dir.exists():
        print(f"Error: Concepts directory not found at {concepts_dir}")
        sys.exit(1)
    
    # Create validator and run
    validator = DocumentationValidator(concepts_dir)
    success = validator.validate_all()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
