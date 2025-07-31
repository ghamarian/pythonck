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
    
    def validate_snippet(self, code: str, file_path: Path, line_num: int, snippet_type: str) -> Tuple[bool, str]:
        """Validate a single code snippet.
        
        Returns:
            (success, error_message)
        """
        # Skip pseudo-code or incomplete snippets
        if 'pass' in code and '...' in code:
            return True, "Skipped: Pseudo-code snippet"
        
        if snippet_type == 'python' and ('def ' in code or 'class ' in code) and '...' in code:
            return True, "Skipped: Function/class definition template"
        
        # Create a test script with necessary imports
        test_code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Standard imports that might be needed
import numpy as np
from pytensor.buffer_view import BufferView, AddressSpaceEnum, MemoryOperationEnum, make_buffer_view
from pytensor.tensor_view import TensorView, make_tensor_view
from pytensor.tensor_descriptor import TensorDescriptor, make_tensor_descriptor
from pytensor.tensor_coordinate import TensorCoordinate, make_tensor_coordinate

# Mock functions for examples
def get_thread_id():
    return 0

def make_static_tile_distribution(encoding):
    return None

def make_tile_window(tensor_view, window_lengths, origin, tile_distribution):
    class MockTileWindow:
        def load(self):
            return MockLoadedTensor()
    return MockTileWindow()

class MockLoadedTensor:
    def get_element(self, indices):
        return 1.0

def sweep_tile(tensor, func):
    pass

# Execute the snippet
try:
    {}
    print("\\nSnippet executed successfully!")
except Exception as e:
    print(f"\\nError: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
""".format(code)
        
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
        
        for i, (code, line_num, snippet_type) in enumerate(snippets):
            self.total_snippets += 1
            print(f"\nSnippet {i+1} (line {line_num}, type: {snippet_type}):")
            print("-" * 40)
            
            # Show first few lines of the snippet
            code_preview = '\n'.join(code.split('\n')[:5])
            if len(code.split('\n')) > 5:
                code_preview += '\n...'
            print(code_preview)
            
            success, error = self.validate_snippet(code, file_path, line_num, snippet_type)
            
            if success:
                self.passed_snippets += 1
                print(f"✅ PASSED{': ' + error if error else ''}")
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
