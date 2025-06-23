#!/bin/bash

# PythonCK Release Script
# This script automates the entire release process:
# 1. Builds the wheel package
# 2. Creates/updates GitHub release
# 3. Updates documentation with new wheel
# 4. Commits and pushes changes

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="pythonck"
GITHUB_REPO="ghamarian/pythonck"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
check_directory() {
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found. Please run this script from the project root."
        exit 1
    fi
}

# Get current version from pyproject.toml
get_version() {
    VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
    if [[ -z "$VERSION" ]]; then
        log_error "Could not extract version from pyproject.toml"
        exit 1
    fi
    log_info "Current version: $VERSION"
}

# Clean previous builds
clean_build() {
    log_info "Cleaning previous builds..."
    rm -rf build/ dist/ *.egg-info/
    log_success "Build directories cleaned"
}

# Build the wheel
build_wheel() {
    log_info "Building wheel package..."
    python -m build --wheel --outdir dist
    
    WHEEL_FILE="dist/${PACKAGE_NAME}-${VERSION}-py3-none-any.whl"
    if [[ ! -f "$WHEEL_FILE" ]]; then
        log_error "Wheel file not created: $WHEEL_FILE"
        exit 1
    fi
    log_success "Wheel built: $WHEEL_FILE"
}

# Update wheel in documentation folder
update_documentation_wheel() {
    log_info "Updating wheel file in documentation folder..."
    cp "$WHEEL_FILE" "documentation/"
    log_success "Wheel copied to documentation folder"
}

# Create or update GitHub release
create_github_release() {
    log_info "Creating/updating GitHub release v$VERSION..."
    
    # Check if release exists
    if gh release view "v$VERSION" >/dev/null 2>&1; then
        log_warning "Release v$VERSION already exists. Updating..."
        gh release upload "v$VERSION" "$WHEEL_FILE" --clobber
    else
        log_info "Creating new release v$VERSION..."
        gh release create "v$VERSION" "$WHEEL_FILE" \
            --title "PythonCK v$VERSION" \
            --notes "Release of PythonCK v$VERSION with pytensor, tensor_transforms, and tile_distribution modules.

## Installation

### For Pyodide/Browser environments:
\`\`\`python
import micropip
await micropip.install('https://github.com/$GITHUB_REPO/releases/download/v$VERSION/${PACKAGE_NAME}-${VERSION}-py3-none-any.whl')
\`\`\`

### For regular Python environments:
\`\`\`bash
pip install https://github.com/$GITHUB_REPO/releases/download/v$VERSION/${PACKAGE_NAME}-${VERSION}-py3-none-any.whl
\`\`\`

## What's Included
- **pytensor**: Core tensor operations and coordinate handling
- **tensor_transforms**: Tensor transformation analysis tools  
- **tile_distribution**: Tile distribution visualization and analysis

See the [documentation](https://$GITHUB_REPO) for usage examples."
    fi
    log_success "GitHub release updated"
}

# Update documentation URLs to use the new version
update_documentation_urls() {
    log_info "Updating documentation URLs..."
    
    # Update index.qmd
    sed -i.bak "s|${PACKAGE_NAME}-[0-9]\+\.[0-9]\+\.[0-9]\+-py3-none-any\.whl|${PACKAGE_NAME}-${VERSION}-py3-none-any.whl|g" documentation/index.qmd
    
    # Update concept files
    sed -i.bak "s|${PACKAGE_NAME}-[0-9]\+\.[0-9]\+\.[0-9]\+-py3-none-any\.whl|${PACKAGE_NAME}-${VERSION}-py3-none-any.whl|g" documentation/concepts/tensor-coordinate.qmd
    sed -i.bak "s|${PACKAGE_NAME}-[0-9]\+\.[0-9]\+\.[0-9]\+-py3-none-any\.whl|${PACKAGE_NAME}-${VERSION}-py3-none-any.whl|g" documentation/concepts/buffer-view.qmd
    
    # Remove backup files
    rm -f documentation/*.bak documentation/concepts/*.bak
    
    log_success "Documentation URLs updated"
}

# Commit and push changes
commit_and_push() {
    log_info "Committing changes..."
    
    # Add files
    git add "documentation/${PACKAGE_NAME}-${VERSION}-py3-none-any.whl"
    git add documentation/index.qmd
    git add documentation/concepts/tensor-coordinate.qmd
    git add documentation/concepts/buffer-view.qmd
    
    # Check if there are changes to commit
    if git diff --staged --quiet; then
        log_warning "No changes to commit"
        return
    fi
    
    # Commit
    git commit -m "Release v$VERSION: Update documentation and wheel file

- Built new wheel package for v$VERSION
- Updated documentation URLs to use new version
- Updated GitHub release with new assets"
    
    # Push
    git push origin master
    log_success "Changes committed and pushed"
}

# Test the documentation locally
test_documentation() {
    log_info "Testing documentation locally..."
    log_info "You can test the documentation by running:"
    echo "  cd documentation && quarto preview"
    log_info "Press Ctrl+C to stop the preview when done testing"
}

# Main execution
main() {
    log_info "Starting PythonCK release process..."
    
    check_directory
    get_version
    clean_build
    build_wheel
    update_documentation_wheel
    create_github_release
    update_documentation_urls
    commit_and_push
    
    log_success "Release process completed successfully!"
    log_info "Release v$VERSION is now available at:"
    echo "  https://github.com/$GITHUB_REPO/releases/tag/v$VERSION"
    log_info "Documentation wheel URL:"
    echo "  https://raw.githubusercontent.com/$GITHUB_REPO/master/documentation/${PACKAGE_NAME}-${VERSION}-py3-none-any.whl"
    
    test_documentation
}

# Help function
show_help() {
    echo "PythonCK Release Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script automates the complete release process:"
    echo "  1. Builds the wheel package"
    echo "  2. Creates/updates GitHub release"
    echo "  3. Updates documentation with new wheel"
    echo "  4. Commits and pushes changes"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --version  Show current version"
    echo ""
    echo "Prerequisites:"
    echo "  - GitHub CLI (gh) installed and authenticated"
    echo "  - Python build tools installed (pip install build)"
    echo "  - Git repository with remote origin set"
    echo ""
    echo "Examples:"
    echo "  $0                 # Run full release process"
    echo "  $0 --version       # Show current version"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -v|--version)
        check_directory
        get_version
        echo "$VERSION"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac 