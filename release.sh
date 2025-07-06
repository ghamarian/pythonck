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

# Check git status and branch
check_git_status() {
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check if there are uncommitted changes (excluding files we'll modify and temporary files)
    if ! git diff --quiet HEAD -- . ':!documentation/' ':!pyproject.toml' ':!requirements.txt' ':!.gitignore' ':!release.sh'; then
        log_warning "You have uncommitted changes. Please commit or stash them before releasing."
        log_warning "Note: Temporary files (.quarto_ipynb) are automatically ignored"
        log_warning "Note: Documentation site (_site/) will be built and committed during release"
        git status --porcelain | grep -v "\.quarto_ipynb$" | grep -v "\.md\.bak$"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Release cancelled"
            exit 0
        fi
    fi
    
    # Show current branch
    CURRENT_BRANCH=$(git branch --show-current)
    log_info "Current branch: $CURRENT_BRANCH"
    
    # Check if we're on the main/master branch
    if [[ "$CURRENT_BRANCH" != "main" && "$CURRENT_BRANCH" != "master" ]]; then
        log_warning "You're not on the main/master branch"
        read -p "Continue with release from '$CURRENT_BRANCH'? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Release cancelled"
            exit 0
        fi
    fi
}

# Check GitHub CLI availability
check_github_cli() {
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is not installed. Please install it first:"
        echo "  https://github.com/cli/cli#installation"
        exit 1
    fi
    
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI is not authenticated. Please run 'gh auth login' first."
        exit 1
    fi
    
    log_info "GitHub CLI is available and authenticated"
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
    
    # Debug: Show Python version and path
    log_info "Python version: $(python --version)"
    log_info "Python path: $(which python)"
    
    # Ensure build module is installed
    if ! python -c "import build" 2>/dev/null; then
        log_warning "Build module not found. Installing..."
        pip install 'build>=0.7.0'
        
        # Verify installation
        if ! python -c "import build" 2>/dev/null; then
            log_error "Failed to install build module"
            exit 1
        fi
        log_success "Build module installed successfully"
    else
        log_info "Build module is already available"
    fi
    
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

# Build documentation site
build_documentation_site() {
    log_info "Building documentation site..."
    
    # Check if quarto is available
    if ! command -v quarto &> /dev/null; then
        log_warning "Quarto not found. Skipping documentation build."
        log_info "Site will be built using existing _site directory"
        return
    fi
    
    # Build the documentation
    cd documentation
    quarto render
    cd ..
    
    log_success "Documentation site built successfully"
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
    
    # Show what files have been modified
    log_info "Files modified during release:"
    git status --porcelain
    
    # Add all release-related files
    log_info "Adding release-related files..."
    
    # Add the new wheel file
    git add "documentation/${PACKAGE_NAME}-${VERSION}-py3-none-any.whl"
    
    # Add only modified tracked files in documentation (safe)
    git add -u documentation/
    
    # Add documentation site (built output for deployment)
    git add documentation/_site/
    
    # Add configuration files that might have been updated (only if they exist and are tracked)
    if git ls-files --error-unmatch pyproject.toml >/dev/null 2>&1; then
        git add pyproject.toml
    fi
    if git ls-files --error-unmatch requirements.txt >/dev/null 2>&1; then
        git add requirements.txt
    fi
    
    # Add release script itself if it was modified
    if git ls-files --error-unmatch release.sh >/dev/null 2>&1; then
        git add release.sh
    fi
    
    # Add .gitignore if it was modified
    if git ls-files --error-unmatch .gitignore >/dev/null 2>&1; then
        git add .gitignore
    fi
    
    # Remove any old wheel files that might conflict
    log_info "Cleaning up old wheel files..."
    find documentation/ -name "${PACKAGE_NAME}-*.whl" ! -name "${PACKAGE_NAME}-${VERSION}-py3-none-any.whl" -delete
    
    # Show what will be committed
    log_info "Files staged for commit:"
    git diff --cached --name-only
    
    # Check if there are changes to commit
    if git diff --staged --quiet; then
        log_warning "No changes to commit"
        return
    fi
    
    # Show a summary of what will be committed
    echo
    log_info "Summary of changes to be committed:"
    git diff --cached --stat
    echo
    
    # Ask for confirmation
    read -p "Proceed with commit and push? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Commit cancelled by user"
        log_info "You can manually commit these changes later with:"
        echo "  git commit -m 'Release v$VERSION'"
        return
    fi
    
    # Commit
    git commit -m "Release v$VERSION: Update documentation and wheel file

- Built new wheel package for v$VERSION
- Updated documentation URLs to use new version
- Built documentation site (_site/) for deployment
- Updated GitHub release with new assets
- Updated project dependencies (build tools)
- Cleaned up old wheel files"
    
    # Detect default branch
    DEFAULT_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "master")
    
    # Push
    git push origin "$DEFAULT_BRANCH"
    log_success "Changes committed and pushed to $DEFAULT_BRANCH"
}

# Create and push git tag
create_git_tag() {
    log_info "Creating git tag v$VERSION..."
    
    # Check if tag already exists
    if git tag -l "v$VERSION" | grep -q "v$VERSION"; then
        log_warning "Tag v$VERSION already exists locally"
        git tag -d "v$VERSION"
        log_info "Deleted existing local tag v$VERSION"
    fi
    
    # Create annotated tag
    git tag -a "v$VERSION" -m "Release v$VERSION

- Built wheel package for v$VERSION
- Updated documentation and dependencies
- GitHub release created with assets"
    
    # Push tag
    git push origin "v$VERSION" --force
    log_success "Git tag v$VERSION created and pushed"
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
    log_info "Script version: Enhanced release script with git safety checks"
    
    check_directory
    check_git_status
    check_github_cli
    get_version
    clean_build
    build_wheel
    update_documentation_wheel
    build_documentation_site
    create_github_release
    update_documentation_urls
    commit_and_push
    create_git_tag
    
    log_success "Release process completed successfully!"
    log_info "Release v$VERSION is now available at:"
    echo "  https://github.com/$GITHUB_REPO/releases/tag/v$VERSION"
    
    # Detect default branch for documentation URL
    DEFAULT_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "master")
    log_info "Documentation wheel URL:"
    echo "  https://raw.githubusercontent.com/$GITHUB_REPO/$DEFAULT_BRANCH/documentation/${PACKAGE_NAME}-${VERSION}-py3-none-any.whl"
    log_info "Documentation site ready for deployment from:"
    echo "  documentation/_site/"
    
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
    echo "  2. Updates documentation with new wheel"
    echo "  3. Builds documentation site (_site/) for deployment"
    echo "  4. Creates/updates GitHub release"
    echo "  5. Commits and pushes changes"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --version  Show current version"
    echo ""
    echo "Prerequisites:"
    echo "  - GitHub CLI (gh) installed and authenticated"
    echo "  - Python build tools (will be installed automatically if missing)"
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