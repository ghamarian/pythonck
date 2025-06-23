#!/bin/bash

echo "🔍 Testing Quarto Documentation Setup"
echo "===================================="

# Check if documentation directory exists
DOC_PATH="/projects/pythonck/documentation/_site"
if [ -d "$DOC_PATH" ]; then
    echo "✅ Documentation directory exists: $DOC_PATH"
    
    # Check permissions
    echo "📁 Directory permissions:"
    ls -la "$DOC_PATH/"
    
    # Check if index.html exists
    if [ -f "$DOC_PATH/index.html" ]; then
        echo "✅ index.html exists"
        echo "📄 index.html size: $(stat -f%z "$DOC_PATH/index.html" 2>/dev/null || stat -c%s "$DOC_PATH/index.html" 2>/dev/null) bytes"
    else
        echo "❌ index.html missing"
    fi
    
    # Check key files
    echo ""
    echo "📋 Key files check:"
    for file in "index.html" "search.json"; do
        if [ -f "$DOC_PATH/$file" ]; then
            echo "✅ $file exists"
        else
            echo "❌ $file missing"
        fi
    done
    
    # Check subdirectories
    echo ""
    echo "📂 Subdirectories:"
    ls -la "$DOC_PATH/" | grep '^d'
    
else
    echo "❌ Documentation directory not found: $DOC_PATH"
    echo "🔍 Checking if it exists elsewhere..."
    find /projects -name "_site" -type d 2>/dev/null || echo "No _site directories found"
fi

echo ""
echo "🌐 Testing documentation access:"
echo "Direct file test:"
if [ -f "$DOC_PATH/index.html" ]; then
    echo "✅ Can read index.html directly"
else
    echo "❌ Cannot read index.html directly"
fi

echo ""
echo "🔧 Nginx configuration hints:"
echo "1. Check nginx error log: sudo tail -f /var/log/nginx/error.log"
echo "2. Test nginx config: sudo nginx -t"
echo "3. Check file permissions: ls -la $DOC_PATH/"
echo "4. Test direct file access: curl -I https://ck.silobrain.com/" 