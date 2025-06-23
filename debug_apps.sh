#!/bin/bash

echo "🔍 Debugging Streamlit Apps and Nginx Configuration"
echo "=================================================="

# Check if Streamlit apps are running
echo "📊 Checking Streamlit app processes..."
ps aux | grep streamlit | grep -v grep

echo ""
echo "🌐 Testing direct app access..."

# Test each app directly
for port in 8501 8502 8503; do
    echo "Testing port $port..."
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port" | grep -q "200"; then
        echo "✅ Port $port: OK"
    else
        echo "❌ Port $port: Failed"
        echo "   Response: $(curl -s -w "%{http_code}" "http://127.0.0.1:$port" || echo "Connection failed")"
    fi
done

echo ""
echo "📝 Checking Streamlit logs..."
if [ -d "logs" ]; then
    echo "Recent log entries:"
    for app in tensor-transform tensor-visualization thread-visualization; do
        if [ -f "logs/${app}.log" ]; then
            echo "--- ${app} ---"
            tail -n 5 "logs/${app}.log"
        fi
    done
else
    echo "No logs directory found"
fi

echo ""
echo "🔧 Nginx Configuration Test:"
echo "Run these commands on your server:"
echo "1. sudo nginx -t"
echo "2. sudo systemctl status nginx"
echo "3. sudo tail -f /var/log/nginx/error.log"

echo ""
echo "🌐 Test URLs:"
echo "Direct app access:"
echo "  http://127.0.0.1:8501"
echo "  http://127.0.0.1:8502" 
echo "  http://127.0.0.1:8503"
echo ""
echo "Through nginx (replace with actual URL):"
echo "  https://ck.silobrain.com/tensor-transform"
echo "  https://ck.silobrain.com/tensor-visualization"
echo "  https://ck.silobrain.com/thread-visualization" 