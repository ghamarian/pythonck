#!/bin/bash

# Deploy Streamlit Apps Script - Fixed Version
# This script starts all three Streamlit apps on different ports without baseUrlPath

set -e  # Exit on any error

echo "🚀 Starting Streamlit Apps Deployment (Fixed)..."

# Kill any existing streamlit processes
echo "🔄 Cleaning up existing processes..."
pkill -f "streamlit run" || true

# Wait a moment for processes to terminate
sleep 2

# Function to start a Streamlit app
start_app() {
    local app_file=$1
    local port=$2
    local app_name=$3
    local base_url_path=$4
    
    echo "🎯 Starting $app_name on port $port with base URL path $base_url_path..."
    
    # Start the app in the background WITH baseUrlPath
    nohup streamlit run "$app_file" \
        --server.port=$port \
        --server.address=0.0.0.0 \
        --server.baseUrlPath="$base_url_path" \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --server.enableWebsocketCompression=false \
        --server.allowRunOnSave=true \
        --server.headless=true \
        > "logs/${app_name}.log" 2>&1 &
    
    # Get the PID
    local pid=$!
    echo "$pid" > "logs/${app_name}.pid"
    echo "✅ $app_name started with PID $pid"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Start each Streamlit app with correct base URL paths
echo "📊 Starting Tensor Transform App..."
start_app "tensor_transform_app.py" 8501 "tensor-transform" "/tensor-transform"

echo "📊 Starting Tile Distribution App..."  
start_app "app.py" 8502 "tile-distribution" "/tile-distribution"

echo "🧵 Starting Thread Visualization App..."
start_app "thread_visualization_app.py" 8503 "thread-visualization" "/thread-visualization"

# Wait for apps to start
echo "⏳ Waiting for apps to start..."
sleep 5

# Check if apps are running
echo "🔍 Checking app status..."
for port in 8501 8502 8503; do
    if curl -s "http://localhost:$port" > /dev/null; then
        echo "✅ App on port $port is responding"
    else
        echo "❌ App on port $port is not responding"
    fi
done

echo "🎉 All apps deployed! Access them via:"
echo "📚 Documentation: https://ck.silobrain.com/"
echo "🔄 Tensor Transform: https://ck.silobrain.com/tensor-transform"
echo "📊 Tile Distribution: https://ck.silobrain.com/tile-distribution"
echo "🧵 Thread Visualization: https://ck.silobrain.com/thread-visualization"

echo ""
echo "🔧 Direct access for testing:"
echo "   http://localhost:8501 (Tensor Transform)"
echo "   http://localhost:8502 (Tile Distribution)"
echo "   http://localhost:8503 (Thread Visualization)"

echo ""
echo "📝 Logs are available in the logs/ directory"
echo "🛑 To stop all apps, run: ./stop_apps.sh"
echo ""
echo "💡 Make sure your nginx configuration is loaded:"
echo "   sudo nginx -t && sudo systemctl reload nginx" 