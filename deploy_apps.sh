#!/bin/bash

# Deploy Streamlit Apps Script
# This script starts all three Streamlit apps on different ports

set -e  # Exit on any error

echo "🚀 Starting Streamlit Apps Deployment..."

# Kill any existing streamlit processes
echo "🔄 Cleaning up existing processes..."
pkill -f "streamlit run" || true

# Wait a moment for processes to terminate
sleep 2

# Function to start a Streamlit app
start_app() {
    local app_file=$1
    local port=$2
    local base_url=$3
    local app_name=$4
    
    echo "🎯 Starting $app_name on port $port..."
    
    # Start the app in the background
    nohup streamlit run "$app_file" \
        --server.port=$port \
        --server.address=127.0.0.1 \
        --server.baseUrlPath="/$base_url" \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --server.enableWebsocketCompression=false \
        --server.allowRunOnSave=true \
        > "logs/${app_name}.log" 2>&1 &
    
    # Get the PID
    local pid=$!
    echo "$pid" > "logs/${app_name}.pid"
    echo "✅ $app_name started with PID $pid"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Start each Streamlit app with the correct base URL paths
echo "📊 Starting Tensor Transform App..."
start_app "tensor_transform_app.py" 8501 "tensor-transform" "tensor-transform"

echo "📊 Starting Tensor Visualization App..."  
start_app "tensor_visualization_app.py" 8502 "tensor-visualization" "tensor-visualization"

echo "🧵 Starting Thread Visualization App..."
start_app "thread_visualization_app.py" 8503 "thread-visualization" "thread-visualization"

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
echo "🔄 Tensor Transform: https://ck.silobrain.com/tensor-transform/"
echo "📊 Tensor Visualization: https://ck.silobrain.com/tensor-visualization/"
echo "🧵 Thread Visualization: https://ck.silobrain.com/thread-visualization/"

echo ""
echo "📝 Logs are available in the logs/ directory"
echo "🛑 To stop all apps, run: ./stop_apps.sh"
echo ""
echo "💡 Make sure your nginx configuration is loaded:"
echo "   sudo nginx -t && sudo systemctl reload nginx" 