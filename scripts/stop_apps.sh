#!/bin/bash

# Stop Streamlit Apps Script
# This script stops all running Streamlit apps

echo "🛑 Stopping all Streamlit apps..."

# Function to stop an app by PID file
stop_app() {
    local app_name=$1
    local pid_file="logs/${app_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "🔄 Stopping $app_name (PID: $pid)..."
            kill "$pid"
            # Wait for graceful shutdown
            sleep 2
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "⚡ Force stopping $app_name..."
                kill -9 "$pid"
            fi
            echo "✅ $app_name stopped"
        else
            echo "⚠️  $app_name (PID: $pid) was not running"
        fi
        rm -f "$pid_file"
    else
        echo "⚠️  No PID file found for $app_name"
    fi
}

# Stop each app
stop_app "tensor-transform"
stop_app "tile-distribution" 
stop_app "thread-visualization"

# Also kill any remaining streamlit processes
echo "🧹 Cleaning up any remaining Streamlit processes..."
pkill -f "streamlit run" || true

echo "✅ All Streamlit apps stopped!"

# Optional: Clean up log files (uncomment if needed)
# echo "🧹 Cleaning up log files..."
# rm -rf logs/*.log 