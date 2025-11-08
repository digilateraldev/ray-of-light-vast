#!/bin/bash
#@reboot bash -x /workspace/ray-of-light-vast/start_job1.sh > /workspace/ray-of-light-vast/output_cron.log 2>&1
# Path to check
DIR="/workspace/ray-of-light-vast"

# Function to run your main job
run_job() {
  # Run both commands simultaneously in a subshell
  (
    cd "$DIR" && \
    python again_try.py arif.mp4 final_arif_with_api.mp4 "Arif" "Dentist" "Mumbai" --intensity soft > "$DIR/output.log" 2>&1
  )

  # Check for MoviePy completion
  if grep -q "MoviePy - video ready" "$DIR/output.log"; then
    echo "✅ Job done, stopping instance..."
    curl -X PUT \
      -d '{"state": "stopped"}' \
      "https://console.vast.ai/api/v0/instances/27613273/?api_key=$VAST_API_KEY"
  else
    echo "❌ MoviePy did not complete successfully" >> "$DIR/output.log"
  fi
}

# Loop until directory exists
while true; do
  if [ -d "$DIR" ]; then
    echo "📁 Directory exists, running job..."
    run_job
    break
  else
    echo "⏳ Directory not found. Retrying in 2 seconds..."
    sleep 2
  fi
done
