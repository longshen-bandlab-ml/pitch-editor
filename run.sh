# #!/bin/bash
# set -e

# cd "$(dirname "$0")"

conda activate pitch-editor

# Open browser after a short delay to let the server start
(sleep 2 && open http://localhost:8767) &

uvicorn main:app --host 0.0.0.0 --port 8767
