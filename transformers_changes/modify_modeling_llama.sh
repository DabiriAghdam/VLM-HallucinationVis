#!/usr/bin/env bash

# Use Python to get the exact file path of the transformers module
TRANSFORMERS_FILE_PATH=$(python3 -c "import transformers; print(transformers.__file__)")

# Extract the directory path from the full file path
TRANSFORMERS_DIR=$(dirname "$TRANSFORMERS_FILE_PATH")

# The destination directory inside transformers
DEST_DIR="$TRANSFORMERS_DIR/models/llama"

# Copy shared_state.py from the current directory (where this script is) to ss.py in the destination
cp modeling_llama_changed.py "$DEST_DIR/modeling_llama.py"

echo "modeling_llama_changed.py has been copied to: $DEST_DIR/modeling_llama.py"
