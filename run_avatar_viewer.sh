#!/bin/bash
# Launch interactive Gradio-based avatar viewer

CHECKPOINT="CKPT"
SUBJECT_ID=4
SOURCE_CAMS="C24 C25"
PORT=7860
ENV_ID="env_002"
EXPR_ID="expr_00011"

echo "🚀 Starting Interactive Avatar Player..."
echo "📱 Access at: http://localhost:$PORT"
echo "🌐 Or use SSH port forwarding: ssh -L $PORT:localhost:$PORT your_server"
echo ""

python gaussian_avatar_player_gradio.py \
    --checkpoint-path "$CHECKPOINT" \
    --subject-id "$SUBJECT_ID" \
    --source-cam-ids $SOURCE_CAMS \
    --env-id "$ENV_ID" \
    --expr-id "$EXPR_ID" \
    --port $PORT

# Use --share flag to get a public link:
# python gaussian_avatar_player_gradio.py ... --share


