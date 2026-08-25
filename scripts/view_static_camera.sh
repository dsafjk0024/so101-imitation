#!/usr/bin/env bash
set -euo pipefail

FFPLAY="${FFPLAY:-$(command -v ffplay)}"
: "${FFPLAY:?ffplay not found — install ffmpeg or set FFPLAY}"

exec "$FFPLAY" \
  -fflags nobuffer -flags low_delay -framedrop \
  -f v4l2 -framerate 30 -video_size 640x480 -input_format mjpeg \
  /dev/cam_top -window_title "top camera"
