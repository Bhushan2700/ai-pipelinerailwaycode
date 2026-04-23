#!/bin/bash
set -e
source .venv/bin/activate
QUEUE=${1:-question_processing}
python -m workers.runner --queues "$QUEUE"
