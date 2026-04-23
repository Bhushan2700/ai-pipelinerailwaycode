#!/bin/bash
set -e

required_vars=(
    "FM__HOST"
    "FM__DATABASE"
    "FM__USERNAME"
    "FM__PASSWORD"
    "OPENAI__API_KEY"
    "REDIS__URL"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Required environment variable $var is not set" >&2
        exit 1
    fi
done

exec "$@"
