#!/usr/bin/env bash
# Upload and download activation_data to/from S3
set -euo pipefail

BUCKET="s3://doxascope-tests"
S3_PREFIX="${BUCKET}/activation_data"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/activation_data"

usage() {
    echo "Usage: $0 {upload|download} [model_name]"
    echo
    echo "Commands:"
    echo "  upload    Sync local activation_data/ to S3"
    echo "  download  Sync S3 activation_data/ to local"
    echo
    echo "Options:"
    echo "  model_name  Optional. Sync only a specific model (e.g. takeru_200M)"
    echo
    echo "Examples:"
    echo "  $0 upload"
    echo "  $0 download takeru_200M"
    exit 1
}

[[ $# -lt 1 ]] && usage

COMMAND="$1"
MODEL="${2:-}"

if [[ -n "$MODEL" ]]; then
    S3_PATH="${S3_PREFIX}/${MODEL}/"
    LOCAL_PATH="${LOCAL_DIR}/${MODEL}/"
else
    S3_PATH="${S3_PREFIX}/"
    LOCAL_PATH="${LOCAL_DIR}/"
fi

case "$COMMAND" in
    upload)
        echo "Uploading ${LOCAL_PATH} -> ${S3_PATH}"
        aws s3 sync "$LOCAL_PATH" "$S3_PATH" --progress-multiline
        echo "Done."
        ;;
    download)
        mkdir -p "$LOCAL_PATH"
        echo "Downloading ${S3_PATH} -> ${LOCAL_PATH}"
        aws s3 sync "$S3_PATH" "$LOCAL_PATH" --progress-multiline
        echo "Done."
        ;;
    *)
        usage
        ;;
esac
