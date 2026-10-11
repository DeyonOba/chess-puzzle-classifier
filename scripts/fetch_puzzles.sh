#!/usr/bin/env bash

usage() {
    cat <<'EOF'

Download the latest Lichess puzzle database.

USAGE:
    fetch_puzzles.sh [OUTPUT_DIR]

ARGUMENTS:
    OUTPUT_DIR  Download directory. If directory does not exist, it will be created. If not specified, the current directory will be used (default: ./data).
EOF
}


if [[ ${1:-} == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

output_dir="${1:-./data}"

if ! command -v wget >/dev/null 2>&1; then
    echo "wget is required to download the puzzle database. Please install wget and try again."
    exit 127
fi

mkdir -p "$output_dir"
output_file="$output_dir/lichess_db_puzzle.csv.zst"
base_url="https://database.lichess.org/lichess_db_puzzle.csv.zst"


if [[ -f "$output_file" ]]; then
    echo "File $output_file already exists. Skipping download."
    exit 0
fi

echo "Downloading Lichess puzzle database to $output_file..."
wget -O "$output_file" "$base_url"
