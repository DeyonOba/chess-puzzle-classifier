#!/usr/bin/env bash

usage() {
    cat <<'EOF'
    
Decompress a Zstandard-compressed file.

USAGE:
    decompress_zstd.sh [FILE_DIR] [FILE_NAME] [OUTPUT_DIR]

ARGUMENTS:
    FILE_DIR    Directory containing the Zstandard-compressed file (default: ./data).
    FILE_NAME   Name of the Zstandard-compressed file (default: lichess_db_puzzle.csv.zst).
    OUTPUT_DIR  Directory to save the decompressed file (default: ./data).
EOF
}

if [[ ${1:-} == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v zstd >/dev/null 2>&1; then
    echo "zstd is required to decompress the file. Please install zstd and try again."
    exit 127
fi

file_dir="${1:-./data}"
file_name="${2:-lichess_db_puzzle.csv.zst}"
output_dir="${3:-./data}"

if [[ ! -d "$file_dir" ]]; then
    echo "Directory $file_dir does not exist. Please create it and try again."
    exit 1
fi

if [[ ! -f "$file_dir/$file_name" ]]; then
    echo "File $file_dir/$file_name does not exist. Please download it first and try again."
    exit 1
fi

if [[ ! -d "$output_dir" ]]; then
    echo "Directory $output_dir does not exist. Please create it and try again."
    exit 1
fi

echo "Decompressing $file_dir/$file_name to $output_dir..."
zstd -d "$file_dir/$file_name" -o "$output_dir/${file_name%.zst}"
echo "Decompression complete."
