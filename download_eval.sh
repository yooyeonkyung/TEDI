#!/bin/bash

# Check if gdown is installed; if not, install it.
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Start installing..."
    pip install gdown
fi

download_and_extract() {
    local file_id=$1
    local output_name=$2
    local target_dir=$3

    echo "Downloading $output_name (ID: $file_id)..."
    
    gdown "$file_id" -O "$output_name"

    if [ -f "$output_name" ]; then
        echo "Extracting to $target_dir..."
        mkdir -p "$target_dir"
        tar -xzf "$output_name" -C "$target_dir" && rm "$output_name"
        rm "$output_name"
    else
        echo "Error: $output_name download failed."
    fi
}

# file id / file name / directory to be saved
# ppl binary file
download_and_extract "1YFWInrdFHUAWteCJpxaMrbDDjiqHOxJU" "ppl_yelp.tar.gz" "./data/Yelp/"
download_and_extract "12TXVQD51nUOIBmPOUhwPpBFulTcYazyI" "ppl_amazon.tar.gz" "./data/Amazon/"

echo "Files are all set!"