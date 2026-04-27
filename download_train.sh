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
# amazon model pth file
download_and_extract "1izBGbQfsg-aTCrUVLat09RVaDANTk0Iz" "reg_d_0807.tar.gz" "./result/amazon_saved_reg_d/240807/240807_2035/"
download_and_extract "10oLrK_waNLDoFIcynLI8AI7ZbL4P2E1w" "reg_0805.tar.gz" "./result/amazon_saved_reg/240805/240805_1437/"
download_and_extract "1MlHW2IKvL-xR60RJQs0XLgvfkb4k4i4L" "reg_0807.tar.gz" "./result/amazon_saved_reg/240807/240807_2048/"

# yelp model pth file
download_and_extract "1jUafvZ38-NE3Sh2rXFHW9wfWlGGZxxQK" "reg_d_1024.tar.gz" "./result/yelp_saved_reg_d/241024/241024_1435/"
download_and_extract "1SNxJ_WlMdFMX1RioBjGXZqHUnlySmGjW" "reg_1024.tar.gz" "./result/yelp_saved_reg/241024/241024_1435/"
download_and_extract "1tB-I7J3pNxA0E9GwGvAWQx6NnGIUtJqa" "reg_1025.tar.gz" "./result/yelp_saved_reg/241025/241025_2153/"

echo "Files are all set!"