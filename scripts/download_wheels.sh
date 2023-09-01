#!/bin/bash

# Set the GitHub release URL
RELEASE_URL="https://api.github.com/repos/casper-hansen/AutoAWQ/releases/tags/v0.0.1"

# Create a directory to download the wheels
mkdir -p dist
cd dist

# Download all the wheel files from the GitHub release
curl -s $RELEASE_URL | \
    jq -r ".assets[].browser_download_url" | \
    grep .whl | \
    xargs -n 1 wget

# Rename the wheels from 'linux_x86_64' to 'manylinux_x86_64'
for file in *linux_x86_64.whl; do
    mv "$file" "$(echo $file | sed 's/linux_x86_64/manylinux2014_x86_64/')"
done

cd ..
