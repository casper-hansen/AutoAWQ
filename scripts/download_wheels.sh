#!/bin/bash

# Set variables
AWQ_VERSION="0.1.6"
RELEASE_URL="https://api.github.com/repos/casper-hansen/AutoAWQ/releases/tags/v${AWQ_VERSION}"

# Create a directory to download the wheels
mkdir -p dist
cd dist

# Download all the wheel files from the GitHub release
# excluding ones with '+cu' (%2B is + but encoded)
curl -s $RELEASE_URL | \
    jq -r ".assets[].browser_download_url" | \
    grep '\.whl' | \
    grep -v '%2Bcu' | \
    xargs -n 1 -P 4 wget

# Rename the wheels from 'linux_x86_64' to 'manylinux_x86_64'
for file in *linux_x86_64.whl; do
    mv "$file" "$(echo $file | sed 's/linux_x86_64/manylinux2014_x86_64/')"
done

cd ..
