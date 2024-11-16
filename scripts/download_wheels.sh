#!/bin/bash

# Set variables
AWQ_VERSION="0.2.7.post1"
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
    grep -v '%2Brocm' | \
    xargs -n 1 -P 4 wget

cd ..
