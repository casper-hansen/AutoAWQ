#!/bin/bash

# Set variables
AWQ_VERSION="0.2.8"
RELEASE_URL="https://github.com/casper-hansen/AutoAWQ/archive/refs/tags/v${AWQ_VERSION}.tar.gz"

# Create a directory to download the wheels
mkdir -p dist

# Download the tar.gz file to dist directory
wget -O "dist/v${AWQ_VERSION}.tar.gz" $RELEASE_URL