#!/bin/bash

#Change Directory
cd /workspace

# Clone the repository
git clone https://github.com/digilateraldev/ray-of-light-vast.git

# Change to the project directory
cd ray-of-light-vast

# Install dependencies
sudo apt update
sudo apt install -y imagemagick sox cargo

# Install Python dependencies

cd ray-of-light-vast/ && python3 -m pip install -r requirements_new.txt
cd ray-of-light-vast/ && python3 -m pip install -U protobuf

echo "Setup complete!"
