#!/bin/bash

#Change Directory
cd /workspace

# Clone the repository
git clone https://github.com/digilateraldev/ray-of-light-vast.git

# Change to the project directory
cd ray-of-light-vast

# Install dependencies
sudo apt update
sudo apt install -y imagemagick sox

# Install Rust
echo | curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Source the Rust environment variables
source $HOME/.cargo/env

# Install Python dependencies
pip install -r requirements_new.txt
pip install -U protobuf

echo "Setup complete!"
