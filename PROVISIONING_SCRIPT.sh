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
cd ray-of-light-vast/ && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Source the Rust environment variables
cd ray-of-light-vast/ && source $HOME/.cargo/env

# Install Python dependencies

cd ray-of-light-vast/ && python3 -m pip install -r requirements_new.txt
cd ray-of-light-vast/ && python3 -m pip install -U protobuf

echo "Setup complete!"
