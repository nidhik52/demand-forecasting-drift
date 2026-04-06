#!/bin/bash

# Install Python 3.11 if not present
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install setuptools==68.2.2
pip install -r requirements.txt

if [ "${ENABLE_PROPHET}" = "1" ]; then
    echo "Installing Prophet + CmdStan dependencies..."
    pip install -r requirements-prophet.txt

    # Install CmdStan
    echo "Installing CmdStan..."
    python -c "
import cmdstanpy
import os
os.environ['CXXFLAGS'] = '-O0 -g0 -std=c++14'
cmdstanpy.install_cmdstan(version='2.33.1', dir=os.path.expanduser('~/.cmdstan'), overwrite=True)
"
fi

echo "Setup complete! Activate with: source .venv/bin/activate"