# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

sudo apt install -y build-essential
sudo apt install -y cmake
sudo apt install -y bzip2 ca-certificates file gcc-13 g++-13 gfortran-13 git gzip lsb-release patch python3 tar unzip xz-utils zstd
sudo apt install -y libssl-dev
sudo apt install -y libomp-19-dev
if ! nvidia-smi &> /dev/null; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install -y nvidia-driver-560
    sudo reboot
fi
