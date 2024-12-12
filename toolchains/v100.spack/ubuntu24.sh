# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

sudo apt install -y build-essential
sudo apt install -y cmake
sudo apt install -y bzip2 ca-certificates file gcc-13 g++-13 gfortran-13 git gzip lsb-release patch python3 tar unzip xz-utils zstd
if ! nvidia-smi &> /dev/null; then
    wget https://uk.download.nvidia.com/tesla/560.35.03/nvidia-driver-local-repo-ubuntu2404-560.35.03_1.0-1_amd64.deb
    sudo cp /var/nvidia-driver-local-repo-ubuntu2404-560.35.03/nvidia-driver-local-BCE469C4-keyring.gpg /usr/share/keyrings/
    sudo dpkg -i nvidia-driver-local-repo-ubuntu2404-560.35.03_1.0-1_amd64.deb
    sudo apt update
    sudo apt install -y nvidia-driver-560
fi
