# GPU Parallelizing of the RNEA Algorithm

This is the code used to write the paper "An investigation into parallelizing the RNEA Algorithm using PyTorch and CUDA.jl"

## References
The base code for the RNEA algorithm is from the A2R Lab's [GRiD repository](https://github.com/A2R-Lab/GRiD).

The batched RNEA algorithm was also based on work done by the [A2R Lab](https://a2r-lab.org/)

## Installation Instructions:
### Install Python Dependencies
The code in this repository uses several external packages: ```beautifulsoup4, lxml, numpy, sympy, pytorch``` which can be automatically installed by running:
```shell
pip3 install -r requirements.txt
```

### Install Julia Dependencies
If Julia isn't installed, it can be installed by running (on a UNIX system):
```shell
curl -fsSL https://install.julialang.org | sh
```

Then install CUDA.jl by running:
```shell
cd Julia
julia Install_CUDA.jl
```

### Install CUDA Dependencies
```
sudo apt-get update
sudo apt-get -y install xorg xorg-dev linux-headers-$(uname -r) apt-transport-https
```
### Download and Install CUDA 
Note: for Ubuntu 20.04 see [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) for other distros
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```
### Add the following to ```~/.bashrc```
```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="opt/nvidia/nsight-compute/:$PATH"
```
