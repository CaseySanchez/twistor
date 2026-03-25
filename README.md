# Twistor

## Overview

Twistor is a header-only C++20/CUDA library for GPU-accelerated gauge field simulation in the Conformal Spacetime Algebra (CSTA), the conformal extension of the Spacetime Algebra (STA).

## Requirements

### Docker

Install [Docker](https://docs.docker.com/engine/install/).

### NVIDIA Container Toolkit

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Docker

### Build

To build the Docker container execute the following command:

```
docker build -t twistor .
```

### Run

To run the Docker container execute the following command:

```
docker run --gpus all --rm -it -v $(pwd):/usr/local/src/twistor twistor
```

## CMake

### Build

To build the examples within the container execute the following command:

```
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build -j $(nproc)
```