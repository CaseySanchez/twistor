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

To run the Docker container execute the following command (requires an NVIDIA GPU):

```
docker run --rm --gpus all -it twistor
```