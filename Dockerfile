FROM nvcr.io/nvidia/cuda:13.1.1-devel-ubuntu24.04

RUN apt update && \
    apt install -y build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/src/hdf5

RUN git clone -b 2.1.0 https://github.com/HDFGroup/hdf5.git . && \
    cmake -S . -B build -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_BUILD_CPP_LIB=ON && \
    cmake --build build --config Release --target install -j $(nproc)

WORKDIR /usr/local/src/gatl

RUN git clone -b csta https://github.com/CaseySanchez/gatl.git . && \
    cmake -S ./cpp -B build -DCMAKE_BUILD_TYPE=Release

WORKDIR /usr/local/src/twistor
