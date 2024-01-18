FROM smijin/remkit1d-ci:latest
LABEL maintainer="stefan.mijin@ukaea.uk"
LABEL version="1.0.0"
LABEL description="Docker container for the ReMKiT1D Workshop"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

# Update and Installs Required Packages for ReMKiT1D and other Packages
RUN apt update \
        && apt install -y \
        python3 \
        pip

WORKDIR /home

# Install the Python library
RUN git clone -b dev-v1.1.0 https://github.com/ukaea/ReMKiT1D-Python.git

RUN pip install ./ReMKiT1D-Python/

WORKDIR /home 

# Install ReMKiT1D
RUN git clone -b dev-v1.1.0 https://github.com/ukaea/ReMKiT1D.git

WORKDIR /home/ReMKiT1D 

RUN mkdir debug && cd debug && cmake .. && make -j 

# Test ReMKiT1D 

WORKDIR /home/ReMKiT1D/debug

RUN make test > /home/ReMKiT1D_debug_test.out

WORKDIR /home/ReMKiT1D 

RUN mkdir build && cd build && cmake .. && make -j 

WORKDIR /home/ReMKiT1D/build

RUN make test > /home/ReMKiT1D_build_test.out

WORKDIR /home

# Install other required Python packages and test RMK_Support

RUN pip install ipywidgets ipykernel jupyter_bokeh pytest 

RUN pytest ./ReMKiT1D-Python/RMK_support/ > /home/RMK_support_test.out

RUN git clone https://github.com/ukaea/ReMKiT1D-Workshop-2024

