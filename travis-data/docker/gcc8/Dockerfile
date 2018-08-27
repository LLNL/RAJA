FROM rajaorg/compiler:bionic-base

LABEL maintainer="Tom Scogland <scogland1@llnl.gov>"
ENV gccver=8

RUN sudo apt-get -qq update \
 && sudo apt-get -qq install -y --no-install-recommends \
            g++-${gccver} gcc-8-offload-nvptx \
 && sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${gccver} 100 \
 && sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${gccver} 100 \
 && sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-${gccver} 100 \
 && sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-${gccver} 100 \
 && sudo rm -rf /var/lib/apt/lists/*

USER raja
WORKDIR /home/raja
