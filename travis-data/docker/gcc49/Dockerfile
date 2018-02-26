FROM rajaorg/compiler:ubuntu-clang-base

LABEL maintainer="Tom Scogland <scogland1@llnl.gov>"
RUN \
       sudo apt-get -qq install -y --no-install-recommends \
         g++-4.9 \
         g++-4.9-multilib \
    && sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 100 \
    && sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 100 \
    && sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-4.9 100 \
    && sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-4.9 100

USER raja
WORKDIR /home/raja
