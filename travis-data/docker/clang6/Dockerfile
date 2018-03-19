FROM rajaorg/compiler:ubuntu-clang-base

LABEL maintainer="Tom Scogland <scogland1@llnl.gov>"
ENV llvmtar=clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-14.04
ENV tarext=.tar.xz
RUN \
       wget -q --no-check-certificate http://releases.llvm.org/6.0.0/${llvmtar}${tarext} \
    && tar xf ${llvmtar}${tarext} \
    && sudo cp -fR ${llvmtar}/* /usr \
    && rm -rf ${llvmtar} \
    && rm ${llvmtar}${tarext}

USER raja
WORKDIR /home/raja
