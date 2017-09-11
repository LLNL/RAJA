export LLVM_PATH=${HOME}/llvm/
export PATH=${LLVM_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${LLVM_PATH}/lib:${LD_LIBRARY_PATH}
[[ -z ${DOWNLOAD_URL+x} ]] && export DOWNLOAD_URL=http://releases.llvm.org/${LLVM_VERSION}/clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-14.04.tar.xz
export TARFILE=${HOME}/download/llvm-${LLVM_VERSION}.tar.xz
if ! [[ -x "${LLVM_PATH}/bin/clang++" ]]; then 
    if ! [[ -f ${TARFILE} ]]; then
        echo "curl -o ${TARFILE} ${DOWNLOAD_URL}"
        curl -o ${TARFILE} ${DOWNLOAD_URL}
    fi
    tar xf ${TARFILE} -C ${HOME}/llvm --strip-components 1
    ln -s ${LLVM_PATH}/bin/clang++ ${LLVM_PATH}/bin/clang++-${LLVM_VERSION}
    ln -s ${LLVM_PATH}/bin/clang ${LLVM_PATH}/bin/clang-${LLVM_VERSION}
fi

