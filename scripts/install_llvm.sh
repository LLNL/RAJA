export LLVM_PATH=${HOME}/llvm/clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-14.04
export PATH=${LLVM_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${LLVM_PATH}/lib:${LD_LIBRARY_PATH}
export DOWNLOAD_URL=http://releases.llvm.org/${LLVM_VERSION}/clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-14.04.tar.xz
export TARFILE=${HOME}/download/llvm-${LLVM_VERSION}.tar.xz
if ! [[ -d "${LLVM_PATH}" ]]; then 
    if ! [[ -f ${TARFILE} ]]; then
        echo "curl -o ${TARFILE} ${DOWNLOAD_URL}"
        curl -o ${TARFILE} ${DOWNLOAD_URL}
    fi
    tar xf ${TARFILE} -C ${HOME}/llvm
fi

