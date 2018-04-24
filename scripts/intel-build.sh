git submodule init && git submodule update
mkdir build-intel && cd build-intel && cmake -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc   -DCMAKE_CXX_FLAGS=' -std=c++11 -qopenmp -O3 -xCORE-AVX2' ../
make -j

