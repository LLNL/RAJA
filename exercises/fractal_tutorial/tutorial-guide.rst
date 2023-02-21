================
Fractal Tutorial
================

This tutorial includes several implementations of a Mandelbrot set Fractal code.
The code originated from Dr. Martin Burtscher of the Efficient Computing Lab at
Texas State University. You can find more here: https://userweb.cs.txstate.edu/~burtscher/research.html

The tutorial first starts with a serial implementation of the fractal code. From
there, we build on native CUDA, RAJA-CUDA, RAJA-HIP, and a few other implementations.
The final lessons include a more complex fractal implementation that includes
RAJA-TEAMS.

To start, let's build the tutorial within the build directory of the RAJA repo:: 

        module load cuda/10.2.89
        module load cmake/3.20.2
        module load gcc/8.3.1
        cmake -DENABLE_CUDA=On -DENABLE_OPENMP=Off -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/tce/packages/cuda/cuda-10.2.89/bin/nvcc \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/tce/packages/cuda/cuda-10.2.89 -DBLT_CXX_STD=c++14 -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_EXERCISES=On ../
        make -j

.. note::
        I am building this code on LC's lassen machine. If these build instructions don't work for you, you can refer to the
        build documentation from RAJA's ReadTheDocs or use one of the provided build scripts.

Now, we can build the serial implementation with `./bin/fractal 1024 256`. The first argument
is the width of the fractal (1024) and the second is the depth of the fractal (256). It may
be interesting to see how the fractal changes with different width and depth values.
