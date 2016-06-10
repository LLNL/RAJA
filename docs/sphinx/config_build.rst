.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For release details and restrictions, please see raja/README-license.txt
.. ##


===================================
Configuring and Building RAJA 
===================================

This section describes RAJA configuration options and how to build RAJA 
with these options.

It is important to note that RAJA requires compiler support for lambda 
functions and a few other C++11 features. Please make sure your compiler
is C++11-compliant before attempting to build RAJA.

CMake System
-----------------

RAJA uses a basic CMake-based system, but you don't need to know much 
about CMake to build RAJA. It does require that you use a CMake version 
greater than 2.8. If needed, you can learn more about CMake and download
it at `<http://www.cmake.org>`_

You can build RAJA like any other CMake project, provided you have a C++
compiler that supports the features in the C++11 standard that we use. The 
simplest way to build the code is to do the following in the top-level RAJA 
directory:

.. code-block:: sh

    $ mkdir build
    $ cd build
    $ cmake ../
    $ make

This will build RAJA and the examples and tests it contains using default
settings (described below).

It is important to note that the RAJA CMake system does not allow
in-source builds so that you can easily configure and build for various
systems and compilers from the same source code.

Configuration Options
----------------------

The RAJA configuration can be set using standard CMake variables along with
RAJA-specific variables. For example, to make a release build with some 
system default GNU compiler and then install the RAJA header files and
libraries in a specific directory location, you could do the following in 
the top-level RAJA directory:

.. code-block:: sh

    $ mkdir build-gnu-release
    $ cd build-gnu-release
    $ cmake \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=../install-gnu-release ../
    $ make
    $ make install

Following CMake convention, RAJA supports three build types: 'Release', 
'RelWithDebInfo', and 'Debug'. Depending on which you choose, you will not
have to explicitly add the '-g' compiler flags to generate debugging symbols.
The default compiler flags for optimization, etc. for various compilers can
be found in the `cmake/SetupCompilers.cmake` file.

RAJA uses a variety of custom variables to control how it is compiled. Many 
of these manifest as compilation directives and definitions that appear in 
the RAJA 'config.hxx' file that is generated during the CMake process. The
RAJA configuration header file is included in other RAJA headers as needed
so all options propagate through the build process consistently.

These variables are turned on or off similar to standard CMake variables; 
e.g., to enable RAJA OpenMP functionality, do this ::

    -DRAJA_ENABLE_OPENMP=On

The `scripts` directory contains several bash shell scripts that can be used 
for common configurations. For example, you can type the following commands
starting at the top-level RAJA directory to build a version of RAJA for 
specific versions of the GNU and Intel compilers in a build subdirectory:

.. code-block:: sh

    $ mkdir my-builds
    $ cd my-builds
    $ ../scripts/gcc-4.9.3.sh 
    $ cd build-gnu-4.9.3-release
    $ make
    $ cd ..
    $ ../scripts/icpc-16.0.109.sh
    $ cd build-icpc-16.0.109-release
    $ make

These scripts also serve as useful examples for those who are not fluent in 
CMake.

The following list describes these variables and their defaults.

  * **Tests and Examples**

     *RAJA_ENABLE_TESTS* -- Controls whether RAJA tests are compiled. 
     
      Default : *RAJA_ENABLE_TESTS=On*

  * **Programming Models**

     Variables to control which RAJA programming model back-ends are enabled
     and their defaults (names are descriptive of what they enable):

     *RAJA_ENABLE_OPENMP=On*

     *RAJA_ENABLE_CUDA=Off*

     *RAJA_ENABLE_CILK=Off*

  * **Data Types, Sizes, Alignment Parameters, etc.**

     RAJA internally uses several parameters to define platform-specific
     constants for index ranges and data alignment. The defaults should be 
     reasonable for most cases. For more details on the data type-related
     options, please see the header file `real_datatypes.hxx`.

     *RAJA_USE_DOUBLE, RAJA_USE_FLOAT* -- Control typedefs that can be used
     to parameterize floating point types in applications and easily switch 
     between types. Exactly one of these can be on at a time. 

     Default : *RAJA_USE_DOUBLE=On*, *RAJA_USE_FLOAT=Off*


     *RAJA_USE_COMPLEX* -- Enables typedefs for complex data types if needed. 
     
     Default : *RAJA_USE_COMPLEX=Off*

     
     There are several options for controlling RAJA floating-point data
     pointer typedefs. Exactly one of these can be on at a time.
 
     *RAJA_USE_BARE_PTR* -- Use standard undecorated pointer
    
     *RAJA_USE_RESTRICT_PTR* -- Use pointer with restrict qualifier

     *RAJA_USE_RESTRICT_ALIGNED_PTR* -- Use pointer with restrict and
     alignment (see *RAJA_DATA_ALIGN* below)

     *RAJA_USE_PTR_CLASS* -- Use pointer class with an overloaded `[]` operator 
     that applies restrict and alignment intrinsics. This is useful when a 
     compiler does not support these attributes in a typedef.

     Default : *RAJA_USE_RESTRICT_PTR=On* for CPU-only code and 
     *RAJA_USE_BARE_PTR=On* when *RAJA_ENABLE_CUDA=On*


     *RAJA_RANGE_ALIGN* -- Constrains alignment of begin/end indices in range 
     segments constructed by index set builder methods. That is, begin and
     end indices of such segments will be multiples of this value. 

     Default : *RAJA_RANGE_ALIGN=4*

 
     *RAJA_RANGE_MIN_LENGTH* -- Defines the minimum length of range segments 
     constructed by index set builder methods; this should be an integer 
     multiple of *RAJA_RANGE_ALIGN*. 

     Default : *RAJA_RANGE_MIN_LENGTH=32* 


     *RAJA_DATA_ALIGN* -- Used in compiler-specific intrinsics and typedefs
     to specify data alignment constraints; units of **bytes**. 

     Default : *RAJA_DATA_ALIGN=64*
     
 
     *RAJA_COHERENCE_BLOCK_SIZE* -- Defines thread coherence value for shared
     memory blocks used by RAJA reduction objects. 

     Default : *RAJA_COHERENCE_BLOCK_SIZE=64*


  * **Timer Options**

     RAJA provides a simple timer class that is used in RAJA example codes
     to determine execution timing and can be used in other apps as well.
     Three variables are available to select the timing mechanism used.
     Exactly one of these can be on at a time.

     *RAJA_USE_GETTIME* -- Use `timespec` from the C-std library time.h file

     *RAJA_USE_CLOCK* -- Use `clock_t` from time.h 

     *RAJA_USE_CYCLE* -- Use `ticks` from the cycle.h file in the FFTW library

     Default : *RAJA_USE_GETTIME=On*. This should be a good choice for most i
     situations.

  * **Fault Tolerance Options**
    
     RAJA contains some internal macros that are used to explore a simple
     experimental loop-level fault tolerance model. By default, this feature 
     is off. To enable it, turn the following variables on:

     *RAJA_ENABLE_FT* -- Enables fault-tolerance mechanism

     *RAJA_REPORT_FT* -- Enables a report of fault-tolerance enabled run 
     (e.g., number of faults detected, recovered from, overhead, etc.)


Did I build RAJA correctly?
---------------------------

You can verify that RAJA is built correctly with the options you want, you 
can run some unit tests...

.. warning:: Need to add a 'make tests' or 'make check' target that 
             compiles (if needed) and runs some basic tests with sensible 
             output that makes it clear to users that their RAJA build is
             good to go or is not.


Additional Information for Using LLNL Platforms
------------------------------------------------

We are the first ones to admit that, while our build system works, it is not
completely 'push button' for all platforms. For some machines at LLNL, there 
are a few platform-specific things you must do to make things work. We note
them here. As things improve, we will update the information here.

  * BG/Q builds

    So far, at LLNL, we have only built and tested RAJA on our BG/Q systems
    using the GNU compiler. We have had moderate success with the clang 
    compiler. To build with the GNU compiler, you need to set the version of 
    CMake before running it. You can do this by typing ::

      $ use cmake-3.1.2

  * nvcc builds

    To compile with 'nvcc' on LC machines that have GPUs that support CUDA, 
    you will have to load the CUDA module and set the host compiler before
    you run CMake. For example, type these lines :: 

      $ module load cudatoolkit/7.5
      $ use gcc-4.9.3p


