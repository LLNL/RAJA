.. ##
.. ## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory
.. ##
.. ## LLNL-CODE-689114
.. ##
.. ## All rights reserved.
.. ##
.. ## This file is part of RAJA.
.. ##
.. ## For details about use and distribution, please read RAJA/LICENSE.
.. ##


.. _getting_started-label:

*************************
Getting Started With RAJA
*************************

This section will show you how to get up and running with RAJA quickly.

============
Requirements
============

The primary requirement for using RAJA is a C++11 compliant compiler.
Accessing the full range of RAJA features, such as all available programming
model back-ends, will require additional support. Available options and how
to enable or disable them are described in :ref:`configopt-label`. To
build and use RAJA in its simplest form requires:

- C++ compiler with C++11 support
- `CMake <https://cmake.org/>`_ 3.3 or greater

==================
Build and Install
==================

----------------
Getting the code
----------------

The RAJA project is hosted on a `GitHub project <https://github.com/LLNL/RAJA>`_.
To get the code, clone the repository into a local working space using
the command::

   $ git clone --recursive https://github.com/LLNL/RAJA.git

The ``--recursive`` argument above is needed to pull in all Git *submodules*
that we use in RAJA. Currently, we have only one, the BLT build system that
we use. For information on BLT, see `BLT <https://github.com/LLNL/blt>`_.

After running the clone command, a copy of the RAJA repository will reside in
the ``RAJA`` subdirectory and you will be on the ``develop`` branch of RAJA,
which is our default branch.

If you forget to pass the ``--recursive`` argument to the ``git clone``
command, you can type the following commands after cloning::

  $ cd RAJA
  $ git submodule init
  $ git submodule update

Either way, the end result is the same and you should be good to go.

.. note:: Any time you switch branches in RAJA, you need to re-run the
          'git submodule update' command to set the Git submodules to
          what is used by the new branch.


--------------
Building RAJA
--------------

RAJA uses CMake to configure a build. Basic configuration looks like::

  $ mkdir build && cd build
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../

.. note:: Builds must be out-of-source.  RAJA does not allow building in
          source directory, so you must create a build directory.

CMake will provide output about the compiler that has been detected, and
which features are discovered. Some features, like OpenMP, will be enabled
if they are discovered. For a complete summary of configuration options, please
see :ref:`configopt-label`.

After CMake successfully completes, RAJA is compiled by running ``make``
in the build directory::

  $ make

.. note:: RAJA is configured to build its unit tests by default. If you do not
          disable them with the appropriate CMake option, you may run them
          after the build completes to check if everything built properly.
          The easiest way to do this is to type::

          $ make test

          You can also run individual tests by invoking individual test 
          executables directly. They live in subdirectories in the ``test`` 
          directory. RAJA tests use the 
          `Google Test framework <https://github.com/google/googletest>`_, 
         so you can also run tests via Google Test commands.


----------------
Installing RAJA
----------------

To install RAJA, run the command ::

  $ make install

This will copy RAJA header files to the ``include`` directory and the RAJA
library will be installed in the ``lib`` directory.


=================
Basic RAJA Usage
=================

Let's take a quick tour through a few key RAJA concepts. Directions for
locating the complete working code for this first RAJA example are given
in :ref:`firstexample-label`.

The central loop traversal concept in RAJA is a ``forall`` template method, 
which runs a loop based on parameters it is given. RAJA ``forall`` methods
encapsulate loop execution details allowing the loop to be run in 
different ways without changing the loop code itself, only the parameters
you pass to the method. 

We use a simple daxpy operation to walk you through how to write a RAJA loop 
kernel and how it compares to a typical C-style for-loop. A C-style loop 
version of the daxpy operation would look something like this::

  const int N = 1000;

  double* a = new double[N];
  double* b = new double[N];

  // Initialize a and b...

  double c = 3.14159;

  for (int i = 0; i < N; i++) {
    a[i] += b[i] * c;
  }

This loop executes sequentially, iterating over the range of ``i``
values [0, N) one after the other.

The RAJA form of this sequential loop replaces the ``for-loop``
with a call to a RAJA ``forall`` method::

  // Initialize a, b, c as before...

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (int i) {
    a[i] += b[i] * c;
  });

The data allocation and loop body are exactly the same as the original code.
The ``RAJA::forall`` method takes as arguments the loop bounds in a
``RAJA::RangeSegment`` object and a C++ lambda expression containing the loop 
body. The method is templated on an `execution policy` type and the 
template specialization for that type determines how the loop will run. Here, 
we use the ``RAJA::seq_exec`` policy to run the loop iterations sequentially, 
in order, exactly like the original loop.

Of course, this example isn't very exciting. 

You may be asking yourself: "Why are they doing this? They're just re-writing 
a simple loop in a more complicated way using C++11 features so that it runs 
exactly the same as I would have written it in C...."

The reason is that RAJA provides mechanisms that make it easy to run the 
loop with different parallel programming model back-ends (which have different
syntaxes and usage concepts) tune for performance by mapping loop 
iterations to different orderings and data layouts without changing the 
code as it appears in an application.

For example, since our example loop is data-parallel (i.e., all
iterations are independent), we can run it in parallel by replacing the
execution policy. For example, to run the loop in parallel using OpenMP
multi-threading, one could use the following execution policy::

  RAJA::omp_parallel_for_exec

Alternatively, to run the loop on an NVIDIA GPU using CUDA, use this
execution policy instead::

  const int CUDA_BLOCK_SIZE = 512;

  RAJA::cuda_exec<CUDA_BLOCK_SIZE>

Here, we specify that the loop should run with 512 threads in a CUDA 
`thread block`. If we omit the thread block size template parameter, this
policy provides 256 threads as the default. 

Note that we have assumed that the data arrays on the GPU device have been
allocated and initialized properly. Also, to exercise different
parallel programming model back-ends that RAJA supports, they must be
enabled when RAJA is configured. See :ref:`configopt-label` for more 
information.


.. _firstexample-label:

------------------------
First RAJA example code
------------------------

If you want to view and run the daxpy example code yourself, the complete 
implementation is located in the file: ``RAJA/examples/ex0-daxpy.cpp``. 

After building RAJA, with the options you select, the executable for this 
example, will reside in the file: ``<build-dir>/examples/bin/ex0-daxpy``.
