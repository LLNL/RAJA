.. ##
.. ## Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

****************
Getting Started
****************

This page will show you how to get up and running with RAJA quickly.

============
Requirements
============

The primary requirement for using RAJA is a C++11 compliant compiler.
Accessing the full range of RAJA features, such as all available programming
model backends, may require additional support. Available options and how
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

The RAJA project is hosted on `GitHub <https://github.com/LLNL/RAJA>`_.
To get the code, clone the repository into a local working space using
the command::

   $ git clone --recursive https://github.com/LLNL/RAJA.git

The '--recursive' argument above is needed to pull in all Git *submodules*
that we use in RAJA. Currently, we have only one -- BLT build system that
we use. For information on BLT, see `BLT <https://github.com/LLNL/blt>`_.

After running the clone command, a copy of the RAJA repo will reside in
the ``RAJA`` subdirectory and you will be on the ``develop`` branch of RAJA,
which is our default branch.

If you forget to pass the '--recursive' argument to the 'git clone'
command, you can type the following commands after cloning::

  $ cd RAJA
  $ git submodule init
  $ git submodule update

Either way, the end result is the same and you are good to go.

.. note:: Any time you switch branches in RAJA, you need to re-run the
          'git submodule update' command to set the Git submodules to
          what is used by the new branch.


--------------
Building RAJA
--------------

RAJA uses CMake to handle builds. Configuration looks like::

  $ mkdir build && cd build
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../

.. note:: Builds must be out-of-source.  RAJA does not allow building in
          source directory, so you must create a build directory.

CMake will provide output about the compiler that has been detected, and
what features are discovered. Some features, like OpenMP, will be enabled
if they are discovered. For a complete summary of configuration options, please
see :ref:`configopt-label`.

Once CMake has completed, RAJA is compiled using Make::

  $ make

.. note:: RAJA is configured to build its unit tests by default. If you don't
          disable them with the appropriate CMake option, you may run them
          after the build completes to check if everything built properly.
          The easiest way to do this is to type::

          $ make test

          You can also run individual tests by invoking the test executables
          directly which live in subdirectories in the ``test`` directory.
          RAJA tests use the `Google Test framework <https://github.com/google/googletest>`_, so you can also run tests via Google Test commands.


----------------
Installing RAJA
----------------

To install RAJA, run the command ::

  $ make install

This will copy RAJA header files to the ``include`` directory in the build
space and the RAJA library will be installed in the ``lib`` directory.

=================
Basic RAJA Usage
=================

Let's take a quick tour through a few important RAJA features. A complete
example based on the tour that you can compile and run is available here
:ref:`fullexample-label`.

The central loop traversal concept in RAJA is a ``forall`` method, which
encapsulates loop execution details allowing the loop to be run in many
different ways without changing the loop code.

We will use a simple daxpy operation to walk through how to write a 
RAJA kernel and execute it using different programming model back-ends. 
A traditional C-style loop version of daxpy would look something like this:

.. code-block:: cpp

  double* a = new double[1000];
  double* b = new double[1000];

  // Initialize a and b...

  double c = 3.14159;

  for (int i = 0; i < 1000; i++) {
    a[i] += b[i] * c;
  }

This loop executes sequentially, iterating over the range of ``i``
values [0, 999] one after the other.

The RAJA form of this sequential loop replaces the ``for`` loop 
with a call to a RAJA ``forall`` method:

.. code-block:: cpp

  // Initialize a, b, c as before...

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });

The data allocation and loop body are exactly the same as the original code.
The ``RAJA::forall`` method takes as arguments the loop bounds in a
``RAJA::RangeSegment`` object and a lambda function containing the loop body. 
The method is templated on an `execution policy`; the template specialization 
selects how the loop will run. Here, we use ``RAJA::seq_exec`` to run the loop 
iterations sequentially, in order, exactly like the original loop.

Of course, this isn't very exciting yet. You may be wondering why we are
doing this: writing a simple loop in a more complicated way so it runs
exactly the same as in its original form....

The reason is that for more complicated situations, RAJA provides mechanisms
that make it easy to run the loop with different programming model backends
and map loop iterations to different orderings and data layouts based on
hardware resources without changing the code as it appears in an application.

For example, since our example loop is data parallel (i.e., all
iterations are independent), we can run it in parallel by replacing the
execution policy. This version will run in parallel using OpenMP
multithreading:

.. code-block:: cpp

  RAJA::forall<RAJA::omp_parallel_for_exec>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });

This version will run on an NVIDIA GPU using CUDA::

  RAJA::forall<RAJA::cuda_exec>(RAJA::RangeSegment(0, 1000), [=] (int i) {
    a[i] += b[i] * c;
  });

Note that we have assumed that the data arrays on the GPU device have been
allocated and initialized properly.

Also, these versions require RAJA to be built with OpenMP and CUDA
enabled, respectively.


.. _fullexample-label:

--------------------
Full example code
--------------------

If you want to run the example yourself, the complete code is located
in the file ``RAJA/examples/ex0-daxpy.cpp``. 

After building RAJA, with the options you select, it can be run using
the executable located at: ``<build-dir>/examples/binex0-daxpy``.
