.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

This section will help get you up and running with RAJA quickly.

============
Requirements
============

The primary requirement for using RAJA is a C++11 compliant compiler.
Accessing various programming model back-ends requires that they be supported
by the compiler you chose. Available options and how to enable or disable 
them are described in :ref:`configopt-label`. To build and use RAJA in its 
simplest form requires:

- C++ compiler with C++11 support
- `CMake <https://cmake.org/>`_ version 3.8.2 or greater for most back-ends, and version 3.9 or greater for CUDA.


==================
Get the Code
==================

The RAJA project is hosted on `GitHub <https://github.com/LLNL/RAJA>`_.
To get the code, clone the repository into a local working space using
the command::

   $ git clone --recursive https://github.com/LLNL/RAJA.git

The ``--recursive`` argument above is needed to pull in other projects
that we use as Git *submodules*. Currently, we have only two:

- `BLT build system <https://github.com/LLNL/blt>`_
- `NVIDIA CUB <https://github.com/NVlabs/cub>`_

You probably don't need to know much about either of these projects to start
using RAJA. But, if you want to know more, click on the links above.

After running the clone command, a copy of the RAJA repository will reside in
a ``RAJA`` subdirectory where you ran the clone command. You will be on the 
``develop`` branch of RAJA, which is our default branch.

If you forget to pass the ``--recursive`` argument to the ``git clone``
command, you can type the following commands after cloning::

  $ cd RAJA
  $ git submodule init
  $ git submodule update

Either way, the end result is the same and you should be good to go.

.. note:: Any time you switch branches in RAJA, you need to re-run the
          'git submodule update' command to set the Git submodules to
          what is used by the new branch.

==================
Build and Install
==================

Building and installing RAJA can be very easy or more complicated, depending
on which features you want to use and how well you understand how to use
your system.

--------------
Building RAJA
--------------

RAJA uses CMake to configure a build. A basic configuration looks like::

  $ mkdir build-dir && cd build-dir
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../

.. note:: Builds must be *out-of-source*.  RAJA does not allow building in
          the source directory, so you must create a build directory.

When you run CMake, it will provide output about the compiler that has been 
found and which features are discovered. Some RAJA features, like OpenMP 
support are enabled if they are discovered. For a complete summary of 
configuration options, please see :ref:`configopt-label`.

After CMake successfully completes, you compile RAJA by executing the ``make``
command in the build directory; i.e.,::

  $ cd build-dir
  $ make

If you have access to a multi-core system you can compile in parallel by running
``make -j`` (to build with all available cores) or ``make -j N`` to build using
N cores.

.. note:: RAJA is configured to build its unit tests by default. If you do not
          disable them with the appropriate CMake option, you can run them
          after the build completes to check if everything compiled properly.
          The easiest way to do this is to type::

          $ make test

          after the build completes.

          You can also run individual tests by invoking individual test 
          executables directly. They live in subdirectories in the ``test`` 
          directory. RAJA tests use the 
          `Google Test framework <https://github.com/google/googletest>`_, 
          so you can also run tests via Google Test commands.


----------------
Installing RAJA
----------------

To install RAJA as a library, run the following command in your build 
directory::

  $ make install

This will copy RAJA header files to the ``include`` directory and the RAJA
library will be installed in the ``lib`` directory you specified using the
``-DCMAKE_INSTALL_PREFIX`` CMake option.


======================
Learning to Use RAJA
======================

If you want to view and run a very simple RAJA example code, a good place to
start is located in the file: ``RAJA/examples/daxpy.cpp``. After building 
RAJA with the options you select, the executable for this code will reside 
in the file: ``<build-dir>/examples/bin/daxpy``. Simply type the name
of the executable in your build directory to run it; i.e.,::

  $ ./examples/bin/daxpy 

The ``RAJA/examples`` directory also contains many other RAJA example codes 
you can run and experiment with.

For an overview of all the main RAJA features, see :ref:`features-label`.
A full tutorial with a variety of examples showing how to use RAJA features
can be found in :ref:`tutorial-label`.
