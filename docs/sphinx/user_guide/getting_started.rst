.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##


.. _getting_started-label:

*************************
Getting Started With RAJA
*************************

This section is intended to help get you up and running with RAJA quickly.

============
Requirements
============

Accessing various programming model back-ends in RAJA requires that they be 
supported by the compiler you chose. Available options and how to enable or 
disable them are described in :ref:`configopt-label`. To build and use RAJA in 
its simplest form requires:

- C++ compiler with C++11 support
- `CMake <https://cmake.org/>`_ version 3.9 or greater.
- The `camp <https://github.com/LLNL/camp>`_ C++ meta-programming library.
  This is included with RAJA as a Git submodule. Instructions for using
  an externally-supplied version of the camp library are below.


==================
Get the Code
==================

The RAJA project is hosted on `GitHub <https://github.com/LLNL/RAJA>`_.
To get the code, clone the repository into a local working space using
the command::

   $ git clone --recursive https://github.com/LLNL/RAJA.git

The ``--recursive`` argument above is needed to pull in other projects
RAJA depends on as Git *submodules*. Currently, RAJA submodule dependencies 
are:

- `BLT build system <https://github.com/LLNL/blt>`_
- `Camp portable utility library <https://github.com/LLNL/camp>`_
- `NVIDIA CUB <https://github.com/NVlabs/cub>`_

You probably don't need to know much about these other projects to start
using RAJA. But, if you want to know more about them, click on the links above.

After running the clone command, a copy of the RAJA repository will reside in
a ``RAJA`` subdirectory where you ran the clone command. You will be on the 
``develop`` branch of RAJA, which is our default branch.

If you forget to pass the ``--recursive`` argument to the ``git clone``
command, you can type the following commands after cloning::

  $ cd RAJA
  $ git submodule init
  $ git submodule update --recursive

Either way, the end result is the same and you should be good to go.

.. note:: Any time you switch branches in RAJA, you may need to re-run the
          ``git submodule update --recursive`` command to set the Git 
          submodule versions to what is used by the new branch. To see if 
          any submodule is not iin sync, execute the command ``git status``

==================
Build and Install
==================

Building and installing RAJA can be very easy or more involved, depending
on which features you want to use and how well you understand how to use
your system.

.. _getting_started_building-label:

--------------
Building RAJA
--------------

RAJA uses CMake to configure a build. A basic configuration looks like::

  $ mkdir build-dir && cd build-dir
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../

.. note:: * RAJA requires a minimum CMake version of 3.9.
          * Builds must be *out-of-source*.  RAJA does not allow building in
            the source directory, so you must create a build directory and
            run CMake in it.

When you run CMake, it will provide output about the compiler and other 
features that are being used. Some RAJA features, like OpenMP 
support are enabled, by default, if they are discovered. For a complete 
summary of configuration options and their defaults, please 
see :ref:`configopt-label`.

After CMake successfully completes, you compile RAJA by executing the ``make``
command in the build directory; i.e.,::

  $ cd build-dir
  $ make

If you have access to a multicore system you can compile in parallel by running
``make -j`` (to build with all available cores) or ``make -j N`` to build using
N cores.

.. note:: RAJA is configured to build its tests, examples, and tutorial 
          exercises by default. If you do not disable them with the appropriate
          CMake option, you can run them after the build completes to check if 
          things were compiled properly. The easiest way to runt the tests,
          for example, is to type the following in the build directory::

          $ make test

          after the build completes.

          You can also run individual tests by invoking individual test 
          executables directly. They live in in the ``test`` directory
          and subdirectories within it. RAJA tests use the 
          `Google Test framework <https://github.com/google/googletest>`_, 
          so you can also run tests via Google Test commands.

.. note:: It is very important to note that the version of Googletest that
          is used in RAJA version v0.11.0 or newer requires CUDA version 
          9.2.x or newer when compiling with nvcc. Thus, if you build
          RAJA with CUDA enabled and want to also enable RAJA tests, you
          must use CUDA version 9.2.x or newer. Earlier versions of RAJA
          allow older versions of CUDA to be used.

.. _build-external-tpl-label:

.. note:: You may use externally-supplied versions of the camp and cub 
          libraries with RAJA if you wish. To do so, pass the following 
          options to CMake:
            * External camp: -DEXTERNAL_CAMP_SOURCE_DIR=<camp dir name>
            * External cub: -DENABLE_EXTERNAL_CUB=On -DCUB_DIR=<cub dir name> 

.. note:: RAJA requires version 3.5 or newer of the rocm software stack to 
          use the Hip back-end, which supports AMD GPUs.

----------------
Installing RAJA
----------------

To install RAJA as a library, run the following command in your build 
directory after compiling::

  $ make install

This will copy RAJA header files to the ``include`` directory and the RAJA
library will be installed in the ``lib`` directory you specified using the
``-DCMAKE_INSTALL_PREFIX`` CMake option.


======================
Learning to Use RAJA
======================

If you want to view and run a very simple RAJA example code, a good place to
start is located in the file: ``RAJA/examples/tut_daxpy.cpp``. After building 
RAJA with the options you select, the executable for this code will reside 
in the file: ``<build-dir>/examples/bin/tut_daxpy``. Simply type the name
of the executable in your build directory to run it; i.e.,::

  $ ./examples/bin/daxpy 

The ``RAJA/examples`` directory contains a variety of other RAJA example codes 
you can run and experiment with. Many of these are discussed in
:ref:`tutorial-label`. The ``RAJA/exercises`` subdirectories contain other
example codes from RAJA tutorials. 
The `RAJA Tutorials <https://github.com/LLNL/RAJA-tutorials>`_ GitHub project 
contains slide presentations that accompany the tutorial code examples.

For an overview of all the main RAJA features, see :ref:`features-label`.
A full tutorial with a variety of examples showing how to use RAJA features
can be found in :ref:`tutorial-label`.
