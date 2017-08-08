.. _getting_started:

===============
Getting Started
===============

This page will show you how to quickly get up and running with RAJA.

------------
Requirements
------------

The primary requirement for RAJA is a compiler that supports C++11.
Additional features have extra requirements detailed in
:doc:`advanced_config`. Specific software versions for configuring and
building are:

- C++ compiler with C++11 support
- CMake 3.2 or greater

------------
Installation
------------

RAJA is hosted on `GitHub <https://github.com/LLNL/RAJA>`_.
To clone the repo into your local working space, use the command:

.. code-block:: bash

   $ git clone https://github.com/LLNL/RAJA.git 


^^^^^^^^^^^^^
Building RAJA
^^^^^^^^^^^^^

RAJA uses CMake to handle builds. Configuration looks like:

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../

.. warning:: Builds must be out-of-source
             RAJA does not allow building in source, so you must create a
             build directory.

CMake will provide output about the compiler that has been detected, and
what features are discovered. Some features, like OpenMP, will be enabled
if they are discovered. For more advanced configuration options, please
see :doc:`advanced_config`.

Once Cmake has completed, RAJA can be compiled using Make:

.. code-block:: bash

  $ make

To install RAJA, run:

.. code-block:: bash

  $ make install

The header files required will be copied to the `include` directory, and the RAJA library will be installed in `lib`.

-----------
Basic Usage
-----------

Let's take a quick tour through RAJA's most important features. A complete
listing that you can try and compile is included at the bottom of the
page. The most important RAJA concept is the `forall` method, which
encapsulates a loop to allow it to be executed in many different ways.

We will use a daxpy example to walk through how you can write a simple
RAJA kernel. In regular C or C++ code, the loop would look something like this:

.. code-block:: cpp
  
  double* a = /* ... */
  double* b = /* ... */
  double c = 3.14159;

  for (int i = 0; i < len; i++) {
    a[i] += b[i] * c;
  }

This loop would execute sequentially, with the variable i taking values starting at 0. To write this loop using RAJA, we would replace the regular for loop with a call to the forall function:

.. code-block:: cpp

  double* a = /* ... */
  double* b = /* ... */
  double c = 3.14159;

  RAJA::forall<RAJA::seq_exec>(0, len, [=] (int i) {
    a[i] += b[i] * c;
  });

The data allocation and loop body are exactly the same as the regular C++ code. However, we call the RAJA::forall function




To build and run this example, you will need to pass the include directory and link against the RAJA library:



.. code-block:: bash

  $ make -I/path/to/install/include -std=c++11 exmaple.cpp 
