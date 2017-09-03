.. _getting_started:

===============
Getting Started
===============

This page will show you how to quickly get up and running with RAJA.

------------
Requirements
------------

The primary requirement for RAJA is a compiler that supports C++11.  Additional
features have extra requirements that are detailed in :doc:`advanced_config`.
Specific software versions for configuring and building are:

- C++ compiler with C++11 support
- `CMake <https://cmake.org/>`_ 3.3 or greater

------------
Installation
------------

RAJA is hosted on `GitHub <https://github.com/LLNL/RAJA>`_.
To clone the repo into your local working space, use the command:

.. code-block:: bash

   $ git clone https://github.com/LLNL/RAJA.git 

Once this command has run, the ``develop`` branch of RAJA will be cloned into
the ``RAJA`` directory.


^^^^^^^^^^^^^
Building RAJA
^^^^^^^^^^^^^

RAJA uses CMake to handle builds. Configuration looks like:

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../

.. warning:: Builds must be out-of-source.  RAJA does not allow building in
             source directory, so you must create a build directory.

CMake will provide output about the compiler that has been detected, and
what features are discovered. Some features, like OpenMP, will be enabled
if they are discovered. For more advanced configuration options, please
see the :doc:`advanced_config`.

Once CMake has completed, RAJA can be compiled using Make:

.. code-block:: bash

  $ make


^^^^^^^^^^^^^^^
Installing RAJA
^^^^^^^^^^^^^^^

To install RAJA, run:

.. code-block:: bash

  $ make install

The header files required will be copied to the ``include`` directory, and the
RAJA library will be installed in ``lib``.

-----------
Basic Usage
-----------

Let's take a quick tour through RAJA's most important features. A complete
listing that you can try and compile is included at the bottom of the
page. The most important RAJA concept is the ``forall`` method, which
encapsulates a loop to allow it to be executed in many different ways.

We will use a daxpy example to walk through how you can write a simple
RAJA kernel. In regular C or C++ code, the loop would look something like this:

.. code-block:: cpp
  
  double* a = new double[1000];
  double* b = new double[1000];

  // Initialize a and b...

  double c = 3.14159;

  for (int i = 0; i < 1000; i++) {
    a[i] += b[i] * c;
  }

This loop would execute sequentially, with the variable ``i`` taking values
starting at 0. To write this loop using RAJA, we would replace the regular
``for`` loop with a call to RAJA's ``forall`` function:

.. code-block:: cpp

  double* a = new double[1000];
  double* b = new double[1000];

  // Initialize a and b...

  double c = 3.14159;

  RAJA::forall<RAJA::seq_exec>(0, len, [=] (int i) {
    a[i] += b[i] * c;
  });

The data allocation and loop body are exactly the same as the regular C++ code.
The ``RAJA::forall`` function takes the beginning and end index arguments, and
a lambda function containing the loop body. The ``forall`` function is
templated on an `execution policy` that selects how the loop iterations are
scheduled to the different programming model backends and corresponding
hardware. In this case, we use ``RAJA::seq_exec`` to execute the loop
executions sequentially, in order, exactly like the basic ``for`` loop.

Since this loop is thread safe (each iteration is completely independent of any
other), we can run this in parallel using OpenMP just by replacing the
execution policy!

.. code-block:: cpp

  RAJA::forall<RAJA::omp_parallel_for_exec>(0, len, [=] (int i) {
    a[i] += b[i] * c;
  });

Of course, this requires a version of RAJA built with OpenMP. 

In case you want to try and run this yourself, here is a complete code listing:

.. code-block:: cpp
  
  #include "RAJA/RAJA.hpp"

  int main(int argc, char* argv[]) {
    double* a = new double[1000];
    double* b = new double[1000];

    double* c = 3.14159;

    for (int i = 0; i < 1000; i++) {
      a[i] = 1.0;
      b[i] = 2.0;
    }

    RAJA::forall<RAJA::seq_exec>(0, 1000, [=] (int i) {
      a[i] += b[i] * c;
    });

    return 0;
  }

To build and run this example, you will need to pass the include directory and
link against the RAJA library:

.. code-block:: bash

  $ make -I/path/to/install/include -std=c++11 example.cpp 

For more examples, you can check out our tutorial programs in the ``examples``
directory. These programs are explained in the :doc:`tutorial`.
