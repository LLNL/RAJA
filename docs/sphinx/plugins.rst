.. _plugins::

=======
Plugins
=======

RAJA provides parallel execution primitives without the need to subscribe to
any other components such as data management, however, through a plugin
mechanism we support additional components to provide additional functionality
and to make writing applications easier.

CHAI
----

Currently we only have one third-party plugin integrated in RAJA, CHAI. CHAI is
an array abstraction that uses RAJA's execution policies to inform data
movement operations. The data can be accessed inside any RAJA kernel, and
regardless of where that kernel executes, CHAI will make the data available.

To build RAJA with CHAI integration, first download and install CHAI. Please
see the CHAI documentation for details. Once CHAI has been installed, RAJA can
be configured with two additional arguments:

.. code-block:: bash

    $ cmake -DRAJA_ENABLE_CHAI=On -Dchai_DIR=/path/to/chai

Once RAJA has been built with CHAI support, applications can use CHAI's
ManangedArray class to transparently access data inside a RAJA kernel.

.. code-block:: cpp

  chai::ManagedArray<float> array(1000);

  RAJA::forall<RAJA::cuda_exec<16> >(0, 1000, [=] __device__ (int i) {
      array[i] = i * 2.0f;
  });

  RAJA::forall<RAJA::seq_exec>(0, 1000, [=] (int i) {
    std::cout << "array[" << i << "]  is " << array[i] << std::endl;
  });
