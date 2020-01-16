.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _plugins-label:

*******
Plugins
*******

RAJA provides a plugin mechanism to support optional components that provide
additional functionality to make writing applications easier. Currently, there
is only one library that provides a RAJA plugin: CHAI.

=======
CHAI
=======

RAJA provides abstractions for parallel execution, but does not support 
a memory model for managing data in heterogeneous memory spaces.
CHAI is an array abstraction that can be used to copy data transparently from 
one memory space to another as needed to run a RAJA-based kernel. 
The data can be accessed inside any RAJA kernel, and regardless of where 
that kernel executes, CHAI will make the data available.

To build CHAI with RAJA integration, you need to download and install CHAI with
the ``ENABLE_RAJA_PLUGIN`` option turned on.  Please see the `CHAI project
<https://github.com/LLNL/CHAI>` for details

After CHAI has been build with RAJA support enabled, applications can use CHAI
``ManangedArray`` objects to access data inside a RAJA kernel; for example,::

  chai::ManagedArray<float> array(1000);

  RAJA::forall<RAJA::cuda_exec<16> >(0, 1000, [=] __device__ (int i) {
      array[i] = i * 2.0f;
  });

  RAJA::forall<RAJA::seq_exec>(0, 1000, [=] (int i) {
    std::cout << "array[" << i << "]  is " << array[i] << std::endl;
  });

Here, the data held by ``array`` is allocated on the host CPU. Then, it is 
initialized on a CUDA GPU device. CHAI sees that the data lives on the CPU
and is needed in a GPU device data environment. So it copies the data from
CPU to GPU, making it available for access in the first RAJA kernel. Next, 
it is printed in the second kernel which runs on the CPU. So CHAI copies the 
data back to the host CPU. All necessary data copies are done
transparently on demand as needed for each kernel.
