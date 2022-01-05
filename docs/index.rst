.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##


################
RAJA 
################

RAJA is a software library of C++ abstractions, developed at Lawrence Livermore
National Laboratory (LLNL), that enable architecture and programming model
portability for high performance computing (HPC) applications. RAJA has two
main goals:

#. To enable application portability with manageable disruption to existing algorithms and programming styles.
#. To achieve performance comparable to using common programming models (e.g., OpenMP, CUDA, etc.) directly.

RAJA targets portable, parallel loop execution by providing building blocks
that extend the generally-accepted *parallel for* idiom.

=============================
Background and Motivation
=============================

Many HPC applications must achieve high performance across a diverse
range of computer architectures including: Mac and Windows laptops,
parallel clusters of multicore commodity processors, and large-scale 
supercomputers with advanced heterogeneous node architectures that combine 
cutting edge CPU and accelerator (e.g., GPU) processors. Exposing fine-grained 
parallelism in a portable, high performance manner on varied and 
potentially disruptive architectures presents significant challenges to 
developers of large-scale HPC applications. This is especially true at US 
Department of Energy (DOE) laboratories where, for decades, large investments 
have been made in highly-scalable MPI-only applications that have been in 
service over multiple platform generations. Often, maintaining developer and 
user productivity requires the ability to build single-source application 
source code bases that can be readily ported to new architectures. RAJA is 
one C++ abstraction layer that helps address this performance portability 
challenge.

RAJA provides portable abstractions for simple and complex loops -- as well 
reductions, scans, atomic operations, sorts, data layouts, views, and loop 
iteration spaces, as well as compile-time loop transformations. Features
are continually growing as new use cases arise due to expanding user adoption.

RAJA uses standard C++11 -- C++ is the programming language model of choice
for many HPC applications. RAJA requirements and design are rooted in a 
decades of developer experience working on production mesh-based 
multiphysics applications. An important RAJA requirement is that
application developers can specialize RAJA concepts for different code
implementation patterns and C++ usage, since data structures and algorithms
vary widely across applications.

RAJA helps developers insulate application loop kernels from underlying 
architecture and programming model-specific implementation details. Loop 
bodies and loop execution are decoupled using C++ lambda expressions 
(loop bodies) and C++ templates (loop execution methods). This approach 
promotes the perspective that application developers should focus on tuning 
loop patterns rather than individual loops as much as possible. RAJA makes it 
relatively straightforward to parameterize an application using execution 
policy types so that it can be compiled in a specific configuration suitable 
to a given architecture.

RAJA support for various execution back-ends is the result of collaborative
development between the RAJA team and academic and industrial partners.
Currently available execution back-ends include: 
sequential, 
`SIMD <https://en.wikipedia.org/wiki/SIMD>`_, 
`Threading Building Blocks (TBB) <https://github.com/oneapi-src/oneTBB>`_,
`NVIDIA CUDA <https://developer.nvidia.com/about-cuda>`_, 
`OpenMP <https://www.openmp.org>`_ CPU multithreading and target offload, and
`AMD HIP <https://github.com/ROCm-Developer-Tools/HIP>`_. Sequential,
CUDA, OpenMP CPU multithreading, and HIP execution are supported for all
RAJA features. Sequential, OpenMP CPU multithreading, and CUDA
are considered the most developed at this point as these have been our primary
focus up to now. Those back-ends are used in a wide variety of production 
applications. OpenMP target offload and TBB back-ends do not support
all RAJA features and should be considered experimental.

=================================
Git Repository and Issue Tracking
=================================

The main interaction hub for RAJA is on `GitHub <https://github.com/LLNL/RAJA>`_

================================
Communicating with the RAJA Team
================================

If you have questions, find a bug, have ideas about expanding the
functionality or applicability, or wish to contribute
to RAJA development, please do not hesitate to contact us. We are always
interested in improving RAJA and exploring new ways to use it. 

The best way to communicate with us is via our email list: ``raja-dev@llnl.gov``

You are also welcome to join our `Google Group <https://groups.google.com/forum/#!forum/raja-users>`_

A brief description of how to start a contribution to RAJA can be found in
:ref:`contributing-label`.

=======================
RAJA User Documentation
=======================

  * :doc:`RAJA User Guide <sphinx/user_guide/index>`
  * `RAJA Tutorials Repo <https://github.com/LLNL/RAJA-tutorials>`_
  * `Source Documentation <doxygen/html/index.html>`_

============================
RAJA Developer Documentation
============================

  * :doc:`RAJA Developer Guide <sphinx/dev_guide/index>`

======================================================
RAJA Copyright and License Information
======================================================

Please see :ref:`raja-copyright`.

.. toctree::
   :hidden: 
   :caption: User Documentation

   sphinx/user_guide/index
   doxygen/html/index

.. toctree::
   :hidden: 
   :caption: Developer Documentation

   sphinx/dev_guide/index
   sphinx/raja_license
