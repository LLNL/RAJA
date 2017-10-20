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


####
RAJA
####

RAJA is a collection of C++ software abstractions, in development at Lawrence
Livermore National Laboratory (LLNL), that enable architecture portability for
High Performance Computing (HPC) applications. RAJA has two main goals: 

#. To make existing C++ applications portable with minimal disruption while maintaining single source computational kernels.
#. To provide a systematic programming model for new applications so that they are portable from inception.

Many HPC applications must execute with high performance across a diverse 
space of computer architectures including: Mac and Windows laptops,
sizable clusters comprised of multicore commodity processors, and large-scale 
supercomputers with advanced heterogeneous node architectures that combine 
cutting edge CPU and accelerator (e.g., GPU) processors. Profitably exposing
fine-grained parallelism on varied and disruptive architecture trends present 
significant challenges to developers of large-scale HPC applications. This
is especially true at US Department of Energy (DOE) laboratories where large
investments have been made over decades in scalable MPI-only applications 
that remain in service over many generations of architectures. Preserving 
developer and user productivity requires the ability to maintain single- 
source high performance code bases, while porting applications to new 
architectures. RAJA is one C++-based programming model abstraction layer 
developed to meet the performance portability challenge.

RAJA provides portable abstractions for single and nested loops, reductions,
scans, atomic operations, data layouts and views, iteration spaces, etc.
Currently supported execution policies/programming model backends include:
sequential, SIMD, CUDA, OpenMP multithreading and target offload, and Threading
Building Blocks.

RAJA uses standard C++11 -- C++ is the predominant programming language in
many LLNL applications. RAJA is rooted in a perspective based on decades
of experience working on production mesh-based multiphysics applications at 
LLNL. Another goal of RAJA is to enable application developers to specialize 
RAJA concepts for different code implementation patterns and C++ usage, 
since data structures and algorithms vary widely across applications.

RAJA helps developers insulate application loops from underlying 
architecture and programming model-specific implementation details. Loop 
bodies and traversals are decoupled via C++ lambda functions (loop bodies) 
and C++ templates (traversal methods). This approach promotes tuning loop 
patterns rather than individual loops, and makes it straightforward to
parametrize loop execution policies so an application can be compiled in 
a specific configuration suitable to a given architecture. 

If you are new to RAJA, check out our :doc:`getting_started` Guide.

See the :doc:`features` for detailed documentation for each feature.

If you are interested in keeping up with RAJA development and communicating
with developers and users, please join our `Google Group
<https://groups.google.com/forum/#!forum/raja-users>`_.

If you have questions, find a bug, or have ideas about expanding the
functionality or applicability of RAJA and are interested in contributing
to its development, please do not hesitate to contact us. We are always
interested in improving RAJA and exploring new ways to use it.

.. tincture::
   :maxdepth: 2
   :caption: Basics

   getting_started
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: Reference

   advanced_config
   features
   plugins

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing
