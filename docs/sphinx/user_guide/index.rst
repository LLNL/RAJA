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


################
RAJA User Guide
################

RAJA is a software library containing a collection of C++ abstractions, under 
development at Lawrence Livermore National Laboratory (LLNL). The RAJA 
abstractions enable architecture portability for High Performance Computing 
(HPC) applications. RAJA has two main goals: 

#. To make C++ applications portable with minimal disruption to existing algorithms and data structures while maintaining single source computational kernels.
#. To provide a systematic programming model for new applications so that they are portable from inception.

=============================
Background and Motivation
=============================

Many HPC applications must execute with high performance across a diverse 
range of computer architectures including: Mac and Windows laptops,
sizable clusters comprised of multicore commodity processors, and large-scale 
supercomputers with advanced heterogeneous node architectures that combine 
cutting edge CPU and accelerator (e.g., GPU) processors. Exposing fine-grained 
parallelism in a portable, high performance manner on varied and 
potentially disruptive architectures presents significant challenges to 
developers of large-scale HPC applications. This is especially true at US 
Department of Energy (DOE) laboratories where large investments have been made 
over decades in highly-scalable MPI-only applications that must remain in 
service over multiple platform generations. Preserving developer and user 
productivity requires the ability to maintain single-source application 
source code bases that can be readily ported to new architectures. RAJA is 
one C++-based programming model abstraction layer that can help to meet this 
performance portability challenge.

RAJA provides portable abstractions for singly-nested and multiply-nested 
loops -- as well as a variety of loop transformations, reductions,
scans, atomic operations, data layouts and views, iteration spaces, etc.
Currently supported execution policies for different programming model 
back-ends include: sequential, SIMD, CUDA, OpenMP multi-threading and target 
offload. Intel Threading Building Blocks (TBB) and OpenACC support is 
available but should be considered experimental at this point.

RAJA uses standard C++11 -- C++ is the predominant programming language in
many LLNL applications. RAJA requirements and design are rooted in a 
perspective based on decades of experience working on production mesh-based 
multiphysics applications at LLNL. An important goal of RAJA is to enable 
application developers to specialize RAJA concepts for different code 
implementation patterns and C++ usage, since data structures and algorithms 
vary widely across applications.

RAJA helps developers insulate application loop kernels from underlying 
architecture and programming model-specific implementation details. Loop 
bodies and traversals are decoupled via C++ lambda expressions (loop bodies) 
and C++ templates (loop traversal methods). This approach promotes tuning loop 
patterns rather than individual loops, and makes it relatively straightforward 
to parametrize execution policy types so that a large application can be 
compiled in a specific configuration suitable to a given architecture. 

================================
Interacting with the RAJA Team
================================

If you are interested in keeping up with RAJA development and communicating
with developers and users, please join our `Google Group
<https://groups.google.com/forum/#!forum/raja-users>`_.

If you have questions, find a bug, or have ideas about expanding the
functionality or applicability of RAJA and are interested in contributing
to its development, please do not hesitate to contact us. We are always
interested in improving RAJA and exploring new ways to use it. A brief 
description of how the RAJA team operates can be found in 
:ref:`contributing-label`.

=============================
What's In This Guide
=============================

If you have some familiarity with RAJA, and want to get up and running quickly, 
check out :ref:`getting_started-label`. This guide contains information 
about accessing the RAJA code, building it, and basic RAJA usage.

If you are completely new to RAJA, please check out the :ref:`tutorial-label`.
It contains some discussion of essential C++ concepts and will walk you 
through a sequence of code examples that illustrate many of the key RAJA
features.

See :ref:`features-label` for a complete, high-level description of RAJA 
features (akin to a reference guide).

.. toctree::
   :maxdepth: 2

   getting_started
   tutorial
   features
   plugins
   config_options
   contributing
   raja_license
