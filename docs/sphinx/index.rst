.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For details and restrictions, please read the README-license.txt file.
.. ##


***********************
What is RAJA?
***********************

RAJA is a collection of C++ software abstractions, being developed at
Lawrence Livermore National Laboratory (LLNL), that enable architecture
portability for HPC applications. The overarching goals of RAJA are to:

  * Make existing (production) applications *portable with minimal disruption*
  * Provide a model for new applications so that they are portable from
    inception.

RAJA uses standard C++11 -- C++ is the predominant programming language in
which many LLNL codes are written. RAJA is rooted in a perspective based on 
substantial experience working on production mesh-based multiphysics 
applications at LLNL. Another goal of RAJA is to enable application developers 
to adapt RAJA concepts and specialize them for different code implementation 
patterns and C++ usage, since data structures and algorithms vary widely 
across applications.

RAJA shares goals and concepts found in
other C++ portability abstraction approaches, such as
`Kokkos <https://github.com/kokkos/kokkos>`_
and `Thrust <https://developer.nvidia.com/thrust>`_. 
However, it includes concepts that are absent in other models and which are 
fundamental to LLNL codes. 

It is important to note that RAJA is very much a work-in-progress.
The community of researchers and application developers at LLNL that are
actively contributing to it and developing new capabilities is growing.
The publicly-released version contains only core pieces of RAJA as they
exist today. While the basic interfaces are fairly stable, the implementation
of the underlying concepts is being refined. Additional features will appear
in future releases.

RAJA lives in a `github repository <https://github.com/llnl/raja>`_.
To clone the repo, use the command:

.. code-block:: sh

   $ git clone https://github.com/llnl/raja.git 

Then, you can build RAJA like any other CMake project, provided you have a C++
compiler that supports the C++11 standard. The simplest way to build the code
is to do the following in the top-level RAJA directory (in-source builds are 
not allowed!):

.. code-block:: sh

    $ mkdir build
    $ cd build
    $ cmake ../
    $ make

For details about RAJA configuration and build options, please see 
:doc:`config_build`.

If you are interested in keeping up with RAJA development and communicating 
with developers and users, please join our mailing list over at Google Groups -
`RAJA Google Group <https://groups.google.com/forum/#!forum/raja-users>`_.

If you have questions, find a bug, or have ideas about expanding the
functionality or applicability of RAJA and are interested in contributing
to its development, please do not hesitate to contact us. We are always
interested in improving RAJA and exploring new ways to use it.

.. seealso::

   More detailed discussion of RAJA, including its use in proxy apps included
   in the RAJA source distribution can be found in these reports:

   `The RAJA Portability Layer: Overview and Status (2014) <_static/RAJAStatus-09.2014_LLNL-TR-661403.pdf>`_

   `RAJA Overview (extracted from ASC Tri-lab Co-design Level 2 Milestone Report 2015) <_static/RAJAOverview-Trilab-09.2015_LLNL-TR-677453.pdf>`_

   Newer reports and other material will appear here over time.


**Contents:**

.. toctree::
   :maxdepth: 2

   primer
   basic_examples
   advanced_examples
   config_build
   dev_guide
   future
   raja_license
