.. ############################################################################
.. # Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory
.. #
.. # LLNL-CODE-666778
.. #
.. # All rights reserved.
.. #
.. # This file is part of Conduit.
.. #
.. # For details, see: http://software.llnl.gov/conduit/.
.. #
.. # Please also read conduit/LICENSE
.. #
.. # Redistribution and use in source and binary forms, with or without
.. # modification, are permitted provided that the following conditions are met:
.. #
.. # * Redistributions of source code must retain the above copyright notice,
.. #   this list of conditions and the disclaimer below.
.. #
.. # * Redistributions in binary form must reproduce the above copyright notice,
.. #   this list of conditions and the disclaimer (as noted below) in the
.. #   documentation and/or other materials provided with the distribution.
.. #
.. # * Neither the name of the LLNS/LLNL nor the names of its contributors may
.. #   be used to endorse or promote products derived from this software without
.. #   specific prior written permission.
.. #
.. # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
.. # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
.. # ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
.. # LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
.. # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
.. # DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
.. # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
.. # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
.. # STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
.. # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
.. # POSSIBILITY OF SUCH DAMAGE.
.. #
.. ############################################################################

.. _building_with_uberenv:

Uberenv
~~~~~~~~~~~~~~~

**Uberenv** automates using `Spack <ttp://www.spack.io>`_ to build and deploy software.

Many projects leverage `Spack <ttp://www.spack.io>`_ to help build the software dependencies needed to develop and deploy their projects on HPC systems. Uberenv is a python script that helps automate using Spack to build
third-party dependencies for development and to deploy Spack packages.

Uberenv was released as part of Conduit (https://github.com/LLNL/conduit/). It is included in-source in several projects. The
https://github.com/llnl/uberenv/ repo is used to hold the latest reference version of Uberenv.


uberenv.py
~~~~~~~~~~~~~~~~~~~~~

``uberenv.py`` is a single file python script that automates fetching Spack, building and installing third party dependencies, and can optionally install packages as well.  To automate the full install process, ``uberenv.py`` uses a target Spack package along with extra settings such as Spack compiler and external third party package details for common HPC platforms.

``uberenv.py`` is included directly in a project's source code repo in the folder: ``scripts/uberenv/``
This folder is also used to store extra Spack and Uberenv configuration files unique to the target project. ``uberenv.py`` uses a ``project.json`` file to specify project details, including the target Spack package name and which Spack repo is used.  Conduit's source repo serves as an example for Uberenv and Spack configuration files, etc:

https://github.com/LLNL/conduit/tree/master/scripts/uberenv


``uberenv.py`` is developed by LLNL in support of the `Ascent <http://github.com/alpine-dav/ascent/>`_, Axom, and `Conduit <https://github.com/llnl/conduit>`_  projects.


Command Line Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build configuration
-------------------

``uberenv.py`` has a few options that allow you to control how dependencies are built:

 ======================= ============================================== ================================================
  Option                  Description                                    Default
 ======================= ============================================== ================================================
  ``--prefix``            Destination directory                          ``uberenv_libs``
  ``--spec``              Spack spec                                     linux: **%gcc**
                                                                         osx: **%clang**
  ``--spack-config-dir``  Folder with Spack settings files               linux: (empty)
                                                                         osx: ``scripts/uberenv/spack_configs/darwin/``
  ``-k``                  Ignore SSL Errors                              **False**
  ``--install``           Fully install target, not just dependencies    **False**
  ``--run_tests``         Invoke tests during build and against install  **False**
  ``--project-json``      File for project specific settings             ``project.json``
 ======================= ============================================== ================================================

The ``-k`` option exists for sites where SSL certificate interception undermines fetching
from github and https hosted source tarballs. When enabled, ``uberenv.py`` clones Spack using:

.. code:: bash

    git -c http.sslVerify=false clone https://github.com/llnl/spack.git

And passes ``-k`` to any Spack commands that may fetch via https.


Default invocation on Linux:

.. code:: bash

    python scripts/uberenv/uberenv.py --prefix uberenv_libs \
                                      --spec %gcc

Default invocation on OSX:

.. code:: bash

    python scripts/uberenv/uberenv.py --prefix uberenv_libs \
                                      --spec %clang \
                                      --spack-config-dir scripts/uberenv/spack_configs/darwin/


Use the ``--install`` option to install the target package (not just its development dependencies):

.. code:: bash

    python scripts/uberenv/uberenv.py --install


If the target Spack package supports Spack's testing hooks, you can run tests during the build process to validate the build and install, using the ``--run_tests`` option:

.. code:: bash

    python scripts/uberenv/uberenv.py --install \
                                      --run_tests

For details on Spack's spec syntax, see the `Spack Specs & dependencies <http://spack.readthedocs.io/en/latest/basic_usage.html#specs-dependencies>`_ documentation.


Uberenv looks for configuration yaml files under ``scripts/uberenv/spack_config/{platform}`` or you can use the **--spack-config-dir** option to specify a directory with compiler and packages yaml files to use with Spack. See the `Spack Compiler Configuration <http://spack.readthedocs.io/en/latest/getting_started.html#manual-compiler-configuration>`_
and `Spack System Packages
<http://spack.readthedocs.io/en/latest/getting_started.html#system-packages>`_
documentation for details.

.. note::
    The bootstrapping process ignores ``~/.spack/compilers.yaml`` to avoid conflicts
    and surprises from a user's specific Spack settings on HPC platforms.

When run, ``uberenv.py`` checkouts a specific version of Spack from github as ``spack`` in the
destination directory. It then uses Spack to build and install the target packages' dependencies into
``spack/opt/spack/``. Finally, the target package generates a host-config file ``{hostname}.cmake``, which is
copied to destination directory. This file specifies the compiler settings and paths to all of the dependencies.


Project configuration
---------------------

Part of the configuration can also be addressed using a json file. By default, it is named ``project.json`` and some settings can be overridden on command line:

 ==================== ========================== ================================================ =======================================
  Setting              Option                     Description                                      Default
 ==================== ========================== ================================================ =======================================
  package_name         ``--package-name``         Spack package name                               **None**
  package_version      **None**                   Spack package version                            **None**
  package_final_phase  ``--package-final-phase``  Controls after which phase Spack should stop     **None**
  package_source_dir   ``--package-source-dir``   Controls the source directory Spack should use   **None**
  spack_url            **None**                   Url where to download Spack                      ``https://github.com/spack/spack.git``
  spack_commit         **None**                   Spack commit to checkout                         **None**
  spack_activate       **None**                   Spack packages to activate                       **None**
 ==================== ========================== ================================================ =======================================


Optimization
------------

``uberenv.py`` also features options to optimize the installation

 ==================== ============================================== ================================================
  Option               Description                                    Default
 ==================== ============================================== ================================================
  ``--mirror``         Location of a Spack mirror                     **None**
  ``--create-mirror``  Creates a Spack mirror at specified location   **None**
  ``--upstream``       Location of a Spack upstream                   **None**
 ==================== ============================================== ================================================


Project Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few notes on using ``uberenv.py`` in a new project:

* For an example of how to craft a ``project.json`` file a target project, see: `Conduit's project.json file <https://github.com/LLNL/conduit/tree/master/scripts/uberenv/project.json>`_

* ``uberenv.py`` hot copies ``packages`` to the cloned Spack install, this allows you to easily version control any Spack package overrides necessary


