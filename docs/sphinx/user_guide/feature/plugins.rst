.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _plugins-label:

========
Plugins
========

------------
About RAJA Plugins
------------

RAJA supports user-made plugins that may be loaded either at the time of compilation or during runtime. These two methods are not mutually exclusive, as plugins loaded dynamically can be run alongside plugins that are loaded statically.

------------
Using RAJA Plugins
------------

^^^^^^^^^^^
Quick Start Guide
^^^^^^^^^^^

^^^^^^^^^^^^^^^^^
Static vd Dynamic Loading
^^^^^^^^^^^^^^^^^

^^^^^^^^^^^
Further Details
^^^^^^^^^^^


------------
Building Plugins For RAJA
------------

^^^^^^^^^^^
Required Functions
^^^^^^^^^^^
The preLaunch and postLaunch functions are automatically called by RAJA before and after loop execution. This applies to RAJA's kernel and forall implementations.

* ``void preLaunch(PluginContext& p) {}`` - Will occur before kernel/forall execution.

* ``void postLaunch(PluginContext& p) {}`` - Will occur after kernel/forall execution.

* ``At least one method of loading the plugin, either statically or dynamically.``

^^^^^^^^^^^
Optional Functions
^^^^^^^^^^^
The init and finalize functions have standard implementations and thus are not needed in a user-made plugin. Init and finalize are never run by RAJA by default and are only run when the user makes a call to RAJA::util::init_plugin() or RAJA::util::finalize_plugin() respectively.

* ``void init(PluginOptions p) {}`` - Called by the user

* ``void finalize() {}``

^^^^^^^^^^^^^^^^^
Static Loading
^^^^^^^^^^^^^^^^^
If the plugin is to be linked to a project at compile time, adding the following one-liner will add the plugin to the RAJA PluginRegistry and will be loaded every time the compiled executable is run. This requires the plugin to be linked either in an ``#include`` statement within the project, or linked by compiler commands.::

  static RAJA::util::PluginRegistry::add<PluginName> P("Name", "Description");


^^^^^^^^^^^^^^^^^
Dynamic Loading
^^^^^^^^^^^^^^^^^
If the plugin is to be dynamically linked to a project during runtime, the RAJA Plugin API requires a few conditions to be met. The following must be true about the plugin, not necessarily of the project using it.

1. **The plugin must have following factory function.** This will return a pointer to an instance of your plugin, and thanks to the ``extern "C"``, a dynamically linked project will be able to access this function as well as the instance it returns.::

  extern "C" RAJA::util::PluginStrategy *getPlugin ()
  {
    return new MyPluginName;
  }
  

2. **The plugin must be compiled to be a shared object with a .so extension.** A simple example containing required flags would be: ``g++ plugin.cpp -lRAJA -fopenmp -fPIC -shared -o plugin.so``. At the moment, RAJA will only attempt to load files with .so extensions. It's worth noting why these flags (or their equivalents) are important. ``-lRAJA -fopenmp`` are the standard flags for compiling the RAJA library. For the purposes of dynamic linking, ``-fPIC`` tells the compiler to produce *position independent code*, which is needed to prevent conflicts in the address space of the executable. ``-shared`` will let the compiler know that you want the resulting object file to be shared, removing the need for a *main* as well as giving dynamically linked executables access to functions flagged with ``Extern "C"``.

3. **The RAJA_PLUGINS environment variable has been set, or the user has made a call to ``RAJA::util::init_plugins("path")`` with a path specified to either a directory or a .so file.** It's worth noting that these are not mutually exclusive, RAJA will look for plugins from the environment variable on program startup and new plugins may be loaded after that using ``init_plugins``.


^^^^^^^^^^^^^^^^^
Example Implementation
^^^^^^^^^^^^^^^^^

The following is an example plugin that simply will print out the number of times a kernel has been launched and has the ability to be loaded either statically or dynamically.::

  #include "RAJA/util/PluginStrategy.hpp"
  #include <iostream>

  class CounterPlugin : public RAJA::util::PluginStrategy
  {
    public:
    void preLaunch(RAJA::util::PluginContext& p) {
      if (p.platform == RAJA::Platform::host)
      {
        std::cout << "Launching host kernel for the " << ++host_counter << " time!" << std::endl;
      }
      else
      {
        std::cout << "Launching device kernel for the " << ++device_counter << " time!" << std::endl;
      }    
    }
  
    void postLaunch(RAJA::util::PluginContext& RAJA_UNUSED_ARG(p)) {
    }
    
    private:
    int host_counter;
    int device_counter;
  };

  // Statically loading plugin.
  static RAJA::util::PluginRegistry::add<CounterPlugin> P("Counter", "Counts number of kernel launches.");
  
  // Dynamically loading plugin.
  extern "C" RAJA::util::PluginStrategy *getPlugin ()
  {
    return new CounterPlugin;
  }
