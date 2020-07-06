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

RAJA supports user-made plugins that may be loaded either at the time of compilation or during runtime. These two methods are not mutually exclusive, as plugins loaded dynamically can be run alongside plugins that are loaded statically.

------------
Plugin API
------------


^^^^^^^^^^^
Required Functions
^^^^^^^^^^^
The preLaunch and postLaunch functions are automatically called by RAJA before and after loop execution. This applies to RAJA's kernel and forall implementations.

* ``void preLaunch(PluginContext& p) {}`` - Will occur before kernel/forall execution.

* ``void postLaunch(PluginContext& p) {}`` - Will occur after kernel/forall execution.

^^^^^^^^^^^
Unrequired Functions
^^^^^^^^^^^
The init and finalize functions have standard implementations and thus are not required to be included in a user-made plugin. Init and finalize are never run by RAJA by default and are only run when the user makes a call to RAJA::util::init_plugin() or RAJA::util::finalize_plugin() respectively.

* ``void init(PluginOptions p) {}`` - Called by the user

* ``void finalize() {}``


-----------------
Static Loading
-----------------

::

  static RAJA::util::PluginRegistry::add<PluginName> P("Name", "Description");


-----------------
Dynamic Loading
-----------------
::

  extern "C" RAJA::util::PluginStrategy *getPlugin ()
  {
    return new PluginName;
  }
  

-----------------
Example Implementation
-----------------

::

  #include "RAJA/util/PluginStrategy.hpp"
  #include <iostream>

  class CounterPlugin : public RAJA::util::PluginStrategy
  {
    public:
    void preLaunch(RAJA::util::PluginContext& p) {
      if (p.platform == RAJA::Platform::host)
       std::cout << "Launching host kernel for the " << ++host_counter << " time!" << std::endl;
      else
       std::cout << "Launching device kernel for the " << ++device_counter << " time!" << std::endl;
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
  
  
