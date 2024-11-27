//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

#include <iostream>
#include <caliper/cali.h>

class CaliperPlugin : public RAJA::util::PluginStrategy
{
public:
  void preLaunch(const RAJA::util::PluginContext&p) override
  {
    CALI_MARK_BEGIN(p.kernel_name->c_str());
  }

  void postLaunch(const RAJA::util::PluginContext& p) override
  {
    CALI_MARK_END(p.kernel_name->c_str());
  }

private:

};

// Dynamically loading plugin.
extern "C" RAJA::util::PluginStrategy *getPlugin()
{
  return new CaliperPlugin;
}

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<CaliperPlugin> P("Caliper", "Enables Caliper Profiling");
