//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//                         
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC                                     
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.                              
//                                                                                                      
// SPDX-License-Identifier: (BSD-3-Clause)                                                              
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//                         
                                                                                                        
// _plugin_example_start                                                                                
#include "RAJA/util/PluginStrategy.hpp"                                                                 
#include "nvToolsExt.h"                                                                                 
                                                                                                        
#include <iostream>                                                                                     
                                                                                                        
class NamePlugin :                                                                                      
  public RAJA::util::PluginStrategy                                                                     
{                                                                                                       
public:                                                                                               
  void preLaunch(const RAJA::util::PluginContext& p) override {                                         
    std::cout << " [Pre launch NamePlugin]: "<< p.name << std::endl;                                    
    nvtxRangePushA(p.name);                                                                             
  }                                                                                                     
                                                                                                        
  void postLaunch(const RAJA::util::PluginContext& p) override {                                        
    nvtxRangePop();                                                                                     
    std::cout << " [Post launch NamePlugin]: "<< p.name << std::endl;                                   
  }                                                                                                     
                                                                                                        
};                                                                                                      
                                                                                                        
// Statically loading plugin.                                                                           
static RAJA::util::PluginRegistry::add<NamePlugin> P("Name", "Names a kernel.");                        
                                                                                                        
// Dynamically loading plugin.                                                                          
extern "C" RAJA::util::PluginStrategy *getPlugin ()                                                     
{                                                                                                       
  return new NamePlugin;                                                                                
}                                                                                                       
// _plugin_example_end
