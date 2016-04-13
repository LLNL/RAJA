/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */
  
#ifndef RAJA_VIEW_HXX__
#define RAJA_VIEW_HXX__

#include <RAJA/Layout.hxx>

namespace RAJA {


template<typename DataType, typename LayoutT>
struct View {
  LayoutT const layout;
  DataType *data;

  template<typename ...Args>
  constexpr inline View(DataType *data_ptr, Args... dim_sizes )
          : layout(dim_sizes...), data(data_ptr)
          { }

  // making this specifically typed would require unpacking the layout,
  // this is easier to maintain
  template<typename ...Args>
  inline DataType &operator()(Args... args) const {
    return data[convertIndex<Index_type>(layout(args...))];
  }
};



} // namespace RAJA

#endif

