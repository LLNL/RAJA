///
/// Source file containing clangcuda multiply defined symbol issue reproducer.
///
/// Immediate Issue: Can't link code with templated global or device functions in headers.
///
/// Larger Issue: Can't link large codes.
///
/// Seen when compiling for sm_60 on Sierra EA systems (IBM p8, Nvidia P100)
/// Seen when compiling for sm_70 on Sierra systems (IBM p9, Nvidia V100)
///
/// Notes: The symbols are not marked .weak in the PTX.
///

template < typename T >
__global__ void templated_global_func()
{
}

extern void other();
