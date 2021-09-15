#ifndef NEW_REDUCE_DEMANGLE
#define NEW_REDUCE_DEMANGLE
  const char* demangle(const char* name)
  {
    char* buf = new char [1024];
    size_t size = 1024;
    int status;
    char* res = abi::__cxa_demangle(name, buf, &size, &status);
    return res;
  }
#endif //  NEW_REDUCE_DEMANGLE
