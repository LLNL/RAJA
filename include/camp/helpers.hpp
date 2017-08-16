#ifndef CAMP_HELPERS_HPP
#define CAMP_HELPERS_HPP

namespace camp
{

template <typename T>
T* declptr();

template <typename... Ts>
void sink(Ts...)
{
}

}

#endif /* CAMP_HELPERS_HPP */
