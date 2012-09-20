#pragma once
#include <thrust/tuple.h>

#ifdef _MSC_VER
//In MS-land, long is only 32 bits.
#define sixty_four long long;
#else
#define sixty_four long
#endif

namespace trove {
namespace detail {

struct two_int {
    int a;
    int b;
};


union dismember {
    unsigned sixty_four m_ulong;
    sixty_four m_long;
    double m_double;
    two_int m_ints;
};

}
}

//Overloads for the built-in __shfl intrinsic

__device__
double __shfl(double in, int idx) {
    union trove::detail::dismember x;
    x.m_double = in;
    x.m_ints.a = __shfl(x.m_ints.a, idx);
    x.m_ints.b = __shfl(x.m_ints.b, idx);
    return x.m_double;
}

__device__
sixty_four __shfl(sixty_four in, int idx) {
    union trove::detail::dismember x;
    x.m_long = in;
    x.m_ints.a = __shfl(x.m_ints.a, idx);
    x.m_ints.b = __shfl(x.m_ints.b, idx);
    return x.m_long;
}

__device__
unsigned sixty_four __shfl(unsigned sixty_four in, int idx) {
    union trove::detail::dismember x;
    x.m_ulong = in;
    x.m_ints.a = __shfl(x.m_ints.a, idx);
    x.m_ints.b = __shfl(x.m_ints.b, idx);
    return x.m_ulong;
}
