#pragma once
namespace trove {

template<int u, int v>
struct static_gcd;

namespace detail {

template<bool u_odd, bool v_odd, int u, int v>
struct static_gcd_helper {
    static const int value = static_gcd<u>>1, v>>1>::value << 1;
};

template<int u, int v>
struct static_gcd_helper<false, true, u, v> {
    static const int value = static_gcd<u>>1, v>::value;
};

template<int u, int v>
struct static_gcd_helper<true, false, u, v> {
    static const int value = static_gcd<u, v>>1>::value;
};

template<int u, int v>
struct static_gcd_helper<true, true, u, v> {
    static const int reduced_u = (u > v) ? (u - v) >> 1 : (v - u) >> 1;
    static const int reduced_v = (u > v) ? v : u;
    static const int value = static_gcd<reduced_u, reduced_v>::value;
};
}

template<int u, int v>
struct static_gcd {
    static const bool u_odd = (u & 0x1) == 1;
    static const bool v_odd = (v & 0x1) == 1;
    static const bool equal = u == v;
    static const int value = equal ? u : detail::static_gcd_helper<u_odd, v_odd, u, v>::value;
};

template<int v>
struct static_gcd<0, v> {
    static const bool value = v;
};

template<int u>
struct static_gcd<u, 0> {
    static const bool value = u;
};

}
