#pragma once

namespace trove {
namespace detail {

template<int a, int m, int r=1>
struct static_mod_inverse;

template<bool done, int a, int m, int r>
struct static_mod_inverse_helper {
    //If you get this returned, it means the mod inverse doesn't exist.
    static const int value = -1;
};

template<int a, int m, int r>
struct static_mod_inverse_helper<false, a, m, r> {
    static const int fx = (r * a) % m;
    static const bool found = (fx == 1);
    static const int value = found ? r : static_mod_inverse<a, m, r+1>::value;
};

template<int a, int m, int r>
struct static_mod_inverse {
    static const bool done = r == m;
    static const int value = static_mod_inverse_helper<done, a, m, r>::value;
};

}
}
