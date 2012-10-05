#pragma once

namespace trove {
namespace detail {

template<typename T>
struct size_in_ints {
    static const int value = (sizeof(T) - 1)/sizeof(int) + 1;
};

template<typename T, int s>
union dismember {
    T d;
    int i[s];
};

}
}
