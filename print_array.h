#pragma once
#include <iostream>

namespace trove {

template<typename T>
void print_array(const array<T, 0>&) {
    std::cout << std::endl;
}

template<typename T, int s>
void print_array(const trove::array<T, s>& ary) {
    std::cout << ary.head << " ";
    print_array(ary.tail);
}

} //ends namespace trove
