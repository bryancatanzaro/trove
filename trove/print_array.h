#pragma once
#include <iostream>
#include <trove/array.h>

namespace trove {

template<typename T>
std::ostream& operator<<(std::ostream& strm, const array<T, 1>& ary) { 
    strm << ary.head;
    return strm;
}

template<typename T, int s>
std::ostream& operator<<(std::ostream& strm, const array<T, s>& ary) { 
    strm << ary.head << " ";
    strm << ary.tail;
    return strm;
}


} //ends namespace trove
