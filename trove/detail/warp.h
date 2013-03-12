#pragma once

namespace trove {
namespace detail {

#define WARP_CONVERGED 0xffffffff

__device__
bool warp_converged() {
    return (__ballot(true) == WARP_CONVERGED);
}

}
}
