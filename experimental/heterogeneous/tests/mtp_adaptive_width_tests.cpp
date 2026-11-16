#include "strata_heterogeneous/mtp_adaptive_width.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using strata::heterogeneous::MtpAdaptiveWidth;

int main() {
    unsigned checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) throw std::runtime_error("adaptive width check failed");
    };
    MtpAdaptiveWidth w;
    check(w.width() == 4 && w.limit(-1) == 0 && w.limit(0) == 0 &&
          w.limit(2) == 2 && w.limit(100) == 4);
    check(!w.observe(5, 0, 0) && !w.observe(4, 5, 0) &&
          !w.observe(4, 2, 3) && w.width() == 4);
    for (unsigned cap = 4; cap; --cap) {
        check(w.width() == cap);
        check(w.observe(cap, cap, 0) && w.width() == cap);
        check(w.observe(cap, cap, 0) && w.width() == cap - 1);
    }
    check(w.limit(100) == 0);
    check(w.observe(0, 0, 0) && w.width() == 0);
    check(w.observe(0, 0, 0) && w.width() == 1);
    for (unsigned cap = 1; cap < 4; ++cap) {
        check(w.observe(cap, cap, cap) && w.width() == cap);
        check(w.observe(cap, cap, cap) && w.width() == cap + 1);
    }
    check(w.observe(4, 4, 0));
    check(w.observe(4, 0, 0)); // no submitted candidate, no learning
    check(w.observe(4, 4, 0) && w.width() == 4);
    check(w.observe(2, 2, 0) && w.width() == 4); // budget clipped
    check(w.observe(4, 4, 0) && w.width() == 3);
    check(w.observe(3, 2, 2) && w.width() == 3);
    check(w.observe(3, 2, 2) && w.width() == 3);
    w = {};
    check(w.width() == 4);
    // Exhaust the legal observation tuples at each reachable width.
    for (unsigned target = 0; target <= 4; ++target)
        for (unsigned budget = 0; budget <= 5; ++budget)
            for (unsigned offered = 0; offered <= 5; ++offered)
                for (unsigned accepted = 0; accepted <= 5; ++accepted) {
                    MtpAdaptiveWidth a;
                    while (a.width() > target) {
                        const auto k = a.width();
                        check(a.observe(k,k,0) && a.observe(k,k,0));
                    }
                    const bool valid = budget <= target && offered <= budget && accepted <= offered;
                    check(a.observe(budget,offered,accepted) == valid);
                    check(a.width() <= 4);
                    if (!valid || budget < target) check(a.width() == target);
                }
    std::cout << "MTP adaptive width: PASS " << checks << " assertions\n";
}
