#pragma once
#include <stdexcept>
namespace strata::kernels::cpu {
namespace fake_pool {inline int created=0,destroyed=0;inline bool fail_create=false;}
class ExpertPool{
public:
 ExpertPool(int n,bool,bool){if(fake_pool::fail_create)throw std::runtime_error("pool ctor failure");if(n<=0||n>28)throw std::runtime_error("bad CPU workers");++fake_pool::created;}
 ~ExpertPool(){++fake_pool::destroyed;}
};
}
