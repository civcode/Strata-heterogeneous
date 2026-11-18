// The mixed split's hook in vanilla's conversation cache (ConversationCache::on_drop, STRATA_MIXED_SPLIT): the host tells the
// front stage about every parked conversation the cache lets go, so the front stage's copy goes too.  It must fire for
// an eviction, a refused put and a clear, and must not fire for take(), which hands a conversation out to be restored
// (the front stage restores its copy right after).

#ifndef STRATA_MIXED_SPLIT
#define STRATA_MIXED_SPLIT 1
#endif
#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    return s;
}
}

int main() {
    std::vector<std::vector<int32_t>> dropped;
    auto watch = [&](ConversationCache& c) {
        c.on_drop = [&](const SavedConversation& s) { dropped.push_back(s.live.ids); };
    };
    {   // eviction by the slot count: the oldest goes, and the hook names it
        ConversationCache cache(1 << 20, 2);
        watch(cache);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        check(dropped.empty(), "nothing dropped yet");
        check(cache.put(image({5, 5, 5})), "park C");
        check(dropped.size() == 1 && dropped[0] == std::vector<int32_t>({1, 2, 3}), "A evicted and reported");
    }
    dropped.clear();
    {   // take() hands a conversation out to be restored: not a drop
        ConversationCache cache(1 << 20, 4);
        watch(cache);
        check(cache.put(image({1, 2, 3})), "park A");
        const SavedConversation out = cache.take(0);
        check(out.live.ids == std::vector<int32_t>({1, 2, 3}), "A taken");
        check(dropped.empty(), "take is not reported");
    }
    dropped.clear();
    {   // a put the budget refuses: the front stage parked its copy first, so it is reported
        ConversationCache cache(16, 4);
        watch(cache);
        check(!cache.put(image({4, 4})), "too large for the budget");
        check(dropped.size() == 1 && dropped[0] == std::vector<int32_t>({4, 4}), "the refused image is reported");
    }
    dropped.clear();
    {   // clear (#879): every parked conversation is reported
        ConversationCache cache(1 << 20, 4);
        watch(cache);
        check(cache.put(image({1})), "park A");
        check(cache.put(image({2})), "park B");
        check(cache.clear() == 2, "two cleared");
        check(dropped.size() == 2, "both reported");
    }
    dropped.clear();
    {   // no hook set (every serve that is not a mixed split): nothing changes
        ConversationCache cache(1 << 20, 1);
        check(cache.put(image({1})), "park A");
        check(cache.put(image({2})), "park B evicts A");
        check(cache.size() == 1, "one parked");
    }
    std::printf("cache_drop_test: %d checks passed\n", checks);
    return 0;
}
