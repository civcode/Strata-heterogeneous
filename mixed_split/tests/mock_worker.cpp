// A front stage worker without a GPU, for the process tests.  Modes: serve | exit3 | slow-exit

#include "stub_backend.hpp"
#include "strata/mixed_split/worker_process.hpp"

#include <cstdio>
#include <cstring>
#include <unistd.h>

int main(int argc, char** argv) {
    using namespace strata::mixed_split;
    const char* mode = argc > 1 ? argv[1] : "serve";
    if (std::strcmp(mode, "exit3") == 0) return 3;
    test::StubBackend backend;
    backend.check_identity = true;
    backend.expect = test::test_identity();
    std::string err;
    const int rc = serve_front(kWorkerSocketFd, backend, err);
    if (rc != 0) std::fprintf(stderr, "mock worker: %s\n", err.c_str());
    if (std::strcmp(mode, "slow-exit") == 0) ::usleep(400 * 1000);   // a worker that drains slowly after the socket closed
    return rc;
}
