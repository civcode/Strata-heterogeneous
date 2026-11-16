#include "strata/core/serve_request_summary.hpp"
#include <string>
#include <stdexcept>

static std::string render(const strata::core::ServeRequestSummary& summary) {
    std::FILE* out = std::tmpfile();
    if (!out) throw std::runtime_error("tmpfile");
    strata::core::print_serve_request_summary(out, summary);
    std::rewind(out);
    char text[512]{};
    const auto n = std::fread(text, 1, sizeof(text), out);
    std::fclose(out);
    return std::string(text, n);
}

int main() {
    if (render({10, 2, 8, 2, 20, 50, 1, 3, 4, false}) !=
        "strata serve: prompt 10 tokens = 2 reused + 8 read in 20 ms (400.0 tok/s), "
        "2 generated in 50 ms (40.0 tok/s), drafts accepted 1 of 3, 4 checkpoints\n") return 1;
    if (render({10, 2, 3, 0, 10, 0, 0, 0, 0, true}) !=
        "strata serve: prompt 10 tokens = 2 reused + 3 of 8 read in 10 ms (300.0 tok/s), "
        "0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, 0 checkpoints (cancelled)\n") return 2;
    return 0;
}
