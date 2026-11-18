// Drives a front stage worker without a host engine (bring-up step 1 of handoff 0066; .ai/SEAM.md).
//
//   mixed_split_front_client <front strata> --pack DIR --native SHARD [--native SHARD ..] [--layers 15] [--context N]
//                       [--chunk 2048] [--prompt 1000] [--windows 8] -- <the front engine's arguments>
//
// Starts the engine with `--front-stage K` appended, sends the hello with a fresh shared region, then: a new
// sequence, a few one-token windows with their commits, a dropped window, a new sequence, one prompt chunk.  After
// each step it checks what the front stage wrote into the shared hand-off (the ready word, finite rows that are not
// all zero).  Host-only: it touches no GPU.  The worker is ended by closing the link, never by a signal.
#include "strata/hetero/front_link.hpp"
#include "strata/hetero/worker_process.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::hetero;

namespace {
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// finite, and not all zero
bool rows_ok(const float* p, size_t n, double& sum_abs) {
    sum_abs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) return false;
        sum_abs += std::fabs((double) p[i]);
    }
    return sum_abs > 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: see the head of hetero/tools/front_client.cpp\n"); return 2; }
    const std::string exe = argv[1];
    std::string pack;
    std::vector<std::string> shards, wargs;
    int layers = 15, n_layers = 48, hc = 4, n_embd = 2560, windows = 8;
    long long context = 196608, chunk = 2048, prompt = 1000;
    std::string tokens_csv, dump_prefix;   // --tokens a,b,c --dump PREFIX: the rows of these tokens, read as a prompt and as windows
    int i = 2;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--") { ++i; break; }
        else if (a == "--pack") pack = val();
        else if (a == "--native") shards.push_back(val());
        else if (a == "--layers") layers = std::atoi(val());
        else if (a == "--context") context = std::atoll(val());
        else if (a == "--chunk") chunk = std::atoll(val());
        else if (a == "--prompt") prompt = std::atoll(val());
        else if (a == "--windows") windows = std::atoi(val());
        else if (a == "--tokens") tokens_csv = val();
        else if (a == "--dump") dump_prefix = val();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    for (; i < argc; ++i) wargs.push_back(argv[i]);
    wargs.push_back("--front-stage");
    wargs.push_back(std::to_string(layers));
    if (prompt > chunk) prompt = chunk;

    LinkIdentity id;
    id.model_fingerprint = model_fingerprint(pack, shards);
    id.n_layers = n_layers;
    id.front_end = layers;
    id.hc = hc;
    id.n_embd = n_embd;
    id.max_rows = 8;
    id.max_context = context;
    const size_t HF = (size_t) hc * n_embd + n_embd + hc, D = (size_t) hc * n_embd;
    std::string err;
    SharedRegion region;
    if (!SharedRegion::create({(size_t) id.max_rows * HF * 4, 64, (size_t) chunk * D * 4}, region, err)) {
        std::fprintf(stderr, "client: %s\n", err.c_str());
        return 1;
    }
    WorkerProcess worker;
    if (!WorkerProcess::launch(exe, wargs, 60000, worker, err)) { std::fprintf(stderr, "client: %s\n", err.c_str()); return 1; }
    int rc = 1;
    {
        RemoteFrontStage front(worker.socket(), 110000);
        const float* hand = static_cast<const float*>(region.slot(0));
        const volatile uint32_t* ready = static_cast<const volatile uint32_t*>(region.slot(1));
        const float* rows = static_cast<const float*>(region.slot(2));
        double t0 = now_ms(), s = 0;
        bool ok = front.hello(id, region, err);
        std::printf("client: hello %s after %.1f s%s%s\n", ok ? "ok" : "FAILED", (now_ms() - t0) / 1000.0, ok ? "" : ": ", ok ? "" : err.c_str());
        t0 = now_ms();
        ok = ok && front.session(SessionOp::Reset, 0, 0, err);
        std::printf("client: serving after another %.1f s\n", (now_ms() - t0) / 1000.0);
        double win_ms = 0;
        for (int w = 0; ok && w < windows; ++w) {
            const int32_t tok = 1000 + 37 * w;
            const double a = now_ms();
            ok = front.run(1, &tok, w, err);
            win_ms += now_ms() - a;
            if (ok && *ready < 1) { err = "the ready word is not up after the window"; ok = false; }
            if (ok && !rows_ok(hand, HF, s)) { err = "the hand-off row is not finite or all zero"; ok = false; }
            if (ok && w == 0) std::printf("client: window 0: ready word %u, sum|row| %.3f\n", (unsigned) *ready, s);
            ok = ok && front.commit(1, err);
        }
        if (ok) std::printf("client: %d one-token windows with commit ok, %.2f ms per window call\n", windows, win_ms / windows);
        if (ok) {   // a window of three, dropped; then the same window again, kept in part
            const int32_t t3[3] = {2000, 2001, 2002};
            ok = front.run(3, t3, windows, err) && front.session(SessionOp::Abandon, 0, 0, err) &&
                 front.run(3, t3, windows, err) && rows_ok(hand, 3 * HF, s) && front.commit(2, err);
            std::printf("client: three-token window, dropped, run again, 2 kept: %s\n", ok ? "ok" : "FAILED");
        }
        if (ok && prompt > 0) {
            std::vector<int32_t> ptok((size_t) prompt);
            for (long long k = 0; k < prompt; ++k) ptok[(size_t) k] = (int32_t) (1000 + (k * 7919) % 50000);
            ok = front.session(SessionOp::Reset, 0, 0, err);
            const double a = now_ms();
            ok = ok && front.prefill(ptok.data(), prompt, 0, true, err);
            const double ms = now_ms() - a;
            if (ok && !rows_ok(rows, (size_t) prompt * D, s)) {
                long long bad = 0, zero_rows = 0, first_bad = -1;
                for (long long r = 0; r < prompt; ++r) {
                    bool all0 = true;
                    for (size_t k = 0; k < D; ++k) {
                        const float v = rows[(size_t) r * D + k];
                        if (!std::isfinite(v)) { ++bad; if (first_bad < 0) first_bad = r; }
                        if (v != 0.0f) all0 = false;
                    }
                    zero_rows += all0;
                }
                err = "the prompt rows are not finite or all zero (" + std::to_string(bad) + " non-finite values, first in row " +
                      std::to_string(first_bad) + "; " + std::to_string(zero_rows) + " all-zero rows; first words";
                for (int k = 0; k < 4; ++k) {
                    uint32_t u = 0;
                    std::memcpy(&u, rows + k, 4);
                    char b[16];
                    std::snprintf(b, sizeof b, " %08x", u);
                    err += b;
                }
                err += ")";
                ok = false;
            }
            if (ok) std::printf("client: prompt chunk of %lld tokens ok in %.0f ms (%.0f tok/s on the front layers), sum|rows| %.1f\n",
                                prompt, ms, 1000.0 * (double) prompt / ms, s);
            const int32_t nt = 3000;   // a window after the prompt
            ok = ok && front.run(1, &nt, prompt, err) && front.commit(1, err);
        }
        if (ok && !tokens_csv.empty() && !dump_prefix.empty()) {
            // the same tokens as one prompt run (prompt rows: R after the front's last layer) and as one-token windows
            // (each window's hand-off row: R, bo, inject), written raw for an offline comparison
            std::vector<int32_t> tk;
            for (size_t p0 = 0; p0 <= tokens_csv.size();) {
                size_t q = tokens_csv.find(',', p0);
                if (q == std::string::npos) q = tokens_csv.size();
                if (q > p0) tk.push_back((int32_t) std::atol(tokens_csv.substr(p0, q - p0).c_str()));
                p0 = q + 1;
            }
            const int64_t n = (int64_t) tk.size();
            ok = front.session(SessionOp::Reset, 0, 0, err) && front.prefill(tk.data(), n, 0, true, err);
            if (ok) {
                std::FILE* f = std::fopen((dump_prefix + ".prompt.f32").c_str(), "wb");
                std::fwrite(rows, sizeof(float), (size_t) n * D, f);
                std::fclose(f);
            }
            ok = ok && front.session(SessionOp::Reset, 0, 0, err);
            std::vector<float> wrows;
            for (int64_t t = 0; ok && t < n; ++t) {
                ok = front.run(1, &tk[(size_t) t], t, err);
                if (ok) wrows.insert(wrows.end(), hand, hand + HF);
                ok = ok && front.commit(1, err);
            }
            if (ok) {
                std::FILE* f = std::fopen((dump_prefix + ".windows.f32").c_str(), "wb");
                std::fwrite(wrows.data(), sizeof(float), wrows.size(), f);
                std::fclose(f);
                std::printf("client: dumped %lld tokens' rows to %s.{prompt,windows}.f32\n", (long long) n, dump_prefix.c_str());
            }
        }
        if (!ok) std::fprintf(stderr, "client: FAILED: %s\n", err.c_str());
        std::string e2;
        if (!front.broken()) (void) front.shutdown(e2);
        rc = ok ? 0 : 1;
    }
    if (!worker.finish(err)) { std::fprintf(stderr, "client: %s\n", err.c_str()); rc = 1; }
    else std::printf("client: the front stage ended cleanly\n");
    return rc;
}
