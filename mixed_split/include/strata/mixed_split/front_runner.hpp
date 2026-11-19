#pragma once

// The front stage's window in flight (.ai/SEAM.md): the front stage answers a window call once the window is
// launched, not when it ends, so the host's first stage launches its own window at once and its graph waits on the
// ready word on the GPU.  Someone must still serve the window to its end (vanilla's Verifier::run serves its layers'
// CPU experts and synchronizes; a pipelined window is serviced and polled).  WindowRunner does that on a thread of
// its own, while the link thread waits for the host's next call; that call first waits for the window (settle).
//
// One job at a time.  The job owns the verifier between start() and settle(): the link thread does not touch it
// meanwhile.  No GPU, engine or transport dependency.

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace strata::mixed_split {

class WindowRunner {
public:
    /// A job runs the window.  It calls `launched()` once the window is launched (later calls do nothing) and returns
    /// whether the window ended without an error (`err` says why not).
    using Job = std::function<bool(const std::function<void()>& launched, std::string& err)>;

    WindowRunner() = default;
    WindowRunner(const WindowRunner&) = delete;
    WindowRunner& operator=(const WindowRunner&) = delete;
    ~WindowRunner() {
        std::string ignored;
        (void) settle(ignored);
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

    /// Settle any job before, then run `job` on the runner's thread and return once it has called launched() (true) or
    /// has ended first (its result; false with `err` when it failed before its launch).  A failure after the launch is
    /// kept for the next settle().
    bool start(Job job, std::string& err) {
        if (!settle(err)) return false;
        if (!th_.joinable()) th_ = std::thread([this] { loop(); });
        std::unique_lock<std::mutex> lk(mu_);
        job_ = std::move(job);
        has_job_ = true;
        busy_ = true;
        launched_ = false;
        cv_.notify_all();
        cv_.wait(lk, [this] { return launched_ || !busy_; });
        if (busy_) return true;   // launched; it goes on
        has_result_ = false;      // ended before (or at) its launch: its result is this call's
        if (!ok_) { err = err_; return false; }
        return true;
    }

    /// Wait for the job in flight (if any) to end.  False, with its error, when a job failed after start() returned;
    /// the error is reported once.
    bool settle(std::string& err) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return !busy_; });
        if (!has_result_) return true;
        has_result_ = false;
        if (ok_) return true;
        err = err_;
        return false;
    }

    /// A job is in flight (started and not ended).
    bool busy() {
        std::lock_guard<std::mutex> lk(mu_);
        return busy_;
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread th_;
    Job job_;
    bool has_job_ = false, busy_ = false, launched_ = false, quit_ = false, has_result_ = false, ok_ = true;
    std::string err_;

    void loop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return has_job_ || quit_; });
                if (quit_ && !has_job_) return;
                job = std::move(job_);
                job_ = nullptr;
                has_job_ = false;
            }
            std::string e;
            auto launched = [this] {
                std::lock_guard<std::mutex> lk(mu_);
                if (busy_ && !launched_) {
                    launched_ = true;
                    cv_.notify_all();
                }
            };
            const bool ok = job(launched, e);
            {
                std::lock_guard<std::mutex> lk(mu_);
                ok_ = ok;
                err_ = ok ? std::string() : (e.empty() ? std::string("the front stage's window failed without a message") : e);
                has_result_ = true;
                busy_ = false;
            }
            cv_.notify_all();
        }
    }
};

} // namespace strata::mixed_split
