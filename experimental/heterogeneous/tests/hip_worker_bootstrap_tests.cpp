#include "strata_heterogeneous/hip_worker_bootstrap.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace strata::heterogeneous;
namespace {
int checks = 0;
void need(bool ok, const char* reason) {
    ++checks;
    if (!ok) throw std::runtime_error(reason);
}
}
int main() {
    try {
        const char* argv[] = {
            "worker", "--worker", "cuda:0:0-3,hip:0:3-7,hip:1:7-10",
            "10", "2", "6", "999", "1", "--config", "/tmp/hip-stage1.cfg"
        };
        HipWorkerInvocation parsed;
        std::string error;
        need(parse_hip_worker_invocation(10, argv, parsed, error), "valid HIP invocation");
        need(parsed.stage_index == 1 && parsed.config_path == "/tmp/hip-stage1.cfg", "config / index");
        need(parsed.plan.stages.size() == 3 && parsed.plan.stages[1].backend == Backend::hip,
             "plan and backend");
        argv[1] = "--worker-shared";
        argv[7] = "2";
        argv[9] = "/tmp/hip-stage2.cfg";
        need(parse_hip_worker_invocation(10, argv, parsed, error), "second HIP stage");
        need(parsed.shared_memory && parsed.stage_index == 2, "shared mode");
        const auto expect_fail = [&](const char* what) {
            need(!parse_hip_worker_invocation(10, argv, parsed, error), what);
            need(!error.empty(), "failure diagnostic");
            need(parsed.stage_index == 2 && parsed.config_path == "/tmp/hip-stage2.cfg",
                 "failure is non-mutating");
        };
        argv[7] = "0"; expect_fail("CUDA stage rejected");
        argv[7] = "3"; expect_fail("out of range");
        argv[7] = "-1"; expect_fail("negative index");
        argv[7] = "4294967296"; expect_fail("overflow index");
        argv[7] = "2x"; expect_fail("numeric suffix");
        argv[7] = "2";
        argv[1] = "--worker-else"; expect_fail("unknown mode");
        argv[1] = "--worker-shared";
        argv[8] = "--not-config"; expect_fail("unknown switch");
        argv[8] = "--config";
        argv[9] = "relative.cfg"; expect_fail("relative path");
        argv[9] = "/tmp/../hip.cfg"; expect_fail("traversal");
        argv[9] = "/tmp/./hip.cfg"; expect_fail("dot component");
        argv[9] = "/tmp/hip\n.cfg"; expect_fail("newline");
        argv[9] = "/tmp/hip\r.cfg"; expect_fail("CR");
        argv[9] = "/tmp/hip-stage2.cfg";
        argv[6] = "0"; expect_fail("zero fingerprint");
        argv[6] = "999";
        argv[2] = "cuda:0:0-3,hip:0:3-9"; expect_fail("missing plan layers");
        argv[2] = "cuda:0:0-3,hip:0:3-7,hip:1:7-10";
        need(!parse_hip_worker_invocation(9, argv, parsed, error), "missing config argument");
        need(!parse_hip_worker_invocation(10, nullptr, parsed, error), "missing argv array");
        need(parsed.stage_index == 2, "failed argc leaves result unchanged");
        const char* group[] = {"worker","--worker-hip-domain-shared","cuda:0:0-3,hip:0:3-7,hip:1:7-10",
            "10","2","6","999","2","--config","/tmp/hip-stage1.cfg","--config","/tmp/hip-stage2.cfg"};
        need(parse_hip_worker_invocation(12,group,parsed,error),"grouped HIP argv");
        need(parsed.stage_index==2 && parsed.shared_memory && parsed.domain_configs.size()==2 &&
            parsed.domain_configs[0]=="/tmp/hip-stage1.cfg" && parsed.domain_configs[1]=="/tmp/hip-stage2.cfg",
            "grouped ordered physical config routing");
        group[7]="1"; need(!parse_hip_worker_invocation(12,group,parsed,error),"group requires terminal endpoint");
        group[7]="2"; group[11]=group[9];
        need(!parse_hip_worker_invocation(12,group,parsed,error),"duplicate config refused");
        group[11]="relative.cfg";
        need(!parse_hip_worker_invocation(12,group,parsed,error),"unsafe second config refused");
        need(parsed.domain_configs[1]=="/tmp/hip-stage2.cfg","group failure leaves output unchanged");
        std::cout << "HIP bootstrap: PASS (" << checks << " assertions, no HIP runtime)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "HIP bootstrap: FAIL after " << checks << ": " << ex.what() << '\n';
        return 1;
    }
}
