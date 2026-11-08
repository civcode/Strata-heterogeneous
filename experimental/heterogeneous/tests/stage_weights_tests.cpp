#include "strata_heterogeneous/stage_weights.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace strata::heterogeneous;
namespace {
int asserts = 0;
void require(bool x, const std::string& why) {
    ++asserts;
    if (!x) throw std::runtime_error(why);
}
std::string row(const std::string& name, uint64_t offset, uint64_t bytes) {
    return name + " 0 0 0 " + std::to_string(bytes) + " " + std::to_string(offset) + " " +
           std::to_string(bytes) + " 16 1 0 0 0 0 0 0 0 0 0 0\n";
}
std::string index(std::vector<std::string> names, uint64_t alignment = 256) {
    std::string content = "# align " + std::to_string(alignment) + " pool " +
        std::to_string(alignment * names.size()) + " tensors " + std::to_string(names.size()) + "\n" +
        "# name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind\n";
    for (size_t i=0;i<names.size();++i) content += row(names[i], alignment * i, 10);
    return content;
}
std::vector<std::string> names() {
    return {"token_embd.weight", "output.weight", "blk.0.attn.weight", "blk.1.attn.weight",
            "blk.2.attn.weight", "blk.3.attn.weight", "blk.4.attn.weight", "blk.5.attn.weight"};
}
struct TempIndex {
    std::string path;
    explicit TempIndex(const std::string& contents) {
        char pattern[] = "/tmp/strata-index-XXXXXX";
        const int fd = ::mkstemp(pattern);
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        ::close(fd);
        path = pattern;
        write(contents);
    }
    void write(const std::string& s) {
        std::ofstream f(path, std::ios::trunc | std::ios::binary);
        if (!f || !(f << s)) throw std::runtime_error("temporary index write failed");
    }
    ~TempIndex() { if (!path.empty()) std::remove(path.c_str()); }
};
Plan plan() { return parse_plan("cuda:0:0-2,hip:0:2-4,hip:1:4-6", 6, {2,3}, 999); }
void successful() {
    auto p = plan();
    TempIndex file(index(names()));
    StageWeightPlan w;
    std::string error;
    for (size_t i=0;i<3;++i) {
        require(plan_stage_weights(p,i,file.path,w,error), "valid stage index accepted " + error);
        require(error.empty(), "success clears diagnostic");
        require(w.full_pool_bytes==2048 && w.selected_pool_bytes==1024,
                "compacted pool matches WeightTable::pool_bytes algorithm");
        require(w.total_tensors==8 && w.selected_tensors==4 && w.stage_layer_tensors==2,
                "correct row counts");
        require(w.skip.size()==4, "four tensors outside stage");
        require(w.skip.count("token_embd.weight")==0 && w.skip.count("output.weight")==0,
                "global head/embedding kept on all stages until loader verifies ownership");
        for (size_t layer=0;layer<6;++layer)
            require(w.skip.count("blk."+std::to_string(layer)+".attn.weight") ==
                    (layer < i*2 || layer >= i*2+2), "correct block layer selection");
    }
    // Ensure non-block names (including similarly-prefixed ones) are never
    // silently removed, regardless of stage.
    auto extended = names(); extended.push_back("blkmeta.2.device_tensor");
    file.write(index(extended));
    require(plan_stage_weights(p,0,file.path,w,error) && !w.skip.count("blkmeta.2.device_tensor"),
            "unknown/global tensor stays resident");
    // Verifies actual indexed tensor bytes are aligned, not assumed fixed.
    std::string special="# align 128 pool 640 tensors 4\n";
    special += row("token_embd.weight",0,100);
    special += row("output.weight",128,200);
    special += row("blk.0.weight",384,10);
    special += row("blk.1.weight",512,20);
    auto single=parse_plan("cuda:0:0-2",2,{2,3},111);
    file.write(special);
    require(plan_stage_weights(single,0,file.path,w,error), "variable-length sizes accepted");
    require(w.selected_pool_bytes==640, "variable sizes aligned as native loader");
    require(w.skip.empty(), "no omitted layers in a whole-range plan");
}
void fails(const std::string& s, const char* why, size_t stage_index=0) {
    TempIndex file(s); StageWeightPlan w; w.full_pool_bytes=123456;
    std::string error;
    require(!plan_stage_weights(plan(),stage_index,file.path,w,error),why);
    require(!error.empty(), std::string(why)+": diagnostic provided");
    require(w.full_pool_bytes==123456 && w.skip.empty(), std::string(why)+": output unchanged");
}
void failures() {
    auto n=names();
    fails("", "missing header and rows");
    fails(row("blk.0.foo",0,10)+index(n), "row before header");
    fails(index(n)+"\n", "blank row rejected, like native loader");
    fails(index(n),"out of range stage index",5);
    auto s=index(n); s.replace(s.find("tensors 8"),9,"tensors 9");
    fails(s,"tensor-count mismatch");
    s=index(n);s += "# align 256 pool 2048 tensors 8\n";
    fails(s,"duplicate header");
    n=names();n[2]=n[3];fails(index(n),"duplicate tensor name");
    n=names();n[2]="blk.5.misplaced";fails(index(n),"missing owned layer");
    n=names();n[2]="blk.6.extra";fails(index(n),"out-of-model layer");
    n=names();n[2]="blk.02.extra";fails(index(n),"noncanonical block layer");
    n=names();n[2]="blk.-1.extra";fails(index(n),"negative block layer");
    n=names();n[2]="blk.x.extra";fails(index(n),"nonnumeric block layer");
    n=names();n[2]="blk.0.";fails(index(n),"malformed empty tensor suffix");
    s=index(names());s += "unexpected\n";fails(s,"bad trailing row");
    s=index(names());s += "# only commentary is safe\n";
    TempIndex t(s); StageWeightPlan w;std::string e;
    require(plan_stage_weights(plan(),0,t.path,w,e),"well-formed comments allowed");
    s=index(names());s.replace(s.find(" 16 1 0"),5," abc 1");
    fails(s,"malformed dimension");
    s=index(names());s.replace(s.find(" 0 0 0 10 0 10"),15," 0 0 0 10 99999 10");
    fails(s,"out-of-header offset");
    s=index(names());s.replace(s.find(" 0 0 0 10 256 10"),17," 0 0 0 10 3 10");
    fails(s,"unaligned destination offset");
    s=index(names());s.replace(s.find(" 0 0 0 10 256 10"),17," 0 0 0 10 0 10");
    fails(s,"overlapping destination offsets");
    s=index(names());s.replace(s.find(" 0 0 0 10 0 10"),15,
                                  " 0 0 18446744073709551615 10 0 10");
    fails(s,"overflowed source span");
    s=index(names());s.replace(s.find("# align 256"),11,"# align 0");
    fails(s,"zero alignment");
    s=index(names());s.replace(s.find(" 0 0 0 10 512 10"),17," 0 0 0 10 512 0");
    fails(s,"selected tensor has no data");
    s=index(names());s += "# " + std::string(1100,'x') + "\n";
    fails(s,"long row exceeds loader fgets limit");
    s=index(names());s.replace(s.find(" 0 0 0 10"),9," 9 0 0 10");
    fails(s,"invalid file id");
    TempIndex x(index(names()));
    const std::string nonexistent=x.path+"not-here";
    w.full_pool_bytes=444;
    require(!plan_stage_weights(plan(),0,nonexistent,w,e),"missing pack index rejected");
    require(w.full_pool_bytes==444,"missing file leaves original result");
    auto bad=plan();bad.stages[0].first_layer=1;
    require(!plan_stage_weights(bad,0,x.path,w,e),"invalid plan rejected before file reads");
}
} // namespace
int main() {
    try { successful(); failures();
        std::cout << "Stage weight planner PASS (" << asserts << " assertions)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Stage weight planner FAIL: " << ex.what() << '\n';
        return 1;
    }
}
