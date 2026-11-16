#include "strata_heterogeneous/vanilla_serve.hpp"
#include <cassert>
using namespace strata::heterogeneous::vanilla;
int main() {
    auto r = parse_gen("GEN 3 ckpt=0 top_k=1 12,13,14", 100, 100, 4);
    assert(r.max_new==3 && r.ids==std::vector<int32_t>({12,13,14}));
    auto greedy = parse_gen("GEN 3 temperature=0 top_p=0.95 min_p=0.05 top_k=20 seed=18446744073709551615 12,13,14", 100, 100, 4);
    assert(greedy.ids==r.ids);
    for (const char* s : {"GEN 3 temperature=0.8 1", "GEN 3 penalty_repeat=1.2 1", "GEN 3 top_p=2 1", "GEN 3 seed=18446744073709551616 1", "GEN 3 1,", "GEN 3 1,,2", "GEN 5 1", "GEN 3 100", "GEN 3 1 2", "GEN 3 ckpt=0 ckpt=0 1"}) {
        bool rejected=false; try { (void)parse_gen(s,100,100,4); } catch (const std::invalid_argument&) { rejected=true; } assert(rejected);
    }
    int f[2]; assert(pipe(f)==0); Input in(128,f[0]);
    assert(in.checkpoint()==Input::Control::proceed);
    assert(write(f[1],"GEN 3 1,2\nST",12)==12);
    assert(in.line(true)=="GEN 3 1,2"); assert(in.checkpoint()==Input::Control::proceed);
    assert(write(f[1],"OP\nQUIT\n",8)==8);
    assert(in.checkpoint()==Input::Control::stop); assert(in.checkpoint()==Input::Control::quit);
    close(f[1]); assert(in.checkpoint()==Input::Control::eof); close(f[0]);
    auto out=std::tmpfile(); assert(out); print_done(out,{3,0,3,2,12,17,0,0,0,false},"length");
    strata::core::print_serve_ready(out, 4096);
    strata::core::print_serve_reused(out, 0);
    strata::core::print_serve_token(out, 42);
    print_progress(out, 2, 3, 20);
    strata::core::print_serve_progress(out, 5, 8, 3, 20);
    std::rewind(out); char line[128]; assert(std::fgets(line,sizeof line,out)); assert(std::string(line)=="DONE 2 3 12.0 17.0 length 0 0 0\n");
    for (const char* expected : {"READY 4096 stop\n", "REUSED 0\n", "T 42\n", "PP 2 3 20 100.0\n", "PP 5 8 20 100.0\n"}) {
        assert(std::fgets(line,sizeof line,out)); assert(std::string(line)==expected);
    }
    std::fclose(out);
}
