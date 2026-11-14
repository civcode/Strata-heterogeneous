#include "strata/core/pipeline_window.hpp"
#include <iostream>
#include <stdexcept>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
// Historical rule from generate.cpp: compare the extracted function to it.
int old_size(const float* pr, int from, int avail, int spec, double min_p) {
    int T = spec;
    if (min_p > 0.0) {
        T = 1;
        while (T < spec && from + T - 1 < avail && pr[from + T - 1] >= (float) min_p) ++T;
    }
    return std::max(1, std::min(T, 1 + avail - from));
}
}
int main() {
    try {
        const int32_t oc[8] = {10,11,12,13,14,15,16,17};
        const float op[8] = {.99f,.85f,.4f,.85f,.2f,.95f,.8f,.9f};
        for (int spec=1; spec<=8; ++spec)
            for (int avail=0; avail<=8; ++avail)
                for (int from=0; from<=avail; ++from)
                    for (double min: {0., .2, .5, .85, 1.}) {
                        int expected = old_size(op, from, avail, spec, min);
                        check(pipeline_window_size(op, from, avail, spec, min)==expected,"window size parity");
                    }
        PipelineWindow A;
        A.T=4; A.seq=7; A.p=100;
        A.tok[0]=1; A.tok[1]=2; A.tok[2]=3; A.tok[3]=4;
        const auto B=pipeline_make_lookahead(A, oc, op, 1, 8, .5f, 4, 0.);
        check(B.ready && B.made && !B.launched && B.seq==8 && B.p==104 && B.T==4,"next window identity");
        check(B.tok[0]==11 && B.tok[1]==12 && B.tok[3]==14 && B.prob[0]==op[2] && B.p_on==.5f*op[1],"chain content");
        check(!pipeline_make_lookahead(A, oc, op, 8, 8, .5f, 4, 0.).made,"no bonus available");
        check(pipeline_make_lookahead(A, oc, op, 7, 8, .5f, 8, 0.).T==1,"bonus only");
        int32_t target[4]={2,3,4,11};
        check(pipeline_accepted_drafts(A,target)==3,"full acceptance");
        check(pipeline_matches_lookahead(A,B,3,target),"full acceptance plus matching bonus");
        target[3]=99;
        check(!pipeline_matches_lookahead(A,B,3,target),"bonus miss");
        target[1]=99;
        check(pipeline_accepted_drafts(A,target)==1,"partial acceptance");
        check(!pipeline_matches_lookahead(A,B,1,target),"partial cannot continue lookahead");
        target[0]=99;
        check(pipeline_accepted_drafts(A,target)==0,"no draft accepted");
        A.T=1; target[0]=11;
        check(pipeline_accepted_drafts(A,target)==0 && pipeline_matches_lookahead(A,B,0,target),"single-row verdict");
        std::cout << "pipeline window policy: PASS (" << checks << " assertions)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
