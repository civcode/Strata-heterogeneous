import unittest
from heterogeneous_workload_bench import endpoint, score, run_case, aggregate, validate_manifest, SCHEMA

class Tests(unittest.TestCase):
    def test_loopback_only(self):
        self.assertEqual(endpoint("http://127.0.0.1:8080"),("127.0.0.1",8080))
        for u in ("http://example.com:80","https://127.0.0.1:8080","http://127.0.0.1:8080/private",
                  "http://x@127.0.0.1:8080"):
            with self.assertRaises(ValueError): endpoint(u)
    def test_quality(self):
        r={"status":"COMPLETE","finish_reason":"stop","message":{"content":" 42\n"}}
        self.assertEqual(score(r,{"kind":"exact","value":"42"}),"PASS")
        self.assertEqual(score(r,None),"UNSCORED")
        self.assertEqual(score({**r,"finish_reason":"length"},{"kind":"exact","value":"42"}),"FAIL")
    def test_fixture_no_execution(self):
        case={"id":"tool","messages":[{"role":"user","content":"lookup"}],"tools":[],
              "max_tokens":32,"fixture_tool":{"name":"lookup","arguments":{"id":7},"result":"42"},
              "expected":{"kind":"exact","value":"42"}}
        validate_manifest({"schema":SCHEMA,"cases":[case]})
        calls=[]
        def sender(url,body,key,timeout,cancel=False):
            calls.append(body)
            if len(calls)==1:
                return {"status":"COMPLETE","finish_reason":"tool_calls","message":{"role":"assistant",
                  "tool_calls":[{"type":"function","id":"1","function":{"name":"lookup","arguments":"{\"id\":7}"}}]}}
            return {"status":"COMPLETE","finish_reason":"stop","message":{"content":"42"}}
        self.assertEqual(run_case("", "", case,"",10,sender)["quality"],"PASS")
        self.assertEqual(calls[-1]["messages"][-1]["content"],"42")
        def bad(*args):
            return {"status":"COMPLETE","finish_reason":"tool_calls","message":{"tool_calls":[]}}
        self.assertEqual(run_case("","",case,"",10,bad)["quality"],"FAIL")
    def test_failures_kept(self):
        rows=[{"id":"a","quality":"FAIL","stages":[{"status":"ERROR"}]},
              {"id":"a","quality":"PASS","stages":[{"status":"COMPLETE","elapsed_s":2,
                       "usage":{"completion_tokens":4}}]}]
        r=aggregate(rows)["a"]
        self.assertEqual((r["attempts"],r["quality_fail"],r["median_end_to_end_completion_tokens_s"]),(2,1,2))
if __name__=="__main__": unittest.main()
