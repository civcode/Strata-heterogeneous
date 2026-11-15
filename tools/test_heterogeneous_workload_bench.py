import unittest
import io
from unittest.mock import Mock, patch
from heterogeneous_workload_bench import endpoint, score, run_case, aggregate, validate_manifest, request, SCHEMA

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
    def test_stream_completion_and_truncation(self):
        data=b'data: {"choices":[{"delta":{"content":"42"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\n'
        for suffix, expected in [(b'data: [DONE]\n\n',"COMPLETE"), (b'',"ERROR")]:
            r=io.BytesIO(data+suffix); r.status=200
            conn=Mock(); conn.getresponse.return_value=r
            with patch("heterogeneous_workload_bench.http.client.HTTPConnection",return_value=conn):
                result=request("http://127.0.0.1:8080",{"stream":True})
            self.assertEqual(result["status"],expected)
            self.assertTrue(r.closed)
            if expected=="COMPLETE":
                self.assertEqual(result["message"]["content"],"42")
                self.assertEqual(result["usage"]["completion_tokens"],1)
                self.assertIsNotNone(result["ttft_s"])

    def test_client_cancel_closes_response(self):
        r=io.BytesIO(b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'); r.status=200
        conn=Mock();conn.getresponse.return_value=r
        with patch("heterogeneous_workload_bench.http.client.HTTPConnection",return_value=conn):
            result=request("http://127.0.0.1:8080",{"stream":True},cancel_after_first=True)
        self.assertEqual(result["status"],"CLIENT_CANCELLED")
        self.assertTrue(r.closed)
        conn.close.assert_called_once()

    def test_failures_kept(self):
        rows=[{"id":"a","quality":"FAIL","stages":[{"status":"ERROR"}]},
              {"id":"a","quality":"PASS","stages":[{"status":"COMPLETE","elapsed_s":2,
                       "usage":{"completion_tokens":4}}]}]
        r=aggregate(rows)["a"]
        self.assertEqual((r["attempts"],r["quality_fail"],r["median_end_to_end_completion_tokens_s"]),(2,1,2))
if __name__=="__main__": unittest.main()
