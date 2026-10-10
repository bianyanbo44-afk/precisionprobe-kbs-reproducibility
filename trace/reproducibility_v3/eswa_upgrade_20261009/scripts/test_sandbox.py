import json
import tempfile
import unittest
from pathlib import Path

from evaluate_validation import matches,run_case


class SandboxChecks(unittest.TestCase):
    def run_source(self,source,input_text="",testtype="stdin",func_name=None,timeout=4):
        with tempfile.TemporaryDirectory(prefix="precisionprobe-smoke-") as directory:
            path=Path(directory)/"candidate.py"
            path.write_text(source,encoding="utf-8")
            return run_case(path,input_text,testtype,func_name,timeout)[0]

    def test_stdio_input_and_decimal_comparison(self):
        result=self.run_source("print(sum(map(int,input().split())))\n","4 7\n")
        self.assertTrue(matches(result,"11.0","stdin"))
        self.assertFalse(matches(result,"12","stdin"))

    def test_function_uses_benchmark_import_prelude(self):
        result=self.run_source("class Solution:\n def total(self, values: List[int]) -> int:\n  return sum(values)\n","[2,3]","functional","total")
        self.assertTrue(matches(result,"5","functional"))

    def test_nested_tuple_not_silently_equal_to_nested_list(self):
        result=self.run_source("class Solution:\n def f(self,x):\n  return [(x,)]\n","1","functional","f")
        self.assertFalse(matches(result,"[[1]]","functional"))

    def test_host_files_are_not_mounted(self):
        result=self.run_source("import os\nprint(os.path.exists('/mnt/f/CCFA'))\n")
        self.assertTrue(matches(result,"False","stdin"))

    def test_syntax_failure_is_recorded(self):
        result=self.run_source("def bad(:\n")
        self.assertEqual(result["status"],"exception")
        self.assertEqual(result["exception"],"SyntaxError")

    def test_timeout_is_recorded(self):
        result=self.run_source("while True: pass\n",timeout=.5)
        self.assertEqual(result["status"],"timeout")


if __name__=="__main__":unittest.main()
