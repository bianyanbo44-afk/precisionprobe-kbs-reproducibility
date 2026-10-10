"""Small trusted launcher, executed only inside the namespace sandbox."""
import contextlib
import io
import json
import resource
import sys

PRELUDE = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\n"
    "import string,re,datetime,collections,heapq,bisect,copy,math,random,statistics\n"
    "import itertools,functools,operator,io,sys,json\nsys.setrecursionlimit(50000)\n"
)


def main():
    resource.setrlimit(resource.RLIMIT_NPROC,(32,32))
    source=open("/candidate.py",encoding="utf-8").read()
    if sys.argv[1]=="stdin":
        namespace={"__name__":"__main__","__file__":"/candidate.py"}
        exec(PRELUDE,namespace)
        exec(compile(source,"/candidate.py","exec"),namespace)
        return
    request=json.loads(sys.stdin.read())
    output=sys.stdout
    try:
        namespace={"__name__":"candidate","__file__":"/candidate.py"}
        with open("/dev/null","w") as sink,contextlib.redirect_stdout(sink):
            exec(PRELUDE,namespace)
            exec(compile(source,"/candidate.py","exec"),namespace)
            args=[json.loads(line) for line in request["input"].split("\n") if line.strip()]
            if "Solution" in namespace:
                method=getattr(namespace["Solution"](),request["func_name"])
            else:
                method=namespace[request["func_name"]]
            value=method(*args)
        if isinstance(value,tuple):value=list(value)
        literal=repr(value)
        try:
            json.dumps(value,allow_nan=False)
            normalized=value
        except (ValueError,TypeError):
            normalized=literal
        result={"status":"ok","value":normalized,"literal":literal}
    except BaseException as exc:
        result={"status":"exception","exception":type(exc).__name__}
    output.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")


if __name__=="__main__":main()

