from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parent
sys.path.insert(0,str(ROOT/"src/precisionprobe"))
from scoring import dominant_semantic_distance_entropy,exact_dominant_semantic_distance_entropy,failure_fraction


def read_jsonl(path):
    if not Path(path).exists():return []
    with Path(path).open(encoding="utf-8") as f:return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def limits():
    resource.setrlimit(resource.RLIMIT_AS,(768*1024*1024,768*1024*1024))
    resource.setrlimit(resource.RLIMIT_CPU,(4,5))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))


def normalize_stdio(value):
    lines=[]
    for line in value.strip().split("\n"):
        words=[]
        for token in line.strip().split():
            try:
                if len(token)>512:raise ValueError("long numeric output")
                number=Decimal(token)
                if not number.is_finite() or abs(number.adjusted())>308:raise ValueError("unbounded numeric output")
                words.append(int(number) if number==number.to_integral() else float(number))
            except (InvalidOperation,ValueError,OverflowError):words.append(token)
        lines.append(words)
    return lines


def run_case(source_path,input_text,testtype,func_name,timeout=4.):
    command=["bwrap","--unshare-all","--die-with-parent","--new-session",
        "--ro-bind","/usr","/usr","--symlink","usr/lib","/lib","--symlink","usr/lib64","/lib64",
        "--symlink","usr/bin","/bin","--proc","/proc","--dev","/dev","--tmpfs","/tmp",
        "--ro-bind",str(source_path),"/candidate.py","--ro-bind",str(BASE/"scripts/sandbox_function.py"),"/launcher.py",
        "--clearenv","--setenv","PATH","/usr/bin","--setenv","LANG","C.UTF-8","--chdir","/tmp",
        "/usr/bin/python3","-I","/launcher.py",testtype]
    payload=input_text if testtype=="stdin" else json.dumps({"input":input_text,"func_name":func_name})
    started=time.perf_counter()
    with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
        try:
            result=subprocess.run(command,input=payload.encode(),stdout=stdout,stderr=stderr,timeout=timeout,preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return {"status":"timeout"},time.perf_counter()-started
        elapsed=time.perf_counter()-started
        stdout.seek(0);text=stdout.read(1024*1024).decode("utf-8",errors="replace")
        stderr.seek(0);error=stderr.read(4096).decode("utf-8",errors="replace")
    if "bwrap:" in error:
        raise RuntimeError("sandbox infrastructure failure: "+error[:500])
    if result.returncode:
        match=re.findall(r"^([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception)):",error,re.M)
        exception=match[-1] if match else "ProcessExit"
        return {"status":"exception","exception":exception,"returncode":result.returncode},elapsed
    if testtype=="stdin":return {"status":"ok","value":normalize_stdio(text),"raw_output":text},elapsed
    try:observation=json.loads(text)
    except json.JSONDecodeError:return {"status":"exception","exception":"OutputProtocolError"},elapsed
    if observation.get("status") not in ("ok","exception"):
        raise RuntimeError("invalid functional launcher protocol")
    return observation,elapsed


def matches(observation,expected,testtype):
    if observation["status"]!="ok":return False
    if testtype=="functional":
        try:return ast.literal_eval(observation["literal"])==json.loads(expected)
        except (ValueError,SyntaxError,RecursionError,MemoryError):return False
    actual_lines=[s.strip() for s in observation["raw_output"].strip().split("\n")]
    expected_lines=[s.strip() for s in expected.strip().split("\n")]
    if len(actual_lines)!=len(expected_lines):return False
    for actual,wanted in zip(actual_lines,expected_lines):
        if actual==wanted:continue
        try:
            if [Decimal(x) for x in actual.split()] != [Decimal(x) for x in wanted.split()]:return False
        except InvalidOperation:return False
    return True


def compact(observation):
    return {k:v for k,v in observation.items() if k in ("status","value","exception")}


def evaluate_task(job):
    question,tests,candidates,condition,timeout=job
    task=question["task_id"]
    name=hashlib.sha256(task.encode()).hexdigest()[:20]
    work=BASE/"runs/livecodebench_v6_increment/sandbox_sources"/condition
    work.mkdir(parents=True,exist_ok=True)
    observations=[]
    probe_times=[]
    sources=[]
    raw_public=[]
    for index,candidate in enumerate(candidates):
        source_path=work/(name+"_"+str(candidate["seed"])+".py")
        source_path.write_text(candidate["solution"],encoding="utf-8")
        sources.append(source_path)
        current,timings,raw=[],[],[]
        for case in tests["public"]:
            observation,elapsed=run_case(source_path,case["input"],case["testtype"],question["metadata"].get("func_name"),timeout)
            current.append(compact(observation));timings.append(elapsed);raw.append(observation)
        observations.append(current);probe_times.append(timings);raw_public.append(raw)
    public_passes=[matches(o,t["output"],t["testtype"]) for o,t in zip(raw_public[0],tests["public"])]
    public_failed=sum(not x for x in public_passes)
    private_checked=0
    private_failure=None
    private_time=0.
    if not public_failed:
        for index,case in enumerate(tests["private"]):
            observation,elapsed=run_case(sources[0],case["input"],case["testtype"],question["metadata"].get("func_name"),timeout)
            private_checked+=1;private_time+=elapsed
            if not matches(observation,case["output"],case["testtype"]):
                private_failure={"index":index,"status":observation["status"],"exception":observation.get("exception")}
                break
    token=candidates[0]["token_confidence"]
    error=int(bool(public_failed or private_failure is not None))
    row={"panel":"lcb_"+candidates[0]["model"],"condition":condition,"task_id":task,
         "model":candidates[0]["model"],"precision":candidates[0]["precision"],"dataset":"LiveCodeBench v6 increment",
         "difficulty":question["difficulty"],"interface":"functional" if question["metadata"].get("func_name") else "stdin",
         "error":error,"probe_count":len(tests["public"]),"probe_source":"public_examples_only",
         "observations":observations,"probe_execution_seconds":probe_times,
         "public_test_passes":public_passes,"public_failure":public_failed/len(public_passes),
         "private_test_count":len(tests["private"]),"private_tests_checked":private_checked,
         "private_first_failure":private_failure,"private_evaluation_seconds":private_time,
         "mean_nll":token["mean_nll"],"mean_topk_entropy":token["mean_topk_entropy"],
         "greedy_failure":failure_fraction(observations[0]),
         "greedy_generation_seconds":candidates[0]["elapsed_seconds"],
         "full_generation_seconds":sum(c["elapsed_seconds"] for c in candidates),
         "full_execution_seconds":sum(sum(x) for x in probe_times),
         "greedy_execution_seconds":sum(probe_times[0]),
         "solution_sha256":candidates[0]["solution_sha256"],
         "target_finish_reason":candidates[0]["finish_reason"],
         "created_at_utc":datetime.now(timezone.utc).isoformat()}
    for k in (1,2,4):
        row[f"dsde_{k}"]=dominant_semantic_distance_entropy(observations[0],observations[:k+1])
        row[f"generation_seconds_{k}"]=sum(c["elapsed_seconds"] for c in candidates[:k+1])
        row[f"execution_seconds_{k}"]=sum(sum(t) for t in probe_times[:k+1])
    row["exact_4"]=exact_dominant_semantic_distance_entropy(observations[0],observations)
    counts=Counter(json.dumps(x,sort_keys=True) for x in observations)
    row["cluster_entropy"]=-sum((n/5)*math.log(n/5) for n in counts.values())
    row["failure_then_nll"]=2*int(row["greedy_failure"]>0)+row["mean_nll"]/(1+row["mean_nll"])
    row["public_then_nll"]=2*int(public_failed>0)+row["mean_nll"]/(1+row["mean_nll"])
    row["public_then_dsde"]=2*int(public_failed>0)+row["dsde_4"]/(1+row["dsde_4"])
    return row


def main():
    if sys.platform!="linux":raise RuntimeError("run this evaluator inside WSL Linux")
    parser=argparse.ArgumentParser()
    parser.add_argument("--condition")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    config=json.loads((BASE/"protocol/validation_config.json").read_text())
    questions={q["task_id"]:q for q in read_jsonl(BASE/"data/livecodebench/questions.jsonl")}
    tests={q["task_id"]:q for q in read_jsonl(BASE/"data/livecodebench/tests.jsonl")}
    run=BASE/config["run_dir"]
    for model in config["models"]:
        condition=model["condition"]
        if args.condition and args.condition!=condition:continue
        if not (run/(condition+"_generation_complete.json")).exists():
            print("generation incomplete",condition,flush=True);continue
        output=run/"evaluations"/(condition+".jsonl")
        output.parent.mkdir(parents=True,exist_ok=True)
        done={r["task_id"] for r in read_jsonl(output)}
        candidates={task:[] for task in questions}
        for row in read_jsonl(run/"generations"/(condition+".jsonl")):candidates[row["task_id"]].append(row)
        jobs=[]
        for task in sorted(questions):
            if task in done:continue
            rows=sorted(candidates[task],key=lambda x:x["seed"])
            if [x["seed"] for x in rows]!=[0,11,29,47,71]:raise ValueError("incomplete candidate key set")
            jobs.append((questions[task],tests[task],rows,condition,config["execution_timeout_seconds"]))
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor,output.open("a",encoding="utf-8") as handle:
            futures=[executor.submit(evaluate_task,job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                row=future.result()
                handle.write(json.dumps(row,ensure_ascii=False)+"\n");handle.flush()
                done.add(row["task_id"])
                print(f"evaluated {condition} {len(done)}/{len(questions)} {row['task_id']}",flush=True)
        if len(done)!=len(questions):raise ValueError("incomplete evaluation")
        (run/(condition+"_evaluation_complete.json")).write_text(json.dumps({"records":len(done),"sha256":sha(output)},indent=2))


if __name__=="__main__":main()
