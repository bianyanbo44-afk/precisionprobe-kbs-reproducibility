from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parent
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from precisionprobe.inference import LlamaServer,append_jsonl,read_jsonl
from collect_token_confidence import summarize_logprobs


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def messages(question):
    functional=bool(question["metadata"].get("func_name"))
    instruction=("Implement the supplied Solution class and preserve the method name and signature. Include all standard-library imports and typing imports required by your code." if functional else "Write a complete Python program that reads standard input and writes only the required answer to standard output.")
    prompt=instruction+"\n\n"+question["question_content"]
    if question["starter_code"]:
        prompt+="\n\nRequired interface:\n"+question["starter_code"]
    return [{"role":"system","content":"You are a careful Python programmer. Return only one complete Python solution, without Markdown fences, tests or explanations."},
            {"role":"user","content":prompt}]


def extract(text):
    match=re.search(r"```(?:python)?\s*(.*?)```",text,flags=re.S|re.I)
    return (match.group(1) if match else text).strip()+"\n"


def verify_lock():
    lock=json.loads((BASE/"protocol/validation_lock.json").read_text())
    for relative,expected in lock["files"].items():
        # Large question/test files are included; model artifacts were separately verified at lock.
        if digest(ROOT/relative)!=expected:
            raise ValueError(f"locked file changed: {relative}")
    return lock


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--condition")
    args=parser.parse_args()
    lock=verify_lock()
    config=json.loads((BASE/"protocol/validation_config.json").read_text())
    questions=read_jsonl(BASE/"data/livecodebench/questions.jsonl")
    questions.sort(key=lambda q:hashlib.sha256((config["task_order_salt"]+"|"+q["task_id"]).encode()).hexdigest())
    run=BASE/config["run_dir"]
    run.mkdir(parents=True,exist_ok=True)
    specs=[("greedy",0,0.,1.)]+[("alternative",s,config["temperature"],config["top_p"]) for s in config["alternative_seeds"]]
    for model in config["models"]:
        condition=model["condition"]
        if args.condition and condition!=args.condition:continue
        output=run/"generations"/(condition+".jsonl")
        completed={(r["task_id"],r["seed"]) for r in read_jsonl(output)}
        if len(completed)==len(questions)*5:
            print("already complete",condition,flush=True);continue
        with LlamaServer(config["server"],model["path"],port=config["port"],
                         context_size=config["context_size"],log_path=run/"logs"/(condition+".log"),
                         completion_prompt_style=model["completion_prompt_style"]) as server:
            warmup=server.generate([{"role":"user","content":"Write a Python function add(a, b) that returns their sum. Return code only."}],seed=0,temperature=0,top_p=1,max_tokens=64,return_logprobs=True)
            summarize_logprobs(warmup["logprobs"])
            for i,q in enumerate(questions,1):
                for kind,seed,temp,topp in specs:
                    if (q["task_id"],seed) in completed:continue
                    generated=server.generate(messages(q),seed=seed,temperature=temp,top_p=topp,
                        max_tokens=config["max_tokens"],timeout_seconds=240,return_logprobs=kind=="greedy",top_logprobs=5)
                    source=extract(generated["text"])
                    row={"task_id":q["task_id"],"condition":condition,"model":model["model"],"precision":model["precision"],
                         "kind":kind,"seed":seed,"temperature":temp,"top_p":topp,"solution":source,
                         "solution_sha256":hashlib.sha256(source.encode()).hexdigest(),"raw_text":generated["text"],
                         "finish_reason":generated["finish_reason"],"usage":generated["usage"],"timings":generated["timings"],
                         "elapsed_seconds":generated["elapsed_seconds"],"created_at_utc":datetime.now(timezone.utc).isoformat(),
                         "model_sha256":model["sha256"],"protocol_lock_sha256":digest(BASE/"protocol/validation_lock.json")}
                    if kind=="greedy":row["token_confidence"]=summarize_logprobs(generated["logprobs"])
                    append_jsonl(output,row)
                    completed.add((q["task_id"],seed))
                status={"condition":condition,"completed_tasks":i,"total_tasks":len(questions),"updated_at_utc":datetime.now(timezone.utc).isoformat()}
                (run/"generation_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
                print(f"{condition} {i}/{len(questions)} {q['task_id']}",flush=True)
        (run/(condition+"_generation_complete.json")).write_text(json.dumps({"records":len(completed),"sha256":digest(output)},indent=2))
    print("generation command finished",flush=True)


if __name__=="__main__":main()
