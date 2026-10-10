from __future__ import annotations

import base64
import hashlib
import io
import json
import pickle
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("non-primitive pickle is forbidden")

    def persistent_load(self, pid):
        raise ValueError("persistent pickle reference is forbidden")


def decode_private(value):
    try:
        data=json.loads(value)
    except json.JSONDecodeError:
        unpacked=PrimitiveUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(value)))).load()
        data=json.loads(unpacked) if isinstance(unpacked,str) else unpacked
    if not isinstance(data,list) or not all(isinstance(t,dict) for t in data):
        raise ValueError("unexpected test payload")
    return data


def main():
    lock_path=BASE/"protocol/validation_lock.json"
    if lock_path.exists():
        raise RuntimeError("Protocol already locked; do not overwrite it")
    raw=BASE/"data/livecodebench/test6.jsonl"
    if sha(raw)!="bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5":
        raise ValueError("dataset hash mismatch")
    problems=[json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(problems)!=175:
        raise ValueError("unexpected panel size")
    question_path=BASE/"data/livecodebench/questions.jsonl"
    test_path=BASE/"data/livecodebench/tests.jsonl"
    with question_path.open("w",encoding="utf-8") as fq,test_path.open("w",encoding="utf-8") as ft:
        for p in problems:
            public=json.loads(p["public_test_cases"])
            private=decode_private(p["private_test_cases"])
            task_id=p["platform"]+"/"+p["question_id"]
            question={k:p[k] for k in ("question_title","question_content","starter_code","difficulty","platform","contest_date")}
            question.update(task_id=task_id,metadata=json.loads(p["metadata"]),public_inputs=[{"input":t["input"],"testtype":t["testtype"]} for t in public])
            fq.write(json.dumps(question,ensure_ascii=False)+"\n")
            ft.write(json.dumps({"task_id":task_id,"public":public,"private":private},ensure_ascii=False)+"\n")
    models=[]
    sources=[("qwen3","runs/rapid_qwen3_confirm60/run_manifest.json",None),
             ("qwen25","runs/quartz_qwen_mbpp/run_manifest.json",None),
             ("deepseek","runs/quartz_deepseek_humaneval/run_manifest.json","deepseek_coder")]
    for name,relative,style in sources:
        old=json.loads((ROOT/relative).read_text(encoding="utf-8"))
        for precision in ("q4","q8"):
            model=dict(old["models"][precision])
            model_path=Path(model["path"])
            if not model_path.is_absolute():model_path=ROOT/model_path
            actual=sha(model_path)
            if actual.lower()!=model["sha256"].lower():
                raise ValueError(f"model hash mismatch: {name}/{precision}")
            models.append({"condition":name+"_"+precision,"model":name,"precision":precision,"path":str(model_path),"sha256":actual,"completion_prompt_style":style})
            print("verified",name,precision,flush=True)
    executable=ROOT/".tools/llama.cpp-b9637/llama-server.exe"
    config={"dataset":"LiveCodeBench-v6-increment","n_tasks":175,"models":models,
        "server":str(executable),"server_sha256":sha(executable),"port":18171,"context_size":6144,
        "max_tokens":1024,"alternative_seeds":[11,29,47,71],"temperature":.6,"top_p":.95,
        "task_order_salt":"eswa-20261009-generation-order","execution_timeout_seconds":4.0,
        "run_dir":"runs/livecodebench_v6_increment"}
    config_path=BASE/"protocol/validation_config.json"
    config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    files=[BASE/"protocol/validation_protocol.md",config_path,question_path,test_path,
           BASE/"scripts/generate_validation.py",BASE/"scripts/evaluate_validation.py",
           BASE/"scripts/sandbox_function.py",ROOT/"src/precisionprobe/inference.py",ROOT/"src/precisionprobe/scoring.py"]
    lock={"locked_at_utc":datetime.now(timezone.utc).isoformat(),"local_date":"2026-10-09 Asia/Shanghai",
          "status":"locked_before_any_validation_model_generation","files":{str(p.relative_to(ROOT)):sha(p) for p in files},
          "generation_records_at_lock":0}
    run_dir=BASE/config["run_dir"]
    if run_dir.exists() and list(run_dir.glob("generations/*.jsonl")):
        raise RuntimeError("generation records already exist")
    lock_path.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print(json.dumps({"lock":str(lock_path),"models":len(models),"tasks":175}),flush=True)


if __name__=="__main__":main()

