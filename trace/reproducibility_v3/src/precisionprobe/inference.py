from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


class LlamaServer:
    def __init__(
        self,
        executable: str | Path,
        model_path: str | Path,
        *,
        port: int,
        log_path: str | Path,
        gpu_layers: int = 99,
        context_size: int = 4096,
        chat_template: str | None = None,
        completion_prompt_style: str | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.model_path = Path(model_path)
        self.port = port
        self.log_path = Path(log_path)
        self.gpu_layers = gpu_layers
        self.context_size = context_size
        self.chat_template = chat_template
        self.completion_prompt_style = completion_prompt_style
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_seconds: float = 180.0) -> None:
        if not self.executable.exists():
            raise FileNotFoundError(self.executable)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        command = [
            str(self.executable),
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--n-gpu-layers",
            str(self.gpu_layers),
            "--parallel",
            "1",
            "--jinja",
        ]
        if self.chat_template:
            command.extend(["--chat-template", self.chat_template])
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}; see {self.log_path}")
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(f"llama-server did not become healthy; see {self.log_path}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
        self.process = None
        self._log_handle = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float = 180.0,
        return_logprobs: bool = False,
        top_logprobs: int = 5,
    ) -> dict[str, Any]:
        if self.completion_prompt_style == "deepseek_coder":
            user_content = "\n\n".join(
                message["content"] for message in messages if message["role"] == "user"
            )
            prompt = (
                "You are an AI programming assistant, utilizing the Deepseek Coder model, "
                "developed by Deepseek Company, and you only answer questions related to "
                "computer science. For politically sensitive questions, security and privacy "
                "issues, and other non-computer science questions, you will refuse to answer.\n"
                f"### Instruction:\n{user_content}\n### Response:\n"
            )
            payload = {
                "model": self.model_path.name,
                "prompt": prompt,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stop": ["<|EOT|>"],
                "stream": False,
            }
            endpoint = "/v1/completions"
        elif self.completion_prompt_style:
            raise ValueError(f"unsupported completion prompt style: {self.completion_prompt_style}")
        else:
            payload = {
                "model": self.model_path.name,
                "messages": messages,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": False,
            }
            endpoint = "/v1/chat/completions"
        if return_logprobs:
            if top_logprobs < 1:
                raise ValueError("top_logprobs must be positive")
            if endpoint == "/v1/completions":
                payload["logprobs"] = int(top_logprobs)
            else:
                payload["logprobs"] = True
                payload["top_logprobs"] = int(top_logprobs)
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        elapsed = time.perf_counter() - started
        choice = body["choices"][0]
        text = choice["text"] if endpoint == "/v1/completions" else choice["message"]["content"]
        return {
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage", {}),
            "timings": body.get("timings", {}),
            "elapsed_seconds": elapsed,
            "request": payload,
            "logprobs": choice.get("logprobs"),
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
