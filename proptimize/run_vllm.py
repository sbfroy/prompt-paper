import shlex
from pathlib import Path
import subprocess
import os
from typing import Optional
from time import sleep

from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file


def start_daemon(
    cmd: str,
    pidfile: Optional[str] = None,
    logfile: Optional[str] = None,
    cwd: Optional[str] = None,
) -> int:
    """
    Start a command as a detached daemon, optionally writing a pidfile and redirecting output to logfile.
    Returns the PID of the started process.
    """
    # Open logfile or /dev/null for child stdout/stderr
    if logfile:
        out = open(logfile, "ab")
        err = out
    else:
        out = open(os.devnull, "wb")
        err = out

    # Start the process detached from the controlling terminal
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=out,
        stderr=err,
        cwd=cwd,
        preexec_fn=os.setsid,
        close_fds=True,
    )

    pid = proc.pid

    # Write pidfile if requested (best-effort)
    if pidfile:
        try:
            Path(pidfile).write_text(str(pid))
        except Exception:
            # Intentionally ignore filesystem errors here; caller can handle if needed
            pass

    return pid


def run_script_daemon(
    script_path: str,
    cwd: str | None = None,
    pidfile: str | None = None,
    logfile: str | None = None,
) -> int:
    """
    Run a bash script as a detached daemon using start_daemon().
    Ensures the script exists and is executable. If cwd is not provided,
    the script's parent directory is used.
    Returns the PID of the started daemon.
    """
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    script = script.resolve()

    # Make executable if not already
    mode = script.stat().st_mode
    if not (mode & 0o111):
        script.chmod(mode | 0o111)

    # Use the script's directory as cwd by default
    if cwd is None:
        cwd = str(script.parent)

    # Quote the script path to be safe when using shell=True in start_daemon
    cmd = f"bash {shlex.quote(str(script))}"

    return start_daemon(cmd, pidfile=pidfile, logfile=logfile, cwd=cwd)


def start_vllm_servers() -> int:
    LLM_server_script = Path("/workspace/scripts/run_vllm_oss120.sh")
    EMBEDD_server_script = Path("/workspace/scripts/run_vllm_qwen3_embedd.sh")

    assert (
        LLM_server_script.exists()
    ), f"LLM server script not found: {LLM_server_script}"
    assert (
        EMBEDD_server_script.exists()
    ), f"Embedding server script not found: {EMBEDD_server_script}"

    run_script_daemon(
        script_path=str(LLM_server_script),
    )

    run_script_daemon(
        script_path=str(EMBEDD_server_script),
    )

    print("#### Waiting for VLLM servers to start... ####")

    while True:

        base_url = f"http://localhost:{os.getenv('LLM_PORT')}/v1"
        api_key = os.getenv("LLM_API_KEY")
        model_name = os.getenv("LLM_MODEL")

        try:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model=model_name,
            )

            if response and isinstance(response.choices[0].message.content, str):
                print(f"#### {model_name} servers are up and running! ####")
                break
        except Exception as e:
            print(f"Waiting for {model_name} servers to be ready... {e}")

        sleep(10)

    print("#### Waiting for Embedding server to start... ####")

    while True:
        base_url = f"http://localhost:{os.getenv('EMBEDD_PORT')}/v1"
        api_key = os.getenv("EMBEDD_API_KEY")
        model_name = os.getenv("EMBEDD_MODEL")

        try:
            client = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                tiktoken_enabled=True,
            )
            response = client.embed_documents(["test embedding"])

            if response and isinstance(response[0], list):
                print(f"#### {model_name} server is up and running! ####")
                break
        except Exception:
            print(f"Waiting for {model_name} server to be ready... {e}")

        sleep(10)

    print()
    print("#######################################")
    print("#### All VLLM servers are running! ####")
    print("#######################################")
    print()

    return 0
