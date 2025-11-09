"""VLLM server management utilities for starting and managing LLM and embedding servers."""

import shlex
from pathlib import Path
import subprocess
import os
import signal
from typing import Optional
from time import sleep

from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Global variable to track embedding server process
_EMBEDDING_SERVER_PID = None


def start_daemon(
    cmd: str,
    pidfile: Optional[str] = None,
    logfile: Optional[str] = None,
    cwd: Optional[str] = None,
) -> int:
    """Start a command as a detached daemon process.
    
    Args:
        cmd: Shell command to execute
        pidfile: Optional path to write process ID
        logfile: Optional path for stdout/stderr output
        cwd: Optional working directory
        
    Returns:
        Process ID of the started daemon
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

    # Write pidfile if requested
    if pidfile:
        try:
            Path(pidfile).write_text(str(pid))
        except Exception:
            pass  # Ignore filesystem errors

    return pid


def run_script_daemon(
    script_path: str,
    cwd: str | None = None,
    pidfile: str | None = None,
    logfile: str | None = None,
) -> int:
    """Run a bash script as a detached daemon.
    
    Args:
        script_path: Path to the bash script
        cwd: Working directory (defaults to script's parent directory)
        pidfile: Optional path to write process ID
        logfile: Optional path for stdout/stderr output
        
    Returns:
        Process ID of the started daemon
        
    Raises:
        FileNotFoundError: If script doesn't exist
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

    # Quote the script path for safe shell execution
    cmd = f"bash {shlex.quote(str(script))}"

    return start_daemon(cmd, pidfile=pidfile, logfile=logfile, cwd=cwd)


def start_vllm_servers() -> int:
    """Start both LLM and embedding VLLM servers as daemons.
    
    Launches server scripts and waits for them to be ready by testing
    connectivity with sample requests.
    
    Returns:
        0 on success
        
    Raises:
        AssertionError: If server scripts are not found
    """
    global _EMBEDDING_SERVER_PID

    LLM_server_script = Path("/workspace/scripts/run_vllm_oss120.sh")
    EMBEDD_server_script = Path("/workspace/scripts/run_vllm_qwen3_embedd.sh")

    assert LLM_server_script.exists(), \
        f"LLM server script not found: {LLM_server_script}"
    assert EMBEDD_server_script.exists(), \
        f"Embedding server script not found: {EMBEDD_server_script}"

    # Start LLM server
    run_script_daemon(script_path=str(LLM_server_script))

    # Start embedding server and store PID
    _EMBEDDING_SERVER_PID = run_script_daemon(script_path=str(EMBEDD_server_script))

    print("#### Waiting for VLLM servers to start... ####")

    # Wait for LLM server to be ready
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

    # Wait for embedding server to be ready
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
        except Exception as e:
            print(f"Waiting for {model_name} server to be ready... {e}")

        sleep(10)

    print()
    print("#######################################")
    print("#### All VLLM servers are running! ####")
    print("#######################################")
    print()

    return 0


def shutdown_embedding_server():
    """Shutdown the embedding server to free GPU resources.
    
    Should be called after embeddings are generated and cached.
    Sends SIGTERM to the embedding server process group.
    """
    global _EMBEDDING_SERVER_PID

    if _EMBEDDING_SERVER_PID is not None:
        os.killpg(
            _EMBEDDING_SERVER_PID,  # Process group ID
            signal.SIGTERM          # Termination signal
        )
        _EMBEDDING_SERVER_PID = None