FROM vllm/vllm-openai:v0.11.0

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC


WORKDIR /workspace
COPY . .
COPY ".env" ".env"

# RUN apt-get update && apt-get install -y python3-tk
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -e .


ENV WANDB_PROJECT="icl-research-team"


# Run the script with unbuffered output; if it exits, keep the container alive for debugging.
ENTRYPOINT ["/bin/bash", "-lc"]

# CMD ["python3 -u /workspace/tasks/financial_ner/run_generate.py || tail -f /dev/null"]
# CMD ["python3 -u /workspace/tasks/financial_ner/run_cluster.py || tail -f /dev/null"]
CMD ["python3 -u /workspace/tasks/financial_ner/run_evolve.py || tail -f /dev/null"]


