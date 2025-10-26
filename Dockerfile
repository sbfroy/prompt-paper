FROM vllm/vllm-openai:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC


WORKDIR /workspace
COPY . .

# RUN apt-get update && apt-get install -y python3-tk
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -e .


ENV WANDB_PROJECT="icl-research-team"


# Run the script with unbuffered output; if it exits, keep the container alive for debugging.
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["python3 -u /workspace/tasks/financial_ner/main.py || tail -f /dev/null"]
