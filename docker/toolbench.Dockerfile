# Canonical StableToolBench harness for gbench's `toolbench` suite, built LOCALLY (never pulled).
#
# Bundles StableToolBench (pinned) + the 236 MB response cache + the cached `/virtual` server, so
# `docker run` is self-contained. The entrypoint starts the server, runs the DFSDT loop against the
# model-under-test's served /v1 endpoint (backbone=chatgpt_function), converts the answer trees, and
# writes them to /out. The SoPR/SoWR JUDGE is NOT here - gbench scores /out with its Gemini cascade.
#
# LEAN DEPS: StableToolBench's requirements.txt pulls the full training stack (torch, transformers,
# vllm, deepspeed, bitsandbytes, peft, accelerate, langchain, gradio) for its LOCAL toolllama
# backbone - which we never use. deepspeed compiles CUDA at install and HANGS the build. We instead
# install only what the chatgpt_function inference + ToolEval convert + the cached server need
# (toolbench/utils.py imports torch+transformers at module load, so those two are required), and we
# STUB the 5 unused heavy backbones so their vllm/peft/etc. imports never load.
#
# The container runs the model's tool calls (the DFSDT loop dispatches them to the /virtual server
# in it), so gbench starts it on a docker bridge network after checking that the cloud metadata
# server is blocked there, and it holds no real key: the model endpoint and the simulator's Gemini
# API are reached through gbench's relays on the network's gateway (see toolbench_entrypoint.sh).
#
# Build:
#   docker build -t gbench-toolbench:aa4ed9f4-v5 -f docker/toolbench.Dockerfile docker
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2

ARG STB_REF=aa4ed9f4737ad98bd706663f01d63623c3427812
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONPATH=/stb

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/THUNLP-MT/StableToolBench /stb && \
    cd /stb && git checkout "${STB_REF}"
WORKDIR /stb

# Every pip install below resolves against toolbench_constraints.txt: the packages of the image
# checked before it existed (installed 2026-09-08), so a rebuild installs the same versions instead
# of the newest ones.
COPY toolbench_constraints.txt /tmp/toolbench_constraints.txt
ENV PIP_CONSTRAINT=/tmp/toolbench_constraints.txt

# torch (CPU) + transformers: required only because toolbench/utils.py does
# `import torch, transformers, transformers.models.llama.modeling_llama` at module load (the
# functions that USE them are toolllama-only and never run on the chatgpt_function path).
# NOTE: `--extra-index-url` (not `--index-url`) so PyPI stays available for build deps like
# flit_core; the `+cpu` local pin forces the small CPU wheel from the pytorch index.
RUN pip install --no-cache-dir torch==2.4.1+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu
# Pin transformers to a stable release: unpinned pulls a bleeding-edge build whose
# integrations/accelerate.py references nn.Module without importing nn (NameError at import).
# 4.44.2 has models.llama.modeling_llama and imports cleanly on torch 2.4.1+cpu.
RUN pip install --no-cache-dir transformers==4.44.2 tiktoken sentencepiece
# Light runtime: chatgpt_function inference, ToolEval convert, and the cached FastAPI server.
RUN pip install --no-cache-dir \
        openai tenacity tqdm termcolor requests httpx "pydantic<2" PyYAML \
        fastapi "uvicorn[standard]" slowapi backoff shortuuid rich numpy huggingface_hub
# The installed packages are exactly the constrained ones (the build fails otherwise); then the
# record of what this build installed: the Python set and the Debian packages (which float with
# the Debian archive), in /opt/gbench.
COPY pinned_pip_check.py /opt/gbench/pinned_pip_check.py
RUN python /opt/gbench/pinned_pip_check.py /tmp/toolbench_constraints.txt \
        --freeze /opt/gbench/pip-freeze.txt \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt

# Stub the 5 unused backbones so `rapidapi_multithread.py`'s eager imports succeed WITHOUT vllm /
# peft / deepspeed / sentence_transformers. Only chatgpt_function (real, openai-based) runs.
RUN set -e; for pair in \
        "tool_llama_model:ToolLLaMA" "tool_llama_vllm_model:ToolLLaMA_vllm" \
        "tool_llama_lora_model:ToolLLaMALoRA" "davinci_model:Davinci" "retriever:ToolRetriever"; do \
      f="${pair%%:*}"; cls="${pair##*:}"; \
      printf 'class %s:\n    """gbench stub: unused backbone (chatgpt_function path only)."""\n    def __init__(self, *a, **k):\n        raise RuntimeError("%s backbone is stubbed in the gbench toolbench image; use backbone_model=chatgpt_function")\n' \
        "$cls" "$cls" > "/stb/toolbench/inference/LLM/${f}.py"; \
    done

# The 236 MB response cache -> server/tool_response_cache + server/tools (deterministic execution),
# at the dataset revision that has held it since 2024-03-18 (stabletoolbench/Cache c6aca0c1, the
# file `resolve/main` served when the image was checked), checked against its sha256.
ARG STB_CACHE_REVISION=c6aca0c1cda74d54915eccabb60984dc083fb14c
ARG STB_CACHE_SHA256=a3ed72052c2dc961d8e9b71860374d90ded6d475ca6f6449c78fb248cb82a5b2
RUN curl -fsSL -o /tmp/server_cache.zip \
        "https://huggingface.co/datasets/stabletoolbench/Cache/resolve/${STB_CACHE_REVISION}/server_cache.zip" && \
    echo "${STB_CACHE_SHA256}  /tmp/server_cache.zip" | sha256sum -c - && \
    unzip -q /tmp/server_cache.zip -d /stb/server && rm /tmp/server_cache.zip && \
    ( [ -d /stb/server/tool_response_cache ] || \
      ( d=$(find /stb/server -maxdepth 2 -type d -name tool_response_cache | head -1); \
        [ -n "$d" ] && ln -s "$d" /stb/server/tool_response_cache ) ) && \
    ( [ -d /stb/server/tools ] || \
      ( d=$(find /stb/server -maxdepth 2 -type d -name tools | head -1); \
        [ -n "$d" ] && ln -s "$d" /stb/server/tools ) ) && true

# Patch StableToolBench's chat_completion_request: (1) route non-GPT models (our served
# google/gemma-4-...) through the OpenAI-compatible client instead of raising "Model not supported",
# (2) drop the interactive `pdb.set_trace()` that hangs/kills a non-interactive container, and
# (3) send the request through gbench's model request settings (toolbench_request_policy.py, on
# PYTHONPATH as /stb/toolbench_request_policy.py): the run's thinking switch, the operator's
# temperature, and with --thinking the run's budget in place of the 1024 max_tokens and a request
# timeout sized to it; failed requests are counted; (4) each DFSDT step with a model request that
# failed at the endpoint, or that the endpoint rejected for its content (HTTP 400/422, the model's
# outcome), is recorded with its test group and query id, and how it ended. Asserts each edit
# matched (build fails on upstream drift). The label tells
# gbench the image applies the settings; gbench refuses an image without it at this version.
COPY toolbench_request_policy.py /stb/toolbench_request_policy.py
COPY toolbench_patch_llm.py /tmp/toolbench_patch_llm.py
RUN python /tmp/toolbench_patch_llm.py \
    && python -c "import sys; sys.path[:0] = ['/stb', '/stb/toolbench/inference']; import toolbench.inference.LLM.chatgpt_function_model as m; assert m._GBENCH_POLICY.prefix == 'TB_'; assert m.ChatGPTFunction.parse.__name__ == '_gbench_parse'"
LABEL org.gbench.model-request-policy="2"

# Patch ToolEval's convert_to_answer_format.py: guard unguarded message['content'] reads so an
# assistant turn with no content key (valid OpenAI schema; model-output dependent) degrades to an
# empty node instead of crashing the judge with KeyError: 'content' (an intermittent rc=1).
COPY toolbench_patch_convert.py /tmp/toolbench_patch_convert.py
RUN python /tmp/toolbench_patch_convert.py

# Patch the cached server: the one model upstream's cache-miss simulator calls (CONFIG['model'],
# gpt-4-turbo) becomes gbench's Gemini cascade (GBENCH_JUDGE_MODELS, rounds, backoff), with
# upstream's request fields and JSON retries unchanged, except max_tokens: upstream's 1024 plus
# 8192 for the Gemini model's thinking (gemini_budget's allowance), with the replies still cut
# there counted in the call record; the real-API proxy call gets a read
# timeout; every /virtual call is recorded (answer type, the Gemini model that simulated it, a
# server error and the stage it failed in) in /out/toolbench_server_calls.jsonl. Patch the DFSDT
# client so each /virtual call carries its test group and query id for that record, and so the
# query's model client carries the query id for its record of failed model requests. Asserts each
# edit matched.
COPY toolbench_patch_simulator.py /tmp/toolbench_patch_simulator.py
COPY toolbench_patch_client.py /tmp/toolbench_patch_client.py
RUN python /tmp/toolbench_patch_simulator.py && python /tmp/toolbench_patch_client.py \
    && python -m py_compile /stb/server/main.py \
        /stb/toolbench/inference/Downstream_tasks/rapidapi_multithread.py

# Counts, per query, the harness-failure observations of the raw answer trees (the entrypoint runs
# it after the inference).
COPY toolbench_observations.py /stb/toolbench_observations.py

COPY toolbench_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# The image records the /virtual calls, the model requests that failed at the endpoint or that the
# endpoint rejected for their content (HTTP 400/422, the model's outcome, not endpoint failures),
# per query, and the answer trees' failure observations, reads the Gemini cascade and relay
# placeholders gbench passes, and runs the queries gbench selects by id (TB_QUERY_IDS_FILE, for
# --shard / --eval-limit), and sends the simulator upstream's max_tokens plus the Gemini thinking
# allowance, counting the replies still cut at it (version 5); gbench refuses an image without
# this label at this version.
LABEL org.gbench.toolbench-harness="5"
ENTRYPOINT ["/entrypoint.sh"]
