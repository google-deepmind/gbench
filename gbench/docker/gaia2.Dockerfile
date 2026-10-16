# Canonical GAIA2 / Meta-ARE orchestrator image for gbench's `gaia2` suite, built LOCALLY.
#
# GAIA2 (Meta Agents Research Environments, github.com/facebookresearch/meta-agents-research-environments)
# is a STATEFUL multi-turn agentic benchmark: the agent acts inside an IN-PROCESS Python simulator
# (contacts/calendar/email/files with time-sensitive + async events), issuing tool calls over many
# turns; scoring is HYBRID - deterministic "hard validation" + a load-bearing LLM judge for
# soft/semantic checks. Unlike skillsbench/wildclawbench there is NO docker-out-of-docker and NO
# sibling task containers here: `are-benchmark` runs the whole simulation in this one container. It
# MUST be a container (not the host environment): ARE 1.2.0 hard-pins ~22 deps with `==` (numpy==2.2.6,
# litellm==1.71.1, mcp==1.11.0, pydantic==2.10.6, ...) that collide with the vLLM/torch stack.
#
# Build (context = gbench/docker):
#   docker build -t gbench-gaia2:are-1.2.0 -f gbench/docker/gaia2.Dockerfile gbench/docker
#
# The container runs on a docker bridge network (the ARE agent's tool calls run in it, so the gbench
# runner checks that the cloud metadata server is blocked there first) and holds no real key: the
# model-under-test is reached through gbench's endpoint relay and the Gemini API through gbench's
# Gemini relay, both on the network's gateway. The judge is swapped to gbench's Gemini cascade via
# an in-container OpenAI-compat proxy (gaia2_cascade_judge.py, on 127.0.0.1 only, serving judge
# calls through wildclawbench's cascade module) with ARE's judge PROMPTS untouched; no knob points
# the judge anywhere else. The launcher's hooks apply the ARE paper's agent protocol and the
# upstream judge path (gaia2_run.py). A graded run needs: a reachable served model +
# GEMINI_API_KEY (judge, kept by the gbench runner). The public cc-by-4.0 datasets are BAKED at build
# at pinned revisions and read offline (no HF token, no Hub request at run time). Nothing contacts a
# model or the Gemini API at build time. See docs/evals/gaia2.md.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*

# Pin the ARE harness. Its 22 hard `==` pins resolve inside this isolated image only; the rest of
# its closure resolves against gaia2.constraints.txt (the resolved set of the image whose runs
# were checked), and the build fails unless the interpreter holds exactly that set. The Python
# set and the Debian packages (which float with the Debian archive) are recorded in /opt/gbench.
ARG ARE_VERSION=1.2.0
COPY gaia2.constraints.txt /opt/gbench/gaia2.constraints.txt
RUN PIP_CONSTRAINT=/opt/gbench/gaia2.constraints.txt \
        pip install "meta-agents-research-environments==${ARE_VERSION}" \
    && python -m pip freeze --all | LC_ALL=C sort > /opt/gbench/pip-freeze.txt \
    && grep -v -e '^#' -e '^$' /opt/gbench/gaia2.constraints.txt | LC_ALL=C sort \
        | diff - /opt/gbench/pip-freeze.txt \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt
ENV ARE_VERSION=1.2.0

# Bake the public GAIA2 datasets (cc-by-4.0, anonymous download - no token) as local copies at pinned
# revisions: the scenarios (`gaia2`, all 7 configs) and the filesystem-app assets
# (`gaia2_filesystem`). The container then runs with the Hub off (HF_HUB_OFFLINE /
# HF_DATASETS_OFFLINE): ARE's loader reads the local gaia2 copy (the launcher passes it as
# --hf-dataset; load_dataset(<dir>, name=<config>, streaming=True) reads the configs from its
# README) and the filesystem app's fallback reads the local gaia2_filesystem copy (ARE's
# config.DEMO_FS_PATH, an env var), so a run makes no Hub request. Online, both made anonymous
# paths-info requests that the Hub rate-limited (429), which dropped whole capability configs.
# A plain `python -c` (not a RUN heredoc, which the classic docker builder runs with an empty
# stdin); the second RUN fails the build unless every config loads offline with rows.
ARG GAIA2_REVISION=78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99
ARG GAIA2_FS_REVISION=132e26376f5e963bb59f64bcccdd02188cb08dee
RUN python -c "from huggingface_hub import snapshot_download; \
print('gaia2 at', snapshot_download('meta-agents-research-environments/gaia2', repo_type='dataset', \
      revision='${GAIA2_REVISION}', local_dir='/opt/gaia2/gaia2')); \
print('gaia2_filesystem at', snapshot_download('meta-agents-research-environments/gaia2_filesystem', \
      repo_type='dataset', revision='${GAIA2_FS_REVISION}', local_dir='/opt/gaia2/gaia2_filesystem'))"
ENV GAIA2_LOCAL_DATASET=/opt/gaia2/gaia2 \
    GAIA2_DATASET_REVISION=${GAIA2_REVISION} \
    GAIA2_FS_REVISION=${GAIA2_FS_REVISION} \
    DEMO_FS_PATH=/opt/gaia2/gaia2_filesystem/demo_filesystem \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1
RUN python -c "import os; from datasets import load_dataset; \
counts = {c: sum(1 for _ in load_dataset(os.environ['GAIA2_LOCAL_DATASET'], name=c, streaming=True)['validation']) \
          for c in ('adaptability', 'ambiguity', 'execution', 'search', 'time', 'mini', 'demo')}; \
print('gaia2 offline rows', counts); assert all(counts.values()), counts; \
assert os.path.isdir(os.environ['DEMO_FS_PATH']), os.environ['DEMO_FS_PATH']"

# gbench adapter: the launcher (drives are-benchmark with gbench's harness-error hooks + parses its
# statistics), the Gemini judge proxy (gaia2_cascade_judge.py) with the cascade module it imports
# (wildclawbench_cascade_judge.py), and the entrypoint.
RUN mkdir -p /app/adapter
COPY gaia2_run.py /app/adapter/gaia2_run.py
COPY wildclawbench_cascade_judge.py /app/adapter/wildclawbench_cascade_judge.py
COPY gaia2_cascade_judge.py /app/adapter/gaia2_cascade_judge.py
COPY gaia2_entrypoint.sh /app/adapter/gaia2_entrypoint.sh
RUN chmod +x /app/adapter/gaia2_entrypoint.sh \
    && python3 -m py_compile /app/adapter/gaia2_run.py /app/adapter/gaia2_cascade_judge.py \
        /app/adapter/wildclawbench_cascade_judge.py
# A --thinking agent request is streamed and read from the raw stream LiteLLM wraps (the hook
# closes it at the first stop sequence in the answer, gaia2_run.py streamed_reply); a non-streamed
# reply's token ids and stop_reason are kept by wrapping the openai provider's module global
# convert_to_model_response_object (gaia2_run.py install_token_ids_hook). The build fails if the
# pinned LiteLLM no longer hands the stream over, no longer has the Timeout the hook raises, or no
# longer converts openai replies through that global.
RUN python3 -c "import inspect; \
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper; \
from litellm.exceptions import Timeout; \
from litellm.llms.openai import openai as o; \
assert 'completion_stream' in inspect.signature(CustomStreamWrapper.__init__).parameters; \
assert Timeout(message='m', model='m', llm_provider='openai').status_code == 408; \
assert callable(o.convert_to_model_response_object); \
assert 'convert_to_model_response_object(' in inspect.getsource(o.OpenAIChatCompletion.completion)"

# The launcher runs ARE with gbench's hooks: the paper's agent protocol (no server-side `stop` on a
# --thinking run, where the server would match it inside the reasoning: the agent request is
# streamed and closed at the first stop sequence in the answer), the verbatim judge reply with the
# case-insensitive verdict match, the measured generation time, the server-parser policy (every
# agent request asks for the token ids; text the server's parsers took out of a reply, including
# text a tool-call parser deleted after the reply's returned reasoning or content, is recovered
# from them and handed to ARE as the model's output; deleted text that cannot be recovered is a
# harness error), --shard (GAIA2_SHARD: each config's scenarios cut to the shard in the stream ARE
# reads), --eval-limit dealt over the configs (GAIA2_CONFIG_LIMITS, label 7: a config with 0 is
# not run), the scenario runs with a harness error (a model endpoint call that failed with
# any status but 400/422 or with none, a judge call that failed, a judge reply without a verdict,
# an agent reply lost to the server's tool-call parser, a validation or run that raised) left out
# of the success rates, every capability config's outcome (a config ARE skipped after it raised)
# and the macro over the capabilities with a validated run (gbench_stats.json), and the datasets
# read offline from the local copies; the gbench runner refuses an image without this label at
# this version.
LABEL org.gbench.gaia2-harness-errors="7"

ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/gaia2_entrypoint.sh"]
