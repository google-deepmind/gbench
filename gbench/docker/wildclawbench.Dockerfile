# Canonical WildClawBench orchestrator image for gbench's `wildclawbench` suite, built LOCALLY.
#
# WildClawBench (github.com/internlm/WildClawBench) is a live, long-horizon agentic benchmark: 60
# tasks, each run in its own Docker container from the prebaked image `wildclawbench-ubuntu:v1.3`
# (a 13.5GB `docker load` from HF), with the OpenClaw agent inside and per-task grading (programmatic
# checks + an LLM judge) run afterwards by `docker exec`. This orchestrator bakes the pinned InternLM
# harness's Python runtime + the gbench adapter, and drives `eval/run_batch.py` docker-OUT-of-docker
# (mount /var/run/docker.sock at run time). The OpenClaw agent is pointed at the gbench /v1 endpoint
# via an injected `my_api.json` custom provider whose baseUrl is the model relay of the proxy this
# image runs (wildclawbench_cascade_judge.py), which adds the model endpoint's key, so no model key
# reaches a task container. Every external model or search service the harness calls, or offers
# the agent as a tool, goes through the Gemini API via that proxy: the tasks' verbatim OpenAI-SDK
# judge calls (the gbench Gemini judge cascade instead of OpenRouter/gpt-5.4; Gemini Google Search
# grounding for the one judge call that asks for OpenRouter's web search) and the OpenClaw agent's
# web_search tool (Gemini Google Search grounding behind the Brave request path of the image's
# OpenClaw, whose tool definition is unchanged). The launcher also deletes the task image's other
# search skills (DuckDuckGo: local-web-search, web-search; Google Custom Search: google-search) from
# every task container, stops the processes the agent left in a task container before its grading
# starts, and runs a network monitor container (the task image, gbench's own script) on the task
# network for the whole run. The proxy loads gbench's search_tool.py, which the gbench runner mounts
# read-only at $WILDCLAW_SEARCH_TOOL_DIR at run time.
#
# Build (context = gbench/docker):
#   docker build -t gbench-wildclawbench:316334cc-v9 -f gbench/docker/wildclawbench.Dockerfile gbench/docker
#
# PROVISIONED HOST CHECKOUT (swe_lancer-style): the large, partly un-redistributable workspace (3
# YouTube videos) and the 13.5GB task image are NOT baked. A graded run needs, on the host:
#   * a WildClawBench checkout with its HF `workspace/` downloaded + `bash script/prepare.sh` run
#     (videos + SAM3 weights); this dir is IDENTITY-mounted into the orchestrator (DooD requires the
#     workspace bind-mount source to resolve on the host);
#   * `docker load` of wildclawbench-ubuntu:v1.3 on the host daemon;
#   * GEMINI_API_KEYS or GEMINI_API_KEY (judge cascade + web search grounding), and a served model at
#     /v1 reachable from this host (the orchestrator's proxy relays the agent's model calls).
# Nothing contacts a model or the Gemini API at build time.
# See docs/evals/wildclawbench.md.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Base: git, curl, python, and the Docker CLI (client only - it talks to the mounted host daemon)
# at a pinned version, from the Ubuntu archive as of UBUNTU_SNAPSHOT (snapshot.ubuntu.com, apt >=
# 2.7), so a rebuild installs the same package versions instead of whatever the archive serves
# that day. The snapshot service is https-only, so ca-certificates is installed first from the
# base image's archive and then again at the snapshot's version (--allow-downgrades). The
# installed package versions are recorded in /opt/gbench/dpkg-versions.txt.
ARG UBUNTU_SNAPSHOT=20260925T120000Z
ARG DOCKER_CLI_VERSION=5:29.8.1-1~ubuntu.24.04~noble
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i -E "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?\$#URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" /etc/apt/sources.list.d/ubuntu.sources \
    && test "$(grep -c "snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" /etc/apt/sources.list.d/ubuntu.sources)" = 2 \
    && apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        ca-certificates curl git gnupg python3 python3-pip \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends "docker-ce-cli=${DOCKER_CLI_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version \
    && mkdir -p /opt/gbench \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt \
    && grep -qx "docker-ce-cli=${DOCKER_CLI_VERSION}" /opt/gbench/dpkg-versions.txt

# The only third-party deps run_batch.py's openclaw path needs at import + run time are python-dotenv
# and pyyaml (weasyprint/pymupdf/yt-dlp/modelscope in the harness requirements are HOST-side
# provisioning / task-image-side grading tools, not orchestrator deps). The judge proxy is stdlib.
# Installed against wildclawbench.constraints.txt (the resolved set of the image whose runs were
# checked); the build fails unless the interpreter holds exactly that set, which is recorded.
COPY wildclawbench.constraints.txt /opt/gbench/wildclawbench.constraints.txt
RUN PIP_CONSTRAINT=/opt/gbench/wildclawbench.constraints.txt \
        pip3 install --no-cache-dir --break-system-packages python-dotenv pyyaml \
    && python3 -m pip freeze --all | LC_ALL=C sort > /opt/gbench/pip-freeze.txt \
    && grep -v -e '^#' -e '^$' /opt/gbench/wildclawbench.constraints.txt | LC_ALL=C sort \
        | diff - /opt/gbench/pip-freeze.txt
LABEL org.gbench.docker_cli="${DOCKER_CLI_VERSION}" org.gbench.ubuntu_snapshot="${UBUNTU_SNAPSHOT}"

# Bake the pinned harness as a provenance/version reference (the graded run uses the IDENTITY-mounted
# host checkout, not this copy - a container path can't be a host bind-mount source). WILDCLAW_REF
# is the SHA gbench pins and docs against.
ARG WILDCLAW_REF=316334ccc4a87b9b5635ad73da99b4dfc0b3887e
RUN git clone https://github.com/internlm/WildClawBench /opt/wildclawbench-ref \
    && cd /opt/wildclawbench-ref && git checkout "${WILDCLAW_REF}" \
    && echo "${WILDCLAW_REF}" > /opt/wildclawbench-ref/.gbench_pinned_ref
ENV WILDCLAW_REF=316334ccc4a87b9b5635ad73da99b4dfc0b3887e

# gbench adapter: the launcher (drives run_batch + parses the score tree), the Gemini proxy (judge
# cascade + agent web search + model relay), and the entrypoint. /app/gbench is where the runner
# mounts search_tool.py.
RUN mkdir -p /app/adapter /app/gbench
COPY wildclawbench_run.py /app/adapter/wildclawbench_run.py
COPY wildclawbench_cascade_judge.py /app/adapter/wildclawbench_cascade_judge.py
COPY wildclawbench_entrypoint.sh /app/adapter/wildclawbench_entrypoint.sh
RUN chmod +x /app/adapter/wildclawbench_entrypoint.sh \
    && python3 -m py_compile /app/adapter/wildclawbench_run.py /app/adapter/wildclawbench_cascade_judge.py

# The launcher and the proxy report harness errors (task time windows and container addresses, the
# proxy's harness events with the requesting container, the time the request arrived and whether
# the Gemini service failed it, the grader's judge calls told apart from the agent's, missing
# provisioned inputs, the processes left in a task container stopped before its grading and why a
# stop failed, every grade() cap hit, the network monitor of the task network with DNS evidence for
# log-only lookups, harness_errors and the answering Gemini models per task in the summary), and
# the proxy forwards each judge request's own temperature, sends a judge request that turns
# thinking off at Gemini thinking level low (the lowest level every model of the default cascade
# accepts) and counts it, and tells a model that is not served from a content refusal; the
# launcher stops on a malformed GBENCH_SHARD and writes an empty-shard summary (nothing run) for a
# shard that holds none of the selected tasks; the launcher sets OpenClaw's web_search timeout to
# 60 s (tools.web.search.timeoutSeconds; OpenClaw's default is 30 s; 2026-10-13, version 9) and the
# proxy gives an agent web search a deadline inside it (57 s) and records an answer that did not reach the agent in time as
# a harness event, the launcher counts the web_search calls OpenClaw aborted as harness errors,
# classifies a timed-out stop by what still runs in the container, records the max_tokens OpenClaw
# sent, the replies cut off at it and the token usage the model relay saw, and the entrypoint hands
# the workdir to the runner's user (WILDCLAW_HOST_OWNER) at exit; the proxy sends each chat
# request's max_tokens plus the Gemini thinking allowance (GEMINI_THINKING_ALLOWANCE), returns the
# grounded path's real finish reason and counts the replies still cut at the raised cap; the gbench
# runner refuses an image without this label at this version.
LABEL org.gbench.wildclawbench-harness-errors="9"

ENV ADAPTER_DIR=/app/adapter
ENV WILDCLAW_SEARCH_TOOL_DIR=/app/gbench
ENTRYPOINT ["/app/adapter/wildclawbench_entrypoint.sh"]
