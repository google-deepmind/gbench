# Canonical SkillsBench orchestrator image for gbench's `skillsbench` suite, built LOCALLY.
#
# SkillsBench (benchflow-ai/skillsbench) is a DETERMINISTIC, execution-based agentic benchmark:
# 87 tasks, each a self-contained per-task Docker environment scored by a pytest verifier that
# writes /logs/verifier/reward.txt (reward in [0,1]) - NO LLM judge anywhere. This orchestrator
# bakes the BenchFlow runner (`bench` CLI, via `uv sync`) + the pinned task set, and drives it
# docker-OUT-of-docker: `bench eval run --backend docker` builds/runs each TASK container as a
# sibling on the host daemon (mount /var/run/docker.sock at run time). The OpenCode agent runs
# INSIDE each task container, pointed at the gbench /v1 endpoint via an injected openai-compatible
# provider; the deterministic verifier produces the reward. No judge, no GEMINI key.
#
# Build (context = gbench/docker):
#   docker build -t gbench-skillsbench:9a1f4dd5-h7 -f gbench/docker/skillsbench.Dockerfile gbench/docker
#
# A graded run needs: a reachable Docker daemon (socket mounted), network egress (the ~28 tasks
# that fetch public toolchains/data at build + the OpenCode agent install), and a served model at
# /v1 reachable FROM the sibling task containers. The model-free `oracle` agent needs only Docker
# + egress. See docs/evals/skillsbench.md.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Base: git, curl, python, and the Docker CLI with the compose plugin (clients only - they talk to
# the mounted host daemon; BenchFlow's docker sandbox drives `docker compose` (v2), which the CLI
# alone lacks) at pinned versions, from the Ubuntu archive as of UBUNTU_SNAPSHOT
# (snapshot.ubuntu.com, apt >= 2.7), so a rebuild installs the same package versions instead of
# whatever the archive serves that day. The snapshot service is https-only, so ca-certificates is
# installed first from the base image's archive and then again at the snapshot's version
# (--allow-downgrades). The installed package versions are recorded in
# /opt/gbench/dpkg-versions.txt. The system python3 (3.12) is the interpreter `uv sync` uses.
ARG UBUNTU_SNAPSHOT=20260925T120000Z
ARG DOCKER_CLI_VERSION=5:29.8.1-1~ubuntu.24.04~noble
ARG DOCKER_COMPOSE_VERSION=5.5.1-1~ubuntu.24.04~noble
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i -E "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?\$#URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" /etc/apt/sources.list.d/ubuntu.sources \
    && test "$(grep -c "snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" /etc/apt/sources.list.d/ubuntu.sources)" = 2 \
    && apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        ca-certificates curl git gnupg python3 python3-venv \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        "docker-ce-cli=${DOCKER_CLI_VERSION}" "docker-compose-plugin=${DOCKER_COMPOSE_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version && docker compose version \
    && mkdir -p /opt/gbench \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt \
    && grep -qx "docker-ce-cli=${DOCKER_CLI_VERSION}" /opt/gbench/dpkg-versions.txt \
    && grep -qx "docker-compose-plugin=${DOCKER_COMPOSE_VERSION}" /opt/gbench/dpkg-versions.txt

# uv (astral) for `uv sync` / `uv run bench`, at a fixed release.
ARG UV_VERSION=0.12.19
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh \
    && /usr/local/bin/uv --version | grep -q "^uv ${UV_VERSION} "
ENV PATH="/usr/local/bin:${PATH}"

# Clone SkillsBench at a pinned commit (bakes tasks/ + the open-model runner) and uv-sync it so
# `uv run bench` resolves the BenchFlow harness.
ARG SKILLSBENCH_REF=9a1f4dd5f7659f75707435da3ce854b6e48321d1
RUN git clone https://github.com/benchflow-ai/skillsbench /app/skillsbench \
    && cd /app/skillsbench && git checkout "${SKILLSBENCH_REF}"
WORKDIR /app/skillsbench
# Exactly the checkout's uv.lock: the build fails if the lock does not match pyproject.toml. The
# environment is built on the system python3 (no managed CPython is downloaded; the build fails
# if uv picks another interpreter), and its resolved set is recorded.
RUN uv sync --locked \
    && .venv/bin/python -c "import sys; assert sys.base_prefix == '/usr', sys.base_prefix" \
    && uv pip freeze --python .venv/bin/python > /opt/gbench/pip-freeze.txt
LABEL org.gbench.docker_cli="${DOCKER_CLI_VERSION}" \
      org.gbench.docker_compose="${DOCKER_COMPOSE_VERSION}" \
      org.gbench.ubuntu_snapshot="${UBUNTU_SNAPSHOT}"

# gbench's fix to BenchFlow's deepagents agent (skillsbench_benchflow_transport.patch), applied to
# the environment's site-packages with each file's sha256 checked before and after: the shim gives
# ChatOpenAI the request timeout gbench passes (GBENCH_MODEL_REQUEST_TIMEOUT_S), and the agent
# install pins deepagents and langchain-openai (with --exclude-newer for their dependencies)
# instead of installing whatever PyPI serves at run time. The labels record the pins and the
# patched files' sha256.
ARG BENCHFLOW_SHIM_UPSTREAM_SHA256=0e44ea10f14d6441c68bfd63aff4c8396207f9234318316a682b11f47a87f8ca
ARG BENCHFLOW_SHIM_SHA256=7f04c4606e1ae9718d8bcb518af0d6a28294284407d8f3928263ae683afc8b95
ARG BENCHFLOW_REGISTRY_UPSTREAM_SHA256=7216f4a5a6379f7ed9b087f50f5a2ef8495e6589966ccc68be850020dc983c35
ARG BENCHFLOW_REGISTRY_SHA256=dd5316f7d45acc14b85cf1b8644afdbc6229aedcd1305a16e6c373f861dd2743
ARG DEEPAGENTS_VERSION=0.7.23
ARG LANGCHAIN_OPENAI_VERSION=1.7.0
COPY skillsbench_benchflow_transport.patch /opt/gbench/skillsbench-benchflow-transport.patch
RUN cd /app/skillsbench/.venv/lib/python3.12/site-packages \
    && echo "${BENCHFLOW_SHIM_UPSTREAM_SHA256}  benchflow/agents/deepagents_acp_shim.py" | sha256sum -c - \
    && echo "${BENCHFLOW_REGISTRY_UPSTREAM_SHA256}  benchflow/agents/registry.py" | sha256sum -c - \
    && git -C /app/skillsbench apply --directory=.venv/lib/python3.12/site-packages /opt/gbench/skillsbench-benchflow-transport.patch \
    && echo "${BENCHFLOW_SHIM_SHA256}  benchflow/agents/deepagents_acp_shim.py" | sha256sum -c - \
    && echo "${BENCHFLOW_REGISTRY_SHA256}  benchflow/agents/registry.py" | sha256sum -c - \
    && grep -q "deepagents==${DEEPAGENTS_VERSION} langchain-openai==${LANGCHAIN_OPENAI_VERSION}" benchflow/agents/registry.py \
    && /app/skillsbench/.venv/bin/python -c "import benchflow.agents.registry as r; c = r.AGENTS['deepagents'].install_cmd; assert 'deepagents==${DEEPAGENTS_VERSION}' in c and 'GBENCH_MODEL_REQUEST_TIMEOUT_S' in r._DEEPAGENTS_SHIM, c"
LABEL org.gbench.deepagents="${DEEPAGENTS_VERSION}" \
      org.gbench.langchain_openai="${LANGCHAIN_OPENAI_VERSION}" \
      org.gbench.benchflow_shim_sha256="${BENCHFLOW_SHIM_SHA256}" \
      org.gbench.benchflow_registry_sha256="${BENCHFLOW_REGISTRY_SHA256}"

# gbench's fix to BenchFlow's docker sandbox (skillsbench_benchflow_build_retry.patch), applied the
# same way with the file's sha256 checked before and after: BenchFlow re-runs a task environment's
# `docker compose build` after the transient failures it lists (twice, 2 s and 5 s later); the
# patch adds the Docker daemon's containerd layer-commit failure ("failed to export layer ...
# failed to commit: rename .../ingest/... .../blobs/...: no such file or directory") to that list.
# On 2026-10-13 it failed lab-unit-harmonization's build before the agent started, and the task
# wrote no reward. The build runs before the agent starts; any other build failure is unchanged.
ARG BENCHFLOW_DOCKER_SANDBOX_UPSTREAM_SHA256=426645c0be56d0f441869ef752b41b6ba0e15ece1e54dac359110acc8d4fe5f2
ARG BENCHFLOW_DOCKER_SANDBOX_SHA256=1265bca48b418492de9d94f9e3dfb1783da3024302d5c67b1b3dcbc1204e6a0f
COPY skillsbench_benchflow_build_retry.patch /opt/gbench/skillsbench-benchflow-build-retry.patch
RUN cd /app/skillsbench/.venv/lib/python3.12/site-packages \
    && echo "${BENCHFLOW_DOCKER_SANDBOX_UPSTREAM_SHA256}  benchflow/sandbox/docker.py" | sha256sum -c - \
    && git -C /app/skillsbench apply --directory=.venv/lib/python3.12/site-packages /opt/gbench/skillsbench-benchflow-build-retry.patch \
    && echo "${BENCHFLOW_DOCKER_SANDBOX_SHA256}  benchflow/sandbox/docker.py" | sha256sum -c - \
    && /app/skillsbench/.venv/bin/python -c "from benchflow.sandbox.docker import _is_retryable_docker_build_error as r; assert r('failed to export layer: CreateDiff: x: failed to commit: rename /var/lib/containerd/io.containerd.content.v1.content/ingest/1/data /var/lib/containerd/io.containerd.content.v1.content/blobs/sha256/2: no such file or directory') and not r('returned a non-zero code: 1')"
LABEL org.gbench.benchflow_docker_sandbox_sha256="${BENCHFLOW_DOCKER_SANDBOX_SHA256}"

# gbench adapter: the launcher (delegates to the upstream runner with a gbench model target +
# parses rewards) and the entrypoint.
RUN mkdir -p /app/adapter
COPY skillsbench_run.py /app/adapter/skillsbench_run.py
COPY skillsbench_entrypoint.sh /app/adapter/skillsbench_entrypoint.sh
RUN chmod +x /app/adapter/skillsbench_entrypoint.sh

# The adapter version: the launcher applies GBENCH_SHARD (a malformed value stops it), draws an
# --eval-limit subset with a seeded draw balanced across the task categories (reported in the
# summary's selection), records the task lists it ran (n_tasks_full, tasks_selected) and runs the
# with-skills arm with `--skill-mode with-skill`; since 5 it deals --eval-limit over the conditions
# of a two-condition run (SKILLSBENCH_CONDITION_LIMITS, nested draws); since 6 it passes
# SKILLSBENCH_AGENT_IDLE_TIMEOUT_S as --agent-idle-timeout and GBENCH_MODEL_REQUEST_TIMEOUT_S to
# the agent (with the BenchFlow patch above); since 7 it records the environment builds BenchFlow
# re-ran (sandbox_build_retries; with the build-retry patch above). The gbench runner refuses an
# image without this label at this value (skillsbench.HARNESS_VERSION), so a stale image is
# rebuilt instead of silently ignoring a shard or taking the head of the task list.
LABEL org.gbench.skillsbench-harness="7"

ENV SKILLSBENCH_DIR=/app/skillsbench
ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/skillsbench_entrypoint.sh"]
