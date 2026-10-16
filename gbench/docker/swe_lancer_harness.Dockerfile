# SWE-Lancer harness image for gbench's `swe_lancer` suite, built LOCALLY (gbench never pulls it).
#
# It holds openai/frontier-evals `project/swelancer` (SWE-Lancer's current home; the old
# openai/SWELancer-Benchmark repository is archived and points there) at a pinned commit, with the
# dependency set of its own uv.lock (`uv sync --frozen`). gbench runs SWE-Lancer's own solver,
# SimpleAgentSolver, through `swelancer/run_swelancer.py`'s main (see swe_lancer_eval.py), on the
# official task set, with the task images SWE-Lancer publishes (swelancer/swelancer_x86_<task>:
# releasev1 for IC SWE, swelancer/swelancer_x86_monolith:releasev1 for SWE Manager), pulled by the
# harness's alcatraz LocalConfig(pull_from_registry=True) exactly as upstream's README runs it.
#
# Build (context = gbench/docker):
#   docker build -t gbench-swe-lancer-harness:51052ced -f gbench/docker/swe_lancer_harness.Dockerfile gbench/docker
#
# gbench runs it docker-out-of-docker: host network, the docker socket, NET_ADMIN (the harness's
# disable_internet=True adds iptables rules to the host's DOCKER-USER chain for each task
# container and removes them after), and the run directory identity-mounted.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# python3 (3.12) for the harness, git to fetch the pinned tree, iptables + sudo for alcatraz's
# network block (it runs `iptables ...` and `sudo iptables ...`). Installed versions are recorded in
# /opt/swe-lancer-dpkg-versions.txt.
# The Ubuntu archive as of UBUNTU_SNAPSHOT (snapshot.ubuntu.com, apt >= 2.7), so a rebuild installs the
# same package versions instead of whatever the archive serves that day. The snapshot service is
# https-only, so ca-certificates is installed first from the base image's archive and then again at
# the snapshot's version (--allow-downgrades).
ARG UBUNTU_SNAPSHOT=20260925T120000Z
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i -E "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?\$#URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" /etc/apt/sources.list.d/ubuntu.sources \
    && test "$(grep -c "snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" /etc/apt/sources.list.d/ubuntu.sources)" = 2 \
    && apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        ca-certificates git iptables python3 python3-pip python3-venv sudo \
    && rm -rf /var/lib/apt/lists/* \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/swe-lancer-dpkg-versions.txt

ARG UV_VERSION=0.12.12
RUN python3 -m pip install --no-cache-dir --break-system-packages "uv==${UV_VERSION}"

# SWE-Lancer at a pinned commit of openai/frontier-evals, and the task table the suite pins.
ARG FRONTIER_EVALS_REF=51052cede8cc608f95bb00346635e03759013e5a
ARG SWELANCER_TASKS_SHA256=5c3a6d4570b49be0d9fced98f5b32487420b16f25c98d6658830e31fa03f049a
RUN git init -q /opt/frontier-evals \
    && git -C /opt/frontier-evals fetch -q --depth 1 https://github.com/openai/frontier-evals "${FRONTIER_EVALS_REF}" \
    && git -C /opt/frontier-evals checkout -q FETCH_HEAD \
    && test "$(git -C /opt/frontier-evals rev-parse HEAD)" = "${FRONTIER_EVALS_REF}" \
    && rm -rf /opt/frontier-evals/.git \
    && echo "${SWELANCER_TASKS_SHA256}  /opt/frontier-evals/project/swelancer/all_swelancer_tasks.csv" | sha256sum -c -

# The project's own lockfile, as upstream's README installs it (`uv sync`), without the dev group
# (mypy, ruff, pytest, playwright for the repo's own tests). The resolved set is recorded in
# /opt/swe-lancer-pip-freeze.txt.
WORKDIR /opt/frontier-evals/project/swelancer
ENV UV_PYTHON=/usr/bin/python3 UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_NO_CACHE=1
RUN uv sync --frozen --no-dev \
    && uv pip freeze --python .venv/bin/python > /opt/swe-lancer-pip-freeze.txt

# SimpleAgentSolver counts tokens with tiktoken (gpt-4's encoding for a model it does not know);
# the encoding file is fetched once here instead of at every run.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache
RUN .venv/bin/python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4').encode('x')"

COPY swe_lancer_eval.py /opt/gbench/swe_lancer_eval.py
COPY swe_lancer_entrypoint.sh /opt/gbench/swe_lancer_entrypoint.sh
ENV PYTHONPATH=/opt/gbench
RUN chmod 0755 /opt/gbench/swe_lancer_entrypoint.sh \
    && cd /tmp && /opt/frontier-evals/project/swelancer/.venv/bin/python -c \
       "import swe_lancer_eval, swelancer.solvers.swelancer_agent.solver, nanoeval_alcatraz.alcatraz_computer_interface, alcatraz.clusters.local"

LABEL org.gbench.frontier_evals="${FRONTIER_EVALS_REF}" \
      org.gbench.swelancer_tasks_sha256="${SWELANCER_TASKS_SHA256}" \
      org.gbench.uv="${UV_VERSION}"

ENTRYPOINT ["/opt/gbench/swe_lancer_entrypoint.sh"]
