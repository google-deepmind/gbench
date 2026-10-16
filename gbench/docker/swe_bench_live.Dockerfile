# Canonical SWE-bench-Live scoring image for gbench's `swe_bench_live` suite, built LOCALLY. It carries
# both harnesses the leaderboard's tabs are scored with: the SWE-bench-Live fork of `swebench` (Lite,
# Full, Verified) and upstream's multi-language harness (the C/C++, C#, Java, TS/JS, Go and Rust tabs).
#
# SWE-bench-Live ships the `swebench` harness as a FORK: same package name, a version incompatible
# with the upstream `swebench` the main gbench env uses (swe_bench_multilingual / copilot_bench_swe).
# Only one `swebench` can be installed per environment, so gbench isolates the fork HERE instead of a
# separate host virtualenv. gbench runs `swebench.harness.run_evaluation` inside this image; the
# harness spawns the per-instance task containers (DockerHub namespace `starryzhang`) on the HOST
# daemon via the mounted docker socket (docker-out-of-docker). The model rollout (patch generation)
# stays in the gbench process against the served /v1 endpoint and needs only `datasets`, so nothing
# fork-versioned leaks into the main env.
#
# Build (context = gbench/docker):
#   docker build -t gbench-swe-bench-live:a145aa87 -f gbench/docker/swe_bench_live.Dockerfile gbench/docker
#
# A graded run needs: a reachable Docker daemon (socket mounted), network egress (the per-instance
# `starryzhang/*` task images are pulled from DockerHub), and a served model at /v1. See
# docs/evals/swe_bench_live.md.
# ubuntu:24.04 by digest (the index digest of the build the fork harness was validated on).
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Base: python + git, and the Docker CLI (client only - it talks to the mounted host daemon) at a
# pinned version. The installed package versions are recorded in /opt/swe-bench-live-dpkg-versions.txt.
# The Ubuntu archive as of UBUNTU_SNAPSHOT (snapshot.ubuntu.com, apt >= 2.7), so a rebuild installs the
# same package versions instead of whatever the archive serves that day. The snapshot service is
# https-only, so ca-certificates is installed first from the base image's archive and then again at
# the snapshot's version (--allow-downgrades).
ARG UBUNTU_SNAPSHOT=20260925T120000Z
ARG DOCKER_CLI_VERSION=5:29.8.0-1~ubuntu.24.04~noble
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
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/swe-bench-live-dpkg-versions.txt \
    && grep -qx "docker-ce-cli=${DOCKER_CLI_VERSION}" /opt/swe-bench-live-dpkg-versions.txt

# The SWE-bench-Live fork of `swebench` at a pinned commit, plus `datasets` (the harness loads the
# dataset inside the container), with every dependency pinned by swe_bench_live.constraints.txt
# (the resolved set the fork harness was validated with); the resolved set of the build is
# recorded next to it. --break-system-packages: ubuntu 24.04's python is externally managed and
# this image is single-purpose.
ARG SWE_BENCH_LIVE_REF=a145aa87c62361532dacaa243398978164b234b7
COPY swe_bench_live.constraints.txt /opt/swe-bench-live-constraints.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        -c /opt/swe-bench-live-constraints.txt \
        "git+https://github.com/SWE-bench-Live/SWE-bench-Live@${SWE_BENCH_LIVE_REF}" datasets \
    && python3 -m pip freeze > /opt/swe-bench-live-pip-freeze.txt \
    && grep -qx "swebench @ git+https://github.com/SWE-bench-Live/SWE-bench-Live@${SWE_BENCH_LIVE_REF}" /opt/swe-bench-live-pip-freeze.txt \
    && ln -sf "$(command -v python3)" /usr/local/bin/python

# Upstream's multi-language harness (evaluation/evaluation.py, the one microsoft/SWE-bench-Live's
# README gives for SWE-bench-Live/MultiLang and SWE-bench-Live/Windows) with the RepoLaunch
# submodule it imports, both at pinned commits; the build checks that the submodule commit the
# harness commit points at is the RepoLaunch commit it fetches. The tree is not pip-installed: its
# pyproject names the package `swebench`, which would replace the fork above. gbench runs it with
# PYTHONPATH=/opt/swe-bench-live-main/launch:/opt/swe-bench-live-main; everything it imports
# (docker, datasets, typing_extensions) is already installed and pinned above.
ARG SWE_BENCH_LIVE_MAIN_REF=62d02936a0726fbf4597b05bd07b09fb99eddae6
ARG REPOLAUNCH_REF=c4b623d930f3728e5338664bb634021b98492cbf
RUN git init -q /opt/swe-bench-live-main \
    && git -C /opt/swe-bench-live-main fetch -q --depth 1 https://github.com/microsoft/SWE-bench-Live "${SWE_BENCH_LIVE_MAIN_REF}" \
    && git -C /opt/swe-bench-live-main checkout -q FETCH_HEAD \
    && test "$(git -C /opt/swe-bench-live-main rev-parse HEAD)" = "${SWE_BENCH_LIVE_MAIN_REF}" \
    && test "$(git -C /opt/swe-bench-live-main ls-tree HEAD launch | awk '{print $3}')" = "${REPOLAUNCH_REF}" \
    && rm -rf /opt/swe-bench-live-main/launch && git init -q /opt/swe-bench-live-main/launch \
    && git -C /opt/swe-bench-live-main/launch fetch -q --depth 1 https://github.com/microsoft/RepoLaunch "${REPOLAUNCH_REF}" \
    && git -C /opt/swe-bench-live-main/launch checkout -q FETCH_HEAD \
    && test "$(git -C /opt/swe-bench-live-main/launch rev-parse HEAD)" = "${REPOLAUNCH_REF}" \
    && rm -rf /opt/swe-bench-live-main/.git /opt/swe-bench-live-main/launch/.git \
    && cd /tmp && PYTHONPATH=/opt/swe-bench-live-main/launch:/opt/swe-bench-live-main \
       python3 -c "import evaluation.evaluation, launch.core.runtime, launch.scripts.parser"

LABEL org.gbench.swe_bench_live_fork="${SWE_BENCH_LIVE_REF}" org.gbench.docker_cli="${DOCKER_CLI_VERSION}" \
      org.gbench.swe_bench_live_main="${SWE_BENCH_LIVE_MAIN_REF}" org.gbench.repolaunch="${REPOLAUNCH_REF}"

# No ENTRYPOINT: gbench's scorers invoke `docker run <image> python -m swebench.harness.run_evaluation ...`
# (the Python tabs) or `docker run -e PYTHONPATH=... <image> python -m evaluation.evaluation ...` (the
# MultiLang tabs), with the predictions + workdir identity-mounted and the docker socket mounted for the
# task containers.
