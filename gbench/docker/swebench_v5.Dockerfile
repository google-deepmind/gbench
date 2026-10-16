# SWE-bench v5 scoring image for gbench's copilot_bench_swe and swe_bench_multilingual, built
# LOCALLY.
#
# The current SWE-bench Verified and Multilingual datasets carry the per-instance `image`,
# `eval_script`, `log_parser` and `eval_type` columns of the v5 harness (swebench 5.x), which is
# how the SWE-bench maintainers grade their current submissions. The main gbench environment
# keeps swebench 4.1 (upgrading it in place is not an additive change), so the v5 harness lives
# HERE. gbench runs `python -m swebench.harness.run_evaluation` inside this image on a pinned
# local snapshot of the dataset; the harness starts the per-instance `swebench/sweb.eval.*`
# containers on the HOST daemon through the mounted docker socket (docker-out-of-docker).
#
# Build (context = gbench/docker):
#   docker build -t gbench-swebench-v5:5.0.2 -f gbench/docker/swebench_v5.Dockerfile gbench/docker
# ubuntu:24.04 by digest (the index digest of the build the v5 harness was validated on).
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# python3 + pip, git, and the Docker CLI (client only - it talks to the mounted host daemon) at a
# pinned version. swebench 5.x stops and removes each task container with `docker stop` /
# `docker rm` (harness/docker_utils.cleanup_container); without the CLI every task container is
# left running. The installed package versions are recorded in /opt/swebench-dpkg-versions.txt.
# The Ubuntu archive as of UBUNTU_SNAPSHOT (snapshot.ubuntu.com, apt >= 2.7), so a rebuild installs the
# same package versions instead of whatever the archive serves that day. The snapshot service is
# https-only, so ca-certificates is installed first from the base image's archive and then again at
# the snapshot's version (--allow-downgrades).
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
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/swebench-dpkg-versions.txt \
    && grep -qx "docker-ce-cli=${DOCKER_CLI_VERSION}" /opt/swebench-dpkg-versions.txt

# --break-system-packages: ubuntu 24.04's python is externally managed and this image is
# single-purpose. swebench's dependencies are pinned by swebench_v5.constraints.txt (the resolved
# set the harness was validated with); the resolved set of the build is recorded next to it.
ARG SWEBENCH_VERSION=5.0.2
COPY swebench_v5.constraints.txt /opt/swebench-constraints.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        -c /opt/swebench-constraints.txt "swebench==${SWEBENCH_VERSION}" \
    && python3 -c "import importlib.metadata as m, sys; sys.exit(m.version('swebench') != '${SWEBENCH_VERSION}')" \
    && python3 -m pip freeze > /opt/swebench-pip-freeze.txt \
    && ln -sf "$(command -v python3)" /usr/local/bin/python

LABEL org.gbench.swebench="${SWEBENCH_VERSION}" org.gbench.docker_cli="${DOCKER_CLI_VERSION}"

# No ENTRYPOINT: gbench's scorer invokes
#   docker run <image> python -m swebench.harness.run_evaluation ...
# with the predictions, the working directory and the dataset snapshot identity-mounted (and,
# only when that run's final report step crashed, the same arguments through
# `python -c <swebench_common._REPORT_STEP>`, which re-runs make_run_report alone).
