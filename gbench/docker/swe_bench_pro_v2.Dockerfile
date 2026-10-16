# SWE-bench Pro V2 harness image for gbench's `swe_bench_pro` suite, built LOCALLY.
#
# SWE-bench Pro V2 ships as Harbor task directories plus locked-protocol tooling in
# scaleapi/SWE-bench_Pro-os (`v2/`), run with Harbor. This image holds Harbor 0.22.0 (the release
# gate's version, v2/GATE.md) and that tree at a pinned commit, checked against the tree's own
# v2/SHA256SUMS. gbench runs two `harbor run` jobs in it (the locked_mini_swe agent job, then the
# patch_replay re-grade) with Harbor's local docker backend: the task sandboxes, the egress-control
# sidecar Harbor uses for the offline agent phase, and its probe run on the HOST daemon through the
# mounted docker socket (docker-out-of-docker). The work directory is mounted at the same path on
# both sides because Harbor bind-mounts trial directories into the sandboxes.
#
# Build (context = gbench/docker):
#   docker build -t gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt -f gbench/docker/swe_bench_pro_v2.Dockerfile gbench/docker
#
# A graded run needs: a reachable Docker daemon whose kernel supports nftables fib rules (Harbor's
# egress control), network egress in the setup and verifier phases (instance images from ghcr.io,
# the in-sandbox mini-swe-agent install, a few Go tasks fetch modules), and the served model
# reachable from the sandboxes. See docs/evals/swe_bench_pro.md.
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# python3 + pip, and the Docker CLI with the compose and buildx plugins (Harbor's docker backend
# drives `docker compose` and builds its egress sidecar with buildx) - clients only - at pinned
# versions, from the Ubuntu archive as of UBUNTU_SNAPSHOT (snapshot.ubuntu.com, apt >= 2.7), so a
# rebuild installs the same package versions instead of whatever the archive serves that day. The
# snapshot service is https-only, so ca-certificates is installed first from the base image's
# archive and then again at the snapshot's version (--allow-downgrades). The installed package
# versions are recorded in /opt/gbench/dpkg-versions.txt.
ARG UBUNTU_SNAPSHOT=20260925T120000Z
ARG DOCKER_CLI_VERSION=5:29.8.1-1~ubuntu.24.04~noble
ARG DOCKER_COMPOSE_VERSION=5.5.1-1~ubuntu.24.04~noble
ARG DOCKER_BUILDX_VERSION=0.37.1-1~ubuntu.24.04~noble
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i -E "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?\$#URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" /etc/apt/sources.list.d/ubuntu.sources \
    && test "$(grep -c "snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" /etc/apt/sources.list.d/ubuntu.sources)" = 2 \
    && apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        ca-certificates curl gnupg python3 python3-pip \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        "docker-ce-cli=${DOCKER_CLI_VERSION}" "docker-compose-plugin=${DOCKER_COMPOSE_VERSION}" \
        "docker-buildx-plugin=${DOCKER_BUILDX_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version && docker compose version && docker buildx version \
    && mkdir -p /opt/gbench \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt \
    && grep -qx "docker-ce-cli=${DOCKER_CLI_VERSION}" /opt/gbench/dpkg-versions.txt \
    && grep -qx "docker-compose-plugin=${DOCKER_COMPOSE_VERSION}" /opt/gbench/dpkg-versions.txt \
    && grep -qx "docker-buildx-plugin=${DOCKER_BUILDX_VERSION}" /opt/gbench/dpkg-versions.txt

# Harbor at the release gate's version, installed against swe_bench_pro_v2.constraints.txt (the
# resolved set of the image whose Harbor install was checked). --break-system-packages: ubuntu
# 24.04's python is externally managed and this image is single-purpose. The build fails unless
# the interpreter holds exactly that set; the set is recorded.
ARG HARBOR_VERSION=0.22.0
COPY swe_bench_pro_v2.constraints.txt /opt/gbench/swe_bench_pro_v2.constraints.txt
RUN PIP_CONSTRAINT=/opt/gbench/swe_bench_pro_v2.constraints.txt \
        python3 -m pip install --no-cache-dir --break-system-packages "harbor==${HARBOR_VERSION}" \
    && python3 -c "import importlib.metadata as m, sys; sys.exit(m.version('harbor') != '${HARBOR_VERSION}')" \
    && python3 -m pip freeze --all | LC_ALL=C sort > /opt/harbor-pip-freeze.txt \
    && grep -v -e '^#' -e '^$' /opt/gbench/swe_bench_pro_v2.constraints.txt | LC_ALL=C sort \
        | diff - /opt/harbor-pip-freeze.txt

# The V2 tree at the pinned commit (GitHub's tarball of that commit), verified against
# v2/SHA256SUMS: 642 task directories, HARD-51 ids, locked-protocol tooling.
ARG SWE_BENCH_PRO_OS_COMMIT=66f92766bba642462d4bbe5479e83f91f9211862
RUN mkdir -p /opt/SWE-bench_Pro-os \
    && curl -fsSL "https://codeload.github.com/scaleapi/SWE-bench_Pro-os/tar.gz/${SWE_BENCH_PRO_OS_COMMIT}" \
        | tar -xz -C /opt/SWE-bench_Pro-os --strip-components=1 \
            "SWE-bench_Pro-os-${SWE_BENCH_PRO_OS_COMMIT}/v2" \
    && cd /opt/SWE-bench_Pro-os/v2 && sha256sum --quiet -c SHA256SUMS \
    && test "$(ls tasks | wc -l)" = 642 && test "$(wc -l < hard51_ids.txt)" = 51 \
    && chmod -R a+rX /opt/SWE-bench_Pro-os

# Harbor's egress-control sidecar (harbor-docker-egress-control-sidecar, a gost transparent proxy
# with HTTP sniffing) carries the agent phase's model requests. gost waits at most `readTimeout`
# for the upstream's response headers, 15 s when unset (go-gost/x v0.10.9,
# handler/redirect/tcp/metadata.go; internal/util/sniffing/sniffer_http.go), and a non-streamed
# completion sends its headers only when the whole reply is ready. 2026-10-11 validation run: 128
# of 198 model requests were closed at <= 15 s (relay client_close_ages max_s 15.0) and every agent
# saw "Server disconnected without sending a response". gost has no setting that disables the
# wait (0 means 15 s), so the sidecar's gost.yaml gets a readTimeout far above any relay limit
# (base.request_timeout_s grows with the tokens a request may generate: 33,968 s for 262,144 tokens
# at 8 tok/s). Harbor builds the sidecar image from this directory at run time, tagged by a hash of
# its contents, so the patched file gives a new sidecar image. The original file is checked first.
ARG EGRESS_SIDECAR_READ_TIMEOUT=8760h
ARG EGRESS_SIDECAR_GOST_YAML_SHA256=78c18e58efbc7c8e226cb311e2a4cf3863d5a2f3e54b3bf15a0ea4b885345290
RUN f="$(python3 -c 'import harbor.environments.docker as d; print(d.EGRESS_CONTROL_SIDECAR_CONTEXT_PATH)')/gost.yaml" \
    && echo "${EGRESS_SIDECAR_GOST_YAML_SHA256}  $f" | sha256sum --quiet -c \
    && sed -i "s/^        sniffing.fallback: true\$/&\n        readTimeout: ${EGRESS_SIDECAR_READ_TIMEOUT}/" "$f" \
    && python3 -c "import sys, yaml; m = yaml.safe_load(open(sys.argv[1]))['services'][0]['handler']['metadata']; sys.exit(m.get('readTimeout') != sys.argv[2])" "$f" "${EGRESS_SIDECAR_READ_TIMEOUT}"

LABEL org.gbench.harbor="${HARBOR_VERSION}"
LABEL org.gbench.egress_sidecar_read_timeout="${EGRESS_SIDECAR_READ_TIMEOUT}"
LABEL org.gbench.swe_bench_pro_os="${SWE_BENCH_PRO_OS_COMMIT}"
LABEL org.gbench.docker_cli="${DOCKER_CLI_VERSION}" \
      org.gbench.docker_compose="${DOCKER_COMPOSE_VERSION}" \
      org.gbench.docker_buildx="${DOCKER_BUILDX_VERSION}" \
      org.gbench.ubuntu_snapshot="${UBUNTU_SNAPSHOT}"

WORKDIR /opt/SWE-bench_Pro-os
ENV PYTHONPATH=/opt/SWE-bench_Pro-os/v2/tooling

# No ENTRYPOINT: gbench runs `docker run <image> sh -c 'exec harbor run ...'` with the work
# directory identity-mounted and the docker socket mounted.
