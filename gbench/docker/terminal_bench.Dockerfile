# Terminal-Bench 2.1 harness image for gbench's `terminal_bench` suite, built LOCALLY.
#
# It holds Harbor 0.20.0, the release of the TB2.1 leaderboard's lock
# (harbor-framework/terminal-bench-2-1@7131e43 leaderboard/uv.lock), with every package of Harbor's
# dependency closure at the version that lock resolves (terminal_bench.constraints.txt; the build
# fails if anything else is installed), and the Docker CLI with its compose and buildx plugins
# (clients only). gbench runs `harbor run` in it with Harbor's local docker backend: the task
# containers run on the HOST daemon through the mounted docker socket (docker-out-of-docker), and
# the jobs directory is mounted at the same path on both sides because Harbor bind-mounts each
# trial's directories into its task container. The container runs as the invoking user, on the host
# network, with an environment of named variables only (no credential), the docker socket, the jobs
# directory, gbench's Harbor task cache and the salvage patch mounted, and Harbor's telemetry off
# (HARBOR_TELEMETRY=0); see docs/evals/terminal_bench.md.
#
# Build (context = gbench/docker; the tag is the one gbench runs):
#   docker build -t gbench-terminal-bench:harbor-0.20.0 -f gbench/docker/terminal_bench.Dockerfile gbench/docker
#
# Base: python:3.12-slim at a fixed digest (Harbor 0.20.0 requires Python 3.12 or later).
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The Docker CLI (from Docker's static release) and the compose and buildx plugins (GitHub
# releases), each checked against its sha256: Harbor's docker backend drives `docker compose` and
# builds with `docker buildx`. The compose and buildx sums are the ones their releases publish.
ARG DOCKER_VERSION=29.1.3
ARG DOCKER_SHA256=c019c608ba2bb009dd673f3230e4d743f36a78d36166c6c2444c05d0aa9ff0d9
ARG COMPOSE_VERSION=2.40.3
ARG COMPOSE_SHA256=dba9d98e1ba5bfe11d88c99b9bd32fc4a0624a30fafe68eea34d61a3e42fd372
ARG BUILDX_VERSION=0.37.1
ARG BUILDX_SHA256=9447199cdb435f25880548343c128a4b6650e8891ee598905d8d29d39a8e359b
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL -o /tmp/docker.tgz \
        "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" \
    && echo "${DOCKER_SHA256}  /tmp/docker.tgz" | sha256sum -c - \
    && tar -xzf /tmp/docker.tgz -C /usr/local/bin --strip-components=1 docker/docker \
    && rm -f /tmp/docker.tgz \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
        "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    && echo "${COMPOSE_SHA256}  /usr/local/lib/docker/cli-plugins/docker-compose" | sha256sum -c - \
    && curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-buildx \
        "https://github.com/docker/buildx/releases/download/v${BUILDX_VERSION}/buildx-v${BUILDX_VERSION}.linux-amd64" \
    && echo "${BUILDX_SHA256}  /usr/local/lib/docker/cli-plugins/docker-buildx" | sha256sum -c - \
    && chmod 0755 /usr/local/lib/docker/cli-plugins/docker-compose \
        /usr/local/lib/docker/cli-plugins/docker-buildx \
    && docker --version && docker compose version && docker buildx version \
    && rm -rf /var/lib/apt/lists/*

# Harbor at the leaderboard's release, every dependency at the lock's version (wheels only), then
# the check that the installed set is exactly the pinned one; the resolved set is recorded.
ARG HARBOR_VERSION=0.20.0
ARG TB21_LOCK_COMMIT=7131e4375048a0e408a8fb404b5f499d726b695b
COPY terminal_bench.constraints.txt /opt/harbor/constraints.txt
COPY terminal_bench_verify_env.py /opt/harbor/verify_env.py
RUN python -m pip install --only-binary=:all: -c /opt/harbor/constraints.txt \
        "harbor==${HARBOR_VERSION}" \
    && python -m pip check \
    && python /opt/harbor/verify_env.py /opt/harbor/constraints.txt "${HARBOR_VERSION}" \
        /opt/harbor/pip-freeze.txt \
    && harbor --version

# gbench runs the container as the invoking user (any uid): HOME is a world-writable directory of
# the image, and gbench mounts its Harbor task cache at $HOME/.cache/harbor and the salvage patch
# under /opt/gbench.
RUN mkdir -p /harbor-home/.cache /harbor-home/.config /opt/gbench \
    && chmod 1777 /harbor-home /harbor-home/.cache /harbor-home/.config
# Harbor's usage telemetry (PostHog events from harbor/telemetry.py) is off in this image, as it is
# in every container gbench starts from it (gbench also passes HARBOR_TELEMETRY=0 itself). It
# changes nothing Harbor runs or scores. The check fails the build if this Harbor ignores it.
ENV HOME=/harbor-home \
    HARBOR_TELEMETRY=0
RUN python -c "import harbor.telemetry as t; assert t._telemetry_disabled(), 'HARBOR_TELEMETRY=0 not honoured'"
WORKDIR /harbor-home

LABEL org.gbench.harbor="${HARBOR_VERSION}" \
      org.gbench.terminal_bench_lock="${TB21_LOCK_COMMIT}" \
      org.gbench.docker_cli="${DOCKER_VERSION}" \
      org.gbench.docker_compose="${COMPOSE_VERSION}" \
      org.gbench.docker_buildx="${BUILDX_VERSION}"

# No ENTRYPOINT: gbench runs `docker run ... <image> python -c <salvage bootstrap> run ...` (or
# `harbor run ...` with GBENCH_TB_SALVAGE_PATCH=0).
