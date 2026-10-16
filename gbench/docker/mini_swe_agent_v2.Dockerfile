# mini-swe-agent 2.x rollout image for gbench's SWE-bench suites, built LOCALLY.
#
# The SWE-bench Verified and Multilingual leaderboards' current mini-SWE-agent rows run
# mini-swe-agent 2.x with its bundled `benchmarks/swebench.yaml` (tool-calling actions). The main
# gbench environment pins mini-swe-agent 1.x for the in-process runner, and the two majors cannot
# share one environment, so the 2.x releases live HERE, one virtualenv per release:
#   /opt/mini-swe-agent-<version>/bin/python
# gbench (swe_agentic) starts one container of this image per instance and runs
# /opt/gbench/mini_swe_agent_v2_run.py with the release the suite pins. The agent starts the
# instance's SWE-bench image on the HOST daemon through the mounted docker socket
# (docker-out-of-docker). The container runs on docker's default bridge network with the
# placeholder key OPENAI_API_KEY=EMPTY and talks to the served model through gbench's relay on
# that network's gateway, which adds the endpoint's key (no credential enters the container).
#
# Build (context = gbench/docker):
#   docker build -t gbench-mini-swe-agent-v2:2.4.2-2.4.6 -f gbench/docker/mini_swe_agent_v2.Dockerfile gbench/docker
#
# Which suite uses which release (see docs/evals/copilot_bench_swe.md and
# docs/evals/swe_bench_multilingual.md):
#   copilot_bench_swe       mini-swe-agent 2.4.2
#   swe_bench_multilingual  mini-swe-agent 2.4.6
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MSWEA_SILENT_STARTUP=1

# python3 + venv, and the Docker CLI (client only - it talks to the mounted host daemon) at a
# pinned version, from the Ubuntu archive as of UBUNTU_SNAPSHOT (snapshot.ubuntu.com, apt >= 2.7),
# so a rebuild installs the same package versions instead of whatever the archive serves that
# day. The snapshot service is https-only, so ca-certificates is installed first from the base
# image's archive and then again at the snapshot's version (--allow-downgrades). The installed
# package versions are recorded in /opt/gbench/dpkg-versions.txt.
ARG UBUNTU_SNAPSHOT=20260925T120000Z
ARG DOCKER_CLI_VERSION=5:29.8.1-1~ubuntu.24.04~noble
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i -E "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?\$#URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" /etc/apt/sources.list.d/ubuntu.sources \
    && test "$(grep -c "snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" /etc/apt/sources.list.d/ubuntu.sources)" = 2 \
    && apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        ca-certificates curl gnupg python3 python3-venv \
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

# One virtualenv per pinned release, installed against mini_swe_agent_v2.constraints.txt (the
# resolved set of the image whose runs were checked; both releases resolve to it). The build fails
# unless each virtualenv holds exactly that set plus its release; each records its set.
ARG MINI_SWE_AGENT_VERSIONS="2.4.2 2.4.6"
COPY mini_swe_agent_v2.constraints.txt /opt/gbench/mini_swe_agent_v2.constraints.txt
RUN set -e; grep -v -e '^#' -e '^$' /opt/gbench/mini_swe_agent_v2.constraints.txt | LC_ALL=C sort \
        > /tmp/pins.txt; \
    for v in ${MINI_SWE_AGENT_VERSIONS}; do \
        python3 -m venv "/opt/mini-swe-agent-$v"; \
        PIP_CONSTRAINT=/opt/gbench/mini_swe_agent_v2.constraints.txt \
            "/opt/mini-swe-agent-$v/bin/pip" install --no-cache-dir "mini-swe-agent==$v"; \
        "/opt/mini-swe-agent-$v/bin/python" -c "import minisweagent, sys; sys.exit(minisweagent.__version__ != '$v')"; \
        "/opt/mini-swe-agent-$v/bin/pip" freeze --all | LC_ALL=C sort > "/opt/mini-swe-agent-$v/pip-freeze.txt"; \
        grep -qx "mini-swe-agent==$v" "/opt/mini-swe-agent-$v/pip-freeze.txt"; \
        grep -vx "mini-swe-agent==$v" "/opt/mini-swe-agent-$v/pip-freeze.txt" | diff /tmp/pins.txt -; \
    done; \
    rm -f /tmp/pins.txt

COPY mini_swe_agent_v2_run.py /opt/gbench/mini_swe_agent_v2_run.py

LABEL org.gbench.mini_swe_agent_versions="${MINI_SWE_AGENT_VERSIONS}" \
      org.gbench.docker_cli="${DOCKER_CLI_VERSION}" \
      org.gbench.ubuntu_snapshot="${UBUNTU_SNAPSHOT}"

# No ENTRYPOINT: gbench runs
#   docker run <image> /opt/mini-swe-agent-<version>/bin/python /opt/gbench/mini_swe_agent_v2_run.py --spec <spec>
# with the output directory identity-mounted and the docker socket mounted.
