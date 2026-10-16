# MSWE-agent rollout image for gbench's `multi_swe_bench` suite, built LOCALLY.
#
# MSWE-agent (multi-swe-bench/MSWE-agent) is the Multi-SWE-bench team's SWE-agent for the benchmark and
# the scaffold of the leaderboard's MSWE-agent rows (multi-swe-bench/experiments evaluation/<lang>/
# verified/*_MSWE-agent_*). This image carries it at a pinned commit, set up as its README says
# (Python 3.9 from environment.yml, `pip install -r requirements.txt`, run from the checkout), with
# every dependency pinned by mswe_agent.constraints.txt.
#
# Two additions, both recorded in labels and checked here:
#   * dataclasses-json: the vendored multi_swe_bench harness imports it and requirements.txt does not
#     list it (without it `import run` fails);
#   * the harness definitions of the six Java repositories the benchmark's Java set holds and the
#     vendored harness at this commit does not register (apache/dubbo, fasterxml/jackson-core,
#     fasterxml/jackson-databind, fasterxml/jackson-dataformat-xml, google/gson,
#     googlecontainertools/jib: the 78 SWE-bench-java instances), copied unchanged from
#     multi-swe-bench at MSWE_HARNESS_JAVA_REF (the commit that announced them) with one line added,
#     `from __future__ import annotations`, so their `Image | None` annotations import on Python 3.9.
#     Without them MSWE-agent cannot load the Java set at all (Instance.create raises for an
#     unregistered repository), while the leaderboard's MSWE-agent Java rows cover all 128 instances.
#
# gbench's driver (mswe_agent_run.py, label org.gbench.mswe_agent_driver=4, which gbench checks)
# adds, at run time, MSWE-agent's registration of the served model, the leaderboard runs' 50-step
# limit, the archive.debian.org fallback for the reset's `apt-get update` on instance images
# whose Debian release the Debian mirrors no longer serve (buster: the gcc:7-based C++ images), a
# record of the exception an instance's rollout raises, and a timeout for MSWE-agent's OpenAI
# client above gbench's relay's per-turn limit (instead of the SDK's 600 s, at which it abandons a
# long model turn and resends it); see its module docstring. The build checks that last one on a
# client built the way MSWE-agent builds it.
#
# gbench runs one container of this image per language:
#   /opt/MSWE-agent/venv/bin/python /opt/gbench/mswe_agent_run.py --spec <spec.json>
# with the run directory mounted at /mswe, the host network (the served model, reached through
# gbench's relay on 127.0.0.1) and the docker socket: MSWE-agent starts each instance image on the
# HOST daemon (docker-out-of-docker), as on the machine upstream ran it on.
#
# Build (context = gbench/docker):
#   docker build -t gbench-mswe-agent:88217624 -f gbench/docker/mswe_agent.Dockerfile gbench/docker
FROM python:3.9.21-slim-bookworm@sha256:40007fe18a72a2e7166be350d52dab86b9fe18f2de08e6a38e26422fb247e81e

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# git and the Docker CLI (client only - it talks to the mounted host daemon) at a pinned version, from
# the Debian archive as of DEBIAN_SNAPSHOT (snapshot.debian.org), so a rebuild installs the same
# package versions. The installed versions are recorded in /opt/mswe-agent-dpkg-versions.txt.
ARG DEBIAN_SNAPSHOT=20260925T120000Z
ARG DOCKER_CLI_VERSION=5:29.8.0-1~debian.12~bookworm
RUN printf 'Types: deb\nURIs: http://snapshot.debian.org/archive/debian/%s\nSuites: bookworm bookworm-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\nCheck-Valid-Until: no\n\nTypes: deb\nURIs: http://snapshot.debian.org/archive/debian-security/%s\nSuites: bookworm-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\nCheck-Valid-Until: no\n' \
        "${DEBIAN_SNAPSHOT}" "${DEBIAN_SNAPSHOT}" > /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends "docker-ce-cli=${DOCKER_CLI_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version && python3 --version | grep -qx "Python 3\.9\.21" \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/mswe-agent-dpkg-versions.txt \
    && grep -qx "docker-ce-cli=${DOCKER_CLI_VERSION}" /opt/mswe-agent-dpkg-versions.txt

# MSWE-agent at a pinned commit (the build checks the commit and the sha256 of config/default.yaml,
# the config gbench runs), its requirements in a venv next to it, and the six Java repository
# definitions described above. The build checks what imports without a Docker daemon (the vendored
# harness's docker_util opens a docker client when it is imported, so `run` and `multirun` are
# byte-compiled here and imported by gbench's driver at run time, with the socket mounted).
ARG MSWE_AGENT_REF=88217624b637646b886cb0462995c07559e96f58
ARG MSWE_AGENT_CONFIG_SHA256=913fc5fd89b3bdffe261b9cd2e38b2e45ad9612ea57bf314bcda9f7e314b4537
ARG MSWE_HARNESS_JAVA_REF=e0dcb3c4ce0ae27997fb213f2a5294e1eaa487b8
ARG MSWE_JAVA_ORGS="apache fasterxml google googlecontainertools"
COPY mswe_agent.constraints.txt /opt/mswe-agent-constraints.txt
RUN git init -q /opt/MSWE-agent \
    && git -C /opt/MSWE-agent fetch -q --depth 1 https://github.com/multi-swe-bench/MSWE-agent "${MSWE_AGENT_REF}" \
    && git -C /opt/MSWE-agent checkout -q FETCH_HEAD \
    && test "$(git -C /opt/MSWE-agent rev-parse HEAD)" = "${MSWE_AGENT_REF}" \
    && echo "${MSWE_AGENT_CONFIG_SHA256}  /opt/MSWE-agent/config/default.yaml" | sha256sum -c - \
    && git init -q /tmp/msb \
    && git -C /tmp/msb fetch -q --depth 1 https://github.com/multi-swe-bench/multi-swe-bench "${MSWE_HARNESS_JAVA_REF}" \
    && git -C /tmp/msb checkout -q FETCH_HEAD \
    && test "$(git -C /tmp/msb rev-parse HEAD)" = "${MSWE_HARNESS_JAVA_REF}" \
    && for org in ${MSWE_JAVA_ORGS}; do \
         test ! -e "/opt/MSWE-agent/multi_swe_bench/harness/repos/java/${org}"; \
         mkdir "/opt/MSWE-agent/multi_swe_bench/harness/repos/java/${org}"; \
         for f in /tmp/msb/multi_swe_bench/harness/repos/java/${org}/*.py; do \
           { echo "from __future__ import annotations"; cat "$f"; } \
             > "/opt/MSWE-agent/multi_swe_bench/harness/repos/java/${org}/$(basename "$f")"; \
         done; \
         echo "from multi_swe_bench.harness.repos.java.${org} import *" \
           >> /opt/MSWE-agent/multi_swe_bench/harness/repos/java/__init__.py; \
       done \
    && rm -rf /tmp/msb \
    && cd /opt/MSWE-agent \
    && python3 -m venv venv \
    && PIP_CONSTRAINT=/opt/mswe-agent-constraints.txt venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && PIP_CONSTRAINT=/opt/mswe-agent-constraints.txt venv/bin/pip install --no-cache-dir -r requirements.txt dataclasses-json \
    && venv/bin/pip freeze > /opt/mswe-agent-pip-freeze.txt \
    && cd /opt/MSWE-agent && venv/bin/python -m py_compile run.py multirun.py \
    && venv/bin/python -c "from swebench import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION; \
import sweagent.agent.models, config, dataclasses_json; \
from multi_swe_bench.harness.instance import Instance; import multi_swe_bench.harness.repos; \
regs = set(Instance._registry); \
need = {'apache/dubbo', 'fasterxml/jackson-core', 'fasterxml/jackson-databind', 'fasterxml/jackson-dataformat-xml', 'google/gson', 'googlecontainertools/jib'}; \
assert need <= regs, need - regs" \
    && chmod -R a+rX /opt/MSWE-agent \
    && git config --system --add safe.directory /opt/MSWE-agent

COPY mswe_agent_run.py /opt/gbench/mswe_agent_run.py
RUN grep -q "^def install_apt_archive_fallback" /opt/gbench/mswe_agent_run.py \
    && grep -q "^def install_rollout_error_record" /opt/gbench/mswe_agent_run.py \
    && cd /opt/MSWE-agent && venv/bin/python -c "import sys; sys.path[:0] = ['/opt/gbench', '.']; \
import mswe_agent_run as d; rec = d.install_client_timeout(d.CLIENT_TIMEOUT_S); \
from sweagent.agent import models; c = models.OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:1/v1'); \
assert rec['read_timeout_s'] == d.CLIENT_TIMEOUT_S == c.timeout.read, (rec, c.timeout); print(rec)"

LABEL org.gbench.mswe_agent="${MSWE_AGENT_REF}" \
      org.gbench.mswe_agent_config_sha256="${MSWE_AGENT_CONFIG_SHA256}" \
      org.gbench.mswe_agent_java_repos="multi-swe-bench@${MSWE_HARNESS_JAVA_REF}: ${MSWE_JAVA_ORGS}" \
      org.gbench.docker_cli="${DOCKER_CLI_VERSION}" \
      org.gbench.mswe_agent_driver="4"

# No ENTRYPOINT: gbench runs `docker run <image> /opt/MSWE-agent/venv/bin/python /opt/gbench/mswe_agent_run.py --spec ...`.
