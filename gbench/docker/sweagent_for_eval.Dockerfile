# SWE-agent-for-eval rollout image for gbench's `swe_bench_live` suite, built LOCALLY.
#
# SWE-bench-Live's evaluation README (microsoft/SWE-bench-Live@62d02936 evaluation/README.md, "The
# forked source codes of the agents we used to run SWE-bench-Live") names njukenanli/SWE-agent-for-eval
# as the SWE-agent the leaderboard's SWE-agent rows ran. This image carries that fork at a pinned
# commit, installed exactly as its README says (a Python 3.12 venv, `pip install --editable .`, then
# its vendored `swerex` copied into the venv's site-packages), with every dependency pinned by
# sweagent_for_eval.constraints.txt.
#
# gbench runs one container of this image per tab and batch:
#   /opt/SWE-agent-for-eval/venv/bin/sweagent run-batch --config /opt/SWE-agent-for-eval/config/default.yaml
#       --instances.type swe_bench --instances.subset <instances.jsonl> --output_dir <dir> ...
# with the output directory identity-mounted, the host network (the served model, reached through
# gbench's relay on 127.0.0.1, and the port each task container publishes) and the docker socket:
# SWE-agent's DockerDeployment builds and starts each instance image on the HOST daemon
# (docker-out-of-docker), as it does on the machine upstream ran it on.
#
# SWE-agent's DockerDeployment builds a runtime layer on top of every task image (swerex
# deployment/docker.py glibc_dockerfile: a standalone Python 3.11.8 under /root/python3.11 with
# swe-rex, whose server runs the model's commands in a bash session). The image applies gbench's
# heredoc fix to the vendored swerex (sweagent_for_eval_swerex_heredoc.patch): upstream that layer
# installs the latest swe-rex from PyPI, whose session split every command with bashlex and lost
# the body of a heredoc such as `cat <<EOF > file` (the shell then waited for the terminator until
# the fork's 90-minute command limit); with the patch the layer installs swe-rex 1.4.0 and its
# dependencies at pinned versions and the patched session code, which runs a command holding a
# heredoc whole, as bash runs a script. The label org.gbench.sweagent_for_eval_swerex_runtime
# records the sha256 of that session code (swerex/runtime/local.py).
#
# The image also applies gbench's fix to the fork's own read of the model's edited files at submit
# (sweagent_for_eval_review_decode.patch, sweagent/agent/agents.py): that read only fills the
# trajectory's edited_files30/50/70 record, and with the patch it decodes with errors="replace" and
# does not read a path that is not a regular file (a git submodule's directory, a FIFO), whose
# entry in the record stays empty; the submitted patch is untouched. The label
# org.gbench.sweagent_for_eval_agents records the sha256 of the patched agents.py.
#
# Build (context = gbench/docker):
#   docker build -t gbench-sweagent-for-eval:82422422-rex3 -f gbench/docker/sweagent_for_eval.Dockerfile gbench/docker
FROM ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.12 (ubuntu 24.04's python3) with venv, git, and the Docker CLI (client only - it talks to
# the mounted host daemon) at a pinned version, from the Ubuntu archive as of UBUNTU_SNAPSHOT
# (snapshot.ubuntu.com), so a rebuild installs the same package versions. The snapshot service is
# https-only, so ca-certificates is installed first from the base image's archive and then again at
# the snapshot's version (--allow-downgrades). The installed versions are recorded in
# /opt/sweagent-for-eval-dpkg-versions.txt.
ARG UBUNTU_SNAPSHOT=20260925T120000Z
ARG DOCKER_CLI_VERSION=5:29.8.0-1~ubuntu.24.04~noble
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i -E "s#^URIs: http://(archive|security)\.ubuntu\.com/ubuntu/?\$#URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" /etc/apt/sources.list.d/ubuntu.sources \
    && test "$(grep -c "snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" /etc/apt/sources.list.d/ubuntu.sources)" = 2 \
    && apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        ca-certificates curl git gnupg python3 python3-venv \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends "docker-ce-cli=${DOCKER_CLI_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version && python3 --version | grep -q "^Python 3\.12\." \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/sweagent-for-eval-dpkg-versions.txt \
    && grep -qx "docker-ce-cli=${DOCKER_CLI_VERSION}" /opt/sweagent-for-eval-dpkg-versions.txt

# The fork at a pinned commit, installed as its README says. The build checks the commit, the sha256
# of the config gbench runs (config/default.yaml), the vendored swerex's runtime/local.py before
# and after gbench's heredoc patch, that the fork's own `sweagent` and vendored (patched) `swerex`
# 1.4.0 are what the venv imports, that the runtime layer it builds installs the pinned swe-rex,
# and that the CLI starts as a non-root user (gbench runs it as the caller's uid; `sweagent` reads
# its own git commit at import, hence safe.directory); the resolved set is recorded next to the
# constraints.
ARG SWE_AGENT_FOR_EVAL_REF=824224225e1e952b48da91f2d1bde38adb2115f2
ARG SWE_AGENT_FOR_EVAL_CONFIG_SHA256=05e707df58e9732ac4053f8bda0c0b175aa10f5c833f5419ba1443a01d9eb981
# sha256 of swerex/runtime/local.py as swe-rex 1.4.0 ships it, and after gbench's heredoc patch.
ARG SWEREX_RUNTIME_UPSTREAM_SHA256=d56f22b48b02d4df6644cfa04cc16c979b0498ef1c26058b82fb2d4a60bf9ac7
ARG SWEREX_RUNTIME_SHA256=d67d68966754cc73a3c2ae084fdfe90e7fb274e3ae36da6e619e1c4155372ed4
# sha256 of sweagent/agent/agents.py at the pinned commit, and after gbench's review-decode patch
# (sweagent_for_eval_review_decode.patch: the read of the model's edited files at submit decodes
# tolerantly and skips paths that are not regular files).
ARG SWEAGENT_AGENTS_UPSTREAM_SHA256=836541b17cc3977f298d88e9653ddb1f433a092e1bb63a9a79dfc60843e1dc49
ARG SWEAGENT_AGENTS_SHA256=5a56a297803679c850ee356554a175a45215226b9ca09e3717ee958be1eb4217
COPY sweagent_for_eval.constraints.txt /opt/sweagent-for-eval-constraints.txt
COPY sweagent_for_eval_swerex_heredoc.patch /opt/sweagent-for-eval-swerex-heredoc.patch
COPY sweagent_for_eval_review_decode.patch /opt/sweagent-for-eval-review-decode.patch
RUN git init -q /opt/SWE-agent-for-eval \
    && git -C /opt/SWE-agent-for-eval fetch -q --depth 1 https://github.com/njukenanli/SWE-agent-for-eval "${SWE_AGENT_FOR_EVAL_REF}" \
    && git -C /opt/SWE-agent-for-eval checkout -q FETCH_HEAD \
    && test "$(git -C /opt/SWE-agent-for-eval rev-parse HEAD)" = "${SWE_AGENT_FOR_EVAL_REF}" \
    && echo "${SWE_AGENT_FOR_EVAL_CONFIG_SHA256}  /opt/SWE-agent-for-eval/config/default.yaml" | sha256sum -c - \
    && cd /opt/SWE-agent-for-eval \
    && echo "${SWEAGENT_AGENTS_UPSTREAM_SHA256}  sweagent/agent/agents.py" | sha256sum -c - \
    && git apply /opt/sweagent-for-eval-review-decode.patch \
    && echo "${SWEAGENT_AGENTS_SHA256}  sweagent/agent/agents.py" | sha256sum -c - \
    && python3 -m venv venv \
    && PIP_CONSTRAINT=/opt/sweagent-for-eval-constraints.txt venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && PIP_CONSTRAINT=/opt/sweagent-for-eval-constraints.txt venv/bin/pip install --no-cache-dir --editable . \
    && cp -r swerex venv/lib/python3.12/site-packages/swerex \
    && echo "${SWEREX_RUNTIME_UPSTREAM_SHA256}  venv/lib/python3.12/site-packages/swerex/runtime/local.py" | sha256sum -c - \
    && git apply --unsafe-paths --directory=/opt/SWE-agent-for-eval/venv/lib/python3.12/site-packages /opt/sweagent-for-eval-swerex-heredoc.patch \
    && echo "${SWEREX_RUNTIME_SHA256}  venv/lib/python3.12/site-packages/swerex/runtime/local.py" | sha256sum -c - \
    && venv/bin/pip freeze > /opt/sweagent-for-eval-pip-freeze.txt \
    && cd /tmp && /opt/SWE-agent-for-eval/venv/bin/python -c "import sweagent, swerex, sys; import sweagent.agent.agents as agents; \
assert sweagent.__file__.startswith('/opt/SWE-agent-for-eval/sweagent/'), sweagent.__file__; \
assert swerex.__file__.startswith('/opt/SWE-agent-for-eval/venv/lib/python3.12/site-packages/swerex/'), swerex.__file__; \
assert swerex.__version__ == '1.4.0', swerex.__version__; \
assert agents.RexCommand.__module__ == 'swerex.runtime.abstract', agents.RexCommand; \
from swerex.deployment import docker as d; from swerex.deployment.config import DockerDeploymentConfig as C; \
assert d.GBENCH_RUNTIME_LOCAL_SHA256 == '${SWEREX_RUNTIME_SHA256}', d.GBENCH_RUNTIME_LOCAL_SHA256; \
f = d.DockerDeployment.from_config(C(image='x', python_standalone_dir='/root')).glibc_dockerfile(); \
assert 'swe-rex==1.4.0' in f and 'install --no-cache-dir swe-rex' not in f, f" \
    && chmod -R a+rX /opt/SWE-agent-for-eval \
    && git config --system --add safe.directory /opt/SWE-agent-for-eval \
    && su -s /bin/sh nobody -c "HOME=/tmp /opt/SWE-agent-for-eval/venv/bin/sweagent run-batch --help > /dev/null"

LABEL org.gbench.sweagent_for_eval="${SWE_AGENT_FOR_EVAL_REF}" \
      org.gbench.sweagent_for_eval_config_sha256="${SWE_AGENT_FOR_EVAL_CONFIG_SHA256}" \
      org.gbench.sweagent_for_eval_swerex_runtime="${SWEREX_RUNTIME_SHA256}" \
      org.gbench.sweagent_for_eval_agents="${SWEAGENT_AGENTS_SHA256}" \
      org.gbench.docker_cli="${DOCKER_CLI_VERSION}"

# No ENTRYPOINT: gbench runs `docker run <image> /opt/SWE-agent-for-eval/venv/bin/sweagent run-batch ...`.
