# Canonical OSWorld orchestrator image for gbench's `ui_control_osworld` suite, built LOCALLY.
#
# OSWorld (github.com/xlang-ai/OSWorld) is an EXECUTION-BASED desktop computer-use benchmark: 369
# real tasks over an Ubuntu-desktop VM. A computer-use agent observes screenshots and emits
# pyautogui mouse/keyboard actions over many steps; each task is scored by a DETERMINISTIC per-task
# evaluator (getters read the final VM state + compare to a reference) -> NO LLM judge. This
# orchestrator bakes the pinned OSWorld harness + its deps + the docker SDK, and drives the run
# docker-OUT-of-docker: OSWorld's `docker` provider (DesktopEnv(provider_name="docker")) spawns a
# SIBLING QEMU-VM container (happysixd/osworld-docker) on the host daemon per task, bind-mounting the
# Ubuntu qcow2 disk. The agent (mm_agents PromptAgent) runs host-side in this orchestrator and calls
# the served model over /v1.
#
# Build (context = gbench/docker):
#   docker build -t gbench-ui-control-osworld:fc31a904 -f gbench/docker/ui_control_osworld.Dockerfile gbench/docker
#
# A GRADED run REQUIRES a host with /dev/kvm (nested virtualization): the OSWorld VM only becomes
# ready (300s /screenshot poll) with hardware acceleration; without KVM QEMU falls back to TCG
# software emulation and the desktop will not boot in time. It also needs the ~12GB Ubuntu.qcow2
# (HF xlangai/ubuntu_osworld) at an identity-mounted path and the served model. NO judge / NO
# GEMINI. See docs/evals/ui_control_osworld.md.
FROM python:3.11@sha256:8ce0c4b7bad2a0939d3fe311e30f1f12b491b0ba335fe1f2daa1b99894588503

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System libs: xvfb (pyautogui/pynput import a display at module load), opencv runtime libs, git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates xvfb libgl1 libglib2.0-0 x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Pin the OSWorld harness (no upstream tags -> pin a commit).
ARG OSWORLD_REF=fc31a9049664292fcb35d6e501ee1dc839f2cf6d
RUN git clone https://github.com/xlang-ai/OSWorld /app/OSWorld \
    && cd /app/OSWorld && git checkout "${OSWORLD_REF}"
ENV OSWORLD_REF=fc31a9049664292fcb35d6e501ee1dc839f2cf6d
WORKDIR /app/OSWorld
# Harness deps + the docker SDK (the `docker` provider uses docker.from_env(), not the CLI),
# installed against ui_control_osworld.constraints.txt (the resolved set of the image whose model
# wiring was checked; requirements.txt leaves most of it unpinned or as ~= ranges). The build fails
# unless the interpreter holds exactly that set. The Python set and the Debian packages (which
# float with the Debian archive) are recorded in /opt/gbench.
COPY ui_control_osworld.constraints.txt /opt/gbench/ui_control_osworld.constraints.txt
RUN export PIP_CONSTRAINT=/opt/gbench/ui_control_osworld.constraints.txt \
    && pip install -r requirements.txt && pip install "docker>=7,<8" \
    && python -m pip freeze --all | LC_ALL=C sort > /opt/gbench/pip-freeze.txt \
    && grep -v -e '^#' -e '^$' /opt/gbench/ui_control_osworld.constraints.txt | LC_ALL=C sort \
        | diff - /opt/gbench/pip-freeze.txt \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt

# gbench adapter.
RUN mkdir -p /app/adapter
COPY ui_control_osworld_run.py /app/adapter/ui_control_osworld_run.py
COPY ui_control_osworld_entrypoint.sh /app/adapter/ui_control_osworld_entrypoint.sh
RUN chmod +x /app/adapter/ui_control_osworld_entrypoint.sh

# The launcher version: it writes into the run's own directory (OSWORLD_RUN_DIR; OSWorld resumes
# from whatever its result_dir holds), stops on a malformed shard or limit, reports the shard it
# applied and how it drew an --eval-limit subset (a seeded draw across the domains), does not
# start OSWorld for an empty shard, records per example every model request that got no reply or a
# failure status (gbench leaves those tasks out of the score) and hands the run directory to the
# gbench runner's user at the end (version 3), and passes OSWORLD_TEMPERATURE as run.py's
# --temperature (version 4). The gbench runner refuses an image without this label at this value
# (ui_control_osworld.LAUNCHER_VERSION).
LABEL org.gbench.ui-control-osworld-launcher="4"

ENV OSWORLD_DIR=/app/OSWorld
ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/ui_control_osworld_entrypoint.sh"]
