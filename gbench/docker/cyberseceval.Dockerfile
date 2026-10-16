# Insecure Code Detector image for gbench's cyberseceval suite, built LOCALLY (gbench never pulls
# it).
#
# CyberSecEval-instruct scores each generated program with Meta's Insecure Code Detector (ICD,
# PurpleLlama CodeShield/insecure_code_detector: a regex analyzer and semgrep per language). This
# image holds PurpleLlama at the commit gbench pins (CodeShield/, the instruct-v2.json prompt
# file and the MIT licenses of both, fetched by commit so git checks their content) and
# semgrep 1.51.0 with PyYAML 6.0.1, the releases upstream's CybersecurityBenchmarks/requirements.txt
# pins, on Python 3.10 (upstream's README: "python 3.10 is required"). semgrep 1.51.0 pins
# urllib3~=1.26 and ruamel.yaml<0.18, which would downgrade the serving environment's urllib3 2.x
# and ruamel.yaml, so it never goes into gbench's own interpreter: gbench runs
# cyberseceval_icd_worker.py in a fresh container of this image for its probe (with the semgrep
# self-test), for the prompt file and for each scoring pass, with --network none, a read-only root
# file system, a tmpfs /tmp, no capabilities and an unprivileged user; no variable of the host
# environment, key or token enters it. See docs/evals/cyberseceval.md.
#
# Build (context = gbench/docker; the tag is the one gbench runs), from the gbench repository
# root:
#   docker build -t gbench-cyberseceval:114d02388213-semgrep-1.51.0 -f gbench/docker/cyberseceval.Dockerfile gbench/docker
#
# Base: python:3.10-slim at a fixed digest, for both stages (recorded in the image's labels; gbench
# compares it with this default).
ARG BASE_IMAGE=python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2
FROM ${BASE_IMAGE} AS purplellama

# PurpleLlama at the pinned commit: a blobless, depth-1 fetch of that commit with a sparse
# checkout of the files the ICD and the prompts need (git verifies every object against the
# commit), then the prompt file's sha256.
ARG PURPLELLAMA_COMMIT=114d02388213ae791650b10fa61c8785661c2602
ARG INSTRUCT_V2_SHA256=df19f7c51911d46d0ae5c069b7093728db8e682890a44317a211ac61cb02c181
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && git init -q /src \
    && git -C /src remote add origin https://github.com/meta-llama/PurpleLlama.git \
    && git -C /src config core.sparseCheckout true \
    && printf '%s\n' /CodeShield/ /CybersecurityBenchmarks/LICENSE \
        /CybersecurityBenchmarks/requirements.txt \
        /CybersecurityBenchmarks/datasets/instruct/instruct-v2.json \
        > /src/.git/info/sparse-checkout \
    && git -C /src fetch -q --depth 1 --filter=blob:none origin "${PURPLELLAMA_COMMIT}" \
    && git -C /src -c advice.detachedHead=false checkout -q FETCH_HEAD \
    && test "$(git -C /src rev-parse HEAD)" = "${PURPLELLAMA_COMMIT}" \
    && echo "${INSTRUCT_V2_SHA256}  /src/CybersecurityBenchmarks/datasets/instruct/instruct-v2.json" \
        | sha256sum -c - \
    && grep -qx 'semgrep==1.51.0' /src/CybersecurityBenchmarks/requirements.txt \
    && grep -qx 'pyyaml==6.0.1' /src/CybersecurityBenchmarks/requirements.txt \
    && mkdir -p /out/PurpleLlama/CybersecurityBenchmarks/datasets/instruct \
    && cp -a /src/CodeShield /out/PurpleLlama/ \
    && cp -a /src/CybersecurityBenchmarks/LICENSE /src/CybersecurityBenchmarks/requirements.txt \
        /out/PurpleLlama/CybersecurityBenchmarks/ \
    && cp -a /src/CybersecurityBenchmarks/datasets/instruct/instruct-v2.json \
        /out/PurpleLlama/CybersecurityBenchmarks/datasets/instruct/ \
    && echo "${PURPLELLAMA_COMMIT}" > /out/PurpleLlama/COMMIT

FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG BASE_IMAGE
ARG PURPLELLAMA_COMMIT=114d02388213ae791650b10fa61c8785661c2602
ARG SEMGREP_VERSION=1.51.0
ARG PYYAML_VERSION=6.0.1

# semgrep and PyYAML at upstream's pins, every other package at the constraints' release (wheels
# only).
COPY cyberseceval.constraints.txt /opt/cyberseceval/constraints.txt
RUN python -m pip install --only-binary=:all: -c /opt/cyberseceval/constraints.txt \
        "semgrep==${SEMGREP_VERSION}" "PyYAML==${PYYAML_VERSION}" \
    && python -m pip check

COPY --from=purplellama /out/PurpleLlama /opt/PurpleLlama
COPY cyberseceval_icd_worker.py /opt/cyberseceval/cyberseceval_icd_worker.py

# The build-time check (as root, so the ICD's import can link osemgrep to semgrep-core inside the
# semgrep package, as upstream's oss.py does): the installed set is exactly the constraints, and
# the ICD flags the self-test with its semgrep analyzer. Then the files are compiled and made
# read-only for the runtime user.
RUN python /opt/cyberseceval/cyberseceval_icd_worker.py --verify-build \
        /opt/cyberseceval/constraints.txt /opt/cyberseceval/pip-freeze.txt \
    && python -m compileall -q /opt/PurpleLlama/CodeShield /opt/cyberseceval \
    && chmod -R a+rX,go-w /opt/PurpleLlama /opt/cyberseceval

LABEL org.gbench.suite="cyberseceval" \
      org.gbench.cyberseceval.purplellama="${PURPLELLAMA_COMMIT}" \
      org.gbench.cyberseceval.semgrep="${SEMGREP_VERSION}" \
      org.gbench.cyberseceval.pyyaml="${PYYAML_VERSION}" \
      org.gbench.cyberseceval.base="${BASE_IMAGE}"

ENV HOME=/tmp
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "/opt/cyberseceval/cyberseceval_icd_worker.py"]
