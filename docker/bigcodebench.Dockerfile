# gbench-bigcodebench: the BigCodeBench evaluation harness for gbench's `bigcodebench` suite,
# built LOCALLY (gbench never pulls a registry image).
#
# Vendored from bigcode-project/bigcodebench Docker/Evaluate.Dockerfile at the pinned commit below
# (the recipe upstream publishes as bigcodebench/bigcodebench-evaluate). gbench deltas, all for
# reproducibility:
#   * the source is checked out at a PINNED commit instead of `git clone` of main HEAD (upstream
#     busts the build cache on every new commit via an ADD of the GitHub commits API);
#   * requirements-eval.txt (the libraries the benchmark tasks import, pinned upstream) is read from
#     that same commit instead of from refs/heads/main;
#   * the base image and the unpinned top-level pip packages (plus torch, which accelerate pulls
#     in) are pinned to the versions installed in upstream's last published evaluate image
#     (bigcodebench/bigcodebench-evaluate v0.2.4 / latest, 2025-02-23), read from its pip freeze.
#     numpy stays at 1.24.3 in that step, as it did upstream (the unpinned `numpy` was already
#     satisfied): requirements-eval.txt then force-reinstalls numpy 1.21.2 over it with `-I`, and
#     a numpy 2 install at that point would leave numpy 2 modules behind that break pyarrow;
#   * after upstream's pip steps, the installed Python packages are pinned to the ones of the
#     image built on 2026-09-25 (bigcodebench.pins.txt, enforced by pinned_pip_check.py), because
#     those steps still resolve their unpinned transitive dependencies at build time, as
#     upstream's do. The Debian packages float with the archive; the build records the versions
#     it installed in /opt/gbench/dpkg-versions.txt (and the Python ones in pip-freeze.txt).
#
# Build (no build context needed beyond this docker/ dir):
#   docker build -t gbench-bigcodebench:09dd993f -f docker/bigcodebench.Dockerfile docker
FROM python:3.10.16-slim-bookworm@sha256:f9fd9a142c9e3bc54d906053b756eb7e7e386ee1cf784d82c251cf640c502512

# install git, g++ and python3-tk
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    python3-tk \
    zip \
    unzip \
    procps \
    r-base \
    libgdal-dev \
    # Add these new dependencies for matplotlib
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    python3-dev \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pip==25.0.1

# Add a new user "bigcodebenchuser"
RUN adduser --disabled-password --gecos "" bigcodebenchuser

RUN rm -rf /bigcodebench

# Acquire benchmark code to local, at the pinned commit (upstream main, 2025-10-15).
ARG BIGCODEBENCH_REF=09dd993f46c3fbf3a799465bb96d524edcb0b199
RUN git clone https://github.com/bigcode-project/bigcodebench.git /bigcodebench && \
    git -C /bigcodebench checkout "${BIGCODEBENCH_REF}"

RUN pip install numpy==1.24.3 pyarrow==14.0.1

RUN cd /bigcodebench && \
    pip install . --no-deps

RUN pip install \
    appdirs==1.4.4 \
    fire==0.7.0 \
    multipledispatch==1.0.0 \
    pqdm==0.2.0 \
    tempdir==0.7.1 \
    termcolor==2.5.0 \
    tqdm==4.67.1 \
    tree_sitter==0.24.0 \
    tree-sitter-python==0.23.6 \
    wget==3.2 \
    transformers==4.49.0 \
    datasets==3.3.2 \
    pyarrow==19.0.1 \
    gradio-client==1.7.1 \
    numpy==1.24.3 \
    rich==13.9.4 \
    accelerate==1.4.0 \
    torch==2.6.0 \
    anthropic==0.46.0 \
    google-generativeai==0.8.4 \
    mistralai==1.5.0 \
    openai==1.64.0 \
    e2b==1.1.0

RUN pip install -I --timeout 2000 -r /bigcodebench/Requirements/requirements-eval.txt

# Ensure the numpy version is compatible with the datasets version
RUN pip install datasets==2.17.0

# The recorded Python package set (see the header), then the record of what this build installed.
COPY bigcodebench.pins.txt pinned_pip_check.py /opt/gbench/
RUN python3 /opt/gbench/pinned_pip_check.py /opt/gbench/bigcodebench.pins.txt \
        --source bigcodebench --enforce --freeze /opt/gbench/pip-freeze.txt \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt

# Pre-install the dataset (the v0.1.4 split, BIGCODEBENCH_VERSION at the pinned commit)
RUN python3 -c "from bigcodebench.data import get_bigcodebench; get_bigcodebench(subset='full'); get_bigcodebench(subset='hard')"

WORKDIR /app

RUN chown -R bigcodebenchuser:bigcodebenchuser /app

RUN chmod -R 777 /app

USER bigcodebenchuser

ENTRYPOINT ["python3", "-m", "bigcodebench.evaluate"]

CMD ["sh", "-c", "pids=$(ps -u $(id -u) -o pid,comm | grep 'bigcodebench' | awk '{print $1}'); if [ -n \"$pids\" ]; then echo $pids | xargs -r kill; fi; rm -rf /tmp/*"]
