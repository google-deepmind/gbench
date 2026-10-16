# gbench-aider-benchmark: isolated image for Aider Polyglot's native-edit-format pass@2 scoring.
#
# This is aider's own benchmark image (Aider-AI/aider benchmark/Dockerfile: python3.11 +
# openjdk-21 + go + rust + node + gcc + aider) with four differences:
# * aider is fetched at the pinned commit AIDER_COMMIT (the suite's aider_polyglot.AIDER_COMMIT)
#   instead of copied from a local checkout, and the image is labelled
#   org.gbench.aider.commit=<that commit>; gbench refuses an image with another label (or none).
#   benchmark.py records the commit it runs (`commit_hash`, the first 7 characters) in every
#   exercise's result file.
# * a pinned setuptools-scm version. The shallow fetch carries no tags, so the upstream
#   `uv pip install -e /aider[dev]` could not derive a version; we set
#   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIDER_CHAT so the editable build succeeds.
# * the JavaScript toolchain pinned: jest to 29.7.0, Node.js to NodeSource's 20.x build of the
#   image this file built on 2026-09-28 (upstream installs the newest 20.x; NodeSource keeps
#   every build in its repository), and the npm packages' own dependencies to the releases that
#   image resolved (npm's `before`, NPM_CONFIG_BEFORE). Upstream installs `jest` unpinned next
#   to the exercises' other devDependencies at their versions; every JavaScript exercise's
#   package.json asks for "jest": "^29.7.0", which is what an unpinned install gave until Jest 30
#   (June 2025). Jest 30 removed the matcher aliases, so the specs that call toThrowError
#   (affine-cipher, resistor-color-trio) fail with "expect(...).toThrowError is not a function"
#   whatever the model wrote (checked 2026-09-27 with the reference solution in an image that had
#   resolved jest 30.5.1).
# * the Python packages pip and uv install resolve against aider_polyglot.constraints.txt (the
#   pip-managed packages of the image this file built on 2026-09-28), and the build fails if
#   anything else is installed. The build records the Python, Debian and toolchain versions it
#   installed in /opt/gbench (Rust's stable toolchain and the Debian and Ubuntu packages still
#   resolve at build time, as upstream's do).
# gbench does not install aider into the serving venv (it would downgrade openai and move
# numpy/hf_hub/pydantic). The runner keeps the container's HOME (/root, where rustup installs
# Rust with no RUSTUP_HOME / CARGO_HOME), as upstream's benchmark/docker.sh does.
#
# Build (context = gbench's docker/ directory, which holds the constraints file and the check):
#   docker build -t aider-benchmark:5dc9490b -f docker/aider_polyglot.Dockerfile docker   # from the repo root
#
# The runner (aider_polyglot.py) then `docker run`s this image against the gbench-served endpoint.

FROM buildpack-deps:jammy@sha256:b2efc38243ba4acd7c6f7e95891874f4ff2030a857e0f209435819a8b2a93bff

# Python 3.11 + Java (openjdk-21) + C/C++ build deps.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    cmake \
    libboost-all-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ca-certificates-java \
    openjdk-21-jdk \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Go (arch-detected).
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then GOARCH="amd64"; \
    elif [ "$ARCH" = "aarch64" ]; then GOARCH="arm64"; \
    else false; fi && \
    curl -L "https://golang.org/dl/go1.21.5.linux-$GOARCH.tar.gz" -o go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && \
    rm go.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Rust.
ADD https://sh.rustup.rs /tmp/rustup.sh
RUN chmod +x /tmp/rustup.sh && /tmp/rustup.sh -y && rm /tmp/rustup.sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Node.js (NodeSource's 20.x build, pinned) + the JS exercise deps (jest/babel/eslint), at the
# exercises' package.json versions; their own dependencies resolve as of NPM_CONFIG_BEFORE (a
# build argument: the time that image ran this step), not as of the build day.
ARG NODE_VERSION=20.20.2-1nodesource1
ARG NPM_CONFIG_BEFORE=2026-09-27T05:30:00Z
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y "nodejs=${NODE_VERSION}" && \
    test "$(dpkg-query -W -f '${Version}' nodejs)" = "${NODE_VERSION}" && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /npm-install && \
    cd /npm-install && \
    npm init -y && \
    npm install \
    jest@29.7.0 \
    @babel/core@7.25.2 \
    @exercism/babel-preset-javascript@0.2.1 \
    @exercism/eslint-config-javascript@0.6.0 \
    @types/jest@29.5.12 \
    @types/node@20.12.12 \
    babel-jest@29.6.4 \
    core-js@3.37.1 \
    eslint@8.49.0

# aider at the pinned commit (a shallow fetch of that one commit), checked, and recorded as a label.
ARG AIDER_COMMIT=5dc9490bb35f9729ef2c95d00a19ccd30c26339c
RUN git init -q /aider \
    && git -C /aider remote add origin https://github.com/Aider-AI/aider.git \
    && git -C /aider fetch -q --depth 1 origin "${AIDER_COMMIT}" \
    && git -C /aider checkout -q --detach FETCH_HEAD \
    && test "$(git -C /aider rev-parse HEAD)" = "${AIDER_COMMIT}"
LABEL org.gbench.aider.commit="${AIDER_COMMIT}"
COPY aider_polyglot.constraints.txt pinned_pip_check.py /opt/gbench/
RUN pip3 install --no-cache-dir -c /opt/gbench/aider_polyglot.constraints.txt --upgrade pip uv
# The shallow fetch has no tags -> pin a version so the setuptools-scm build succeeds.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIDER_CHAT=0.0.0
RUN uv pip install --system --no-cache-dir -c /opt/gbench/aider_polyglot.constraints.txt \
        -e /aider[dev]
# The pip-managed packages are exactly the constrained ones; then the record of what this build
# installed.
RUN python3 /opt/gbench/pinned_pip_check.py /opt/gbench/aider_polyglot.constraints.txt \
        --source aider-chat --ignore-dir /usr/lib/python3/dist-packages \
        --freeze /opt/gbench/pip-freeze.txt \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt \
    && { rustc --version; cargo --version; go version; node --version; npm --version; \
         java -version 2>&1 | head -1; g++ --version | head -1; } > /opt/gbench/toolchain-versions.txt
# benchmark.py reads `repo.head.object.hexsha` (the aider repo's commit) for its run label; the
# bind-mounted exercises and results belong to another uid, so git must trust every directory.
RUN git config --global --add safe.directory '*' \
    && git config --global user.email "gbench@localhost" \
    && git config --global user.name "gbench" \
    && test -z "$(git -C /aider status --porcelain --untracked-files=no)"
WORKDIR /aider
