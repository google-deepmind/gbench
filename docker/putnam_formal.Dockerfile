# gbench-putnam-formal: the Lean 4 environment of the putnam_formal suite.
#
# It is PutnamBench's own lean4/ Lake project at the pinned commit (the commit putnam_formal.py
# reads its statements from), with the Lean toolchain and Mathlib that commit pins:
# lean4/lean-toolchain = leanprover/lean4:v4.33.1, lean4/lakefile.lean requires mathlib
# @ "v4.33.1", and lean4/lake-manifest.json locks it to 0df444a360eaa60ab8c11dca51a86af692955474.
# The build repeats PutnamBench's CI (.github/workflows/lean.yml): fetch the Mathlib olean cache,
# build all 672 statements, insert the answers with scripts/rewrite_solutions.py and build
# putnam_with_solutions. A statement that does not compile under the pinned toolchain fails the
# build.
#
# Build (the context is not used):
#   docker build -t gbench-putnam-formal:97b5e945 -f docker/putnam_formal.Dockerfile docker
# putnam_formal uses that tag by default (GBENCH_PUTNAM_FORMAL_LEAN_IMAGE overrides it) and
# checks the image's toolchain, Mathlib revision and PutnamBench commit at startup.
#
# The build needs network access (Debian packages, the elan toolchain, the Mathlib olean cache
# and the PutnamBench tarball). The image is about 25 GB, mostly Mathlib's oleans.
# `lake exe cache get` unpacks thousands of files in parallel; if it fails with "too many open
# files", pass `--ulimit nofile=1048576:1048576` to `docker build`.

# Debian 12 with elan 3.1.1 (the elan release PutnamBench's CI installs). Only elan is used from
# it: the toolchain comes from the project's lean-toolchain file.
FROM leanprovercommunity/lean4@sha256:d61f7052fa82e7e726db46984ef4f11c84525eabd4a8d1d20ba80f1ccee34018

ARG PUTNAMBENCH_REVISION=97b5e9458b04e3d1044e8f8afe1724ff133f5a53
# `git rev-parse <PUTNAMBENCH_REVISION>:lean4/src`
ARG PUTNAMBENCH_SRC_TREE=f10ab3f77620e127ca58cae55924f6449da403ce
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.33.1
ARG MATHLIB_REV=0df444a360eaa60ab8c11dca51a86af692955474

USER root
# python3 runs PutnamBench's scripts/rewrite_solutions.py, as its CI does.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PutnamBench's lean4/ directory at the pinned commit is the Lake project.
RUN curl -fsSL -o /tmp/putnambench.tar.gz \
        "https://codeload.github.com/trishullab/PutnamBench/tar.gz/${PUTNAMBENCH_REVISION}" \
    && mkdir -p /opt/putnamenv \
    && tar -xzf /tmp/putnambench.tar.gz -C /opt/putnamenv --strip-components=2 \
        "PutnamBench-${PUTNAMBENCH_REVISION}/lean4" \
    && rm /tmp/putnambench.tar.gz
WORKDIR /opt/putnamenv

# The statements must be the pinned release (their git tree id), and the toolchain and Mathlib
# pins must be the ones putnam_formal.py checks for.
RUN python3 -c "import hashlib, os, sys; \
blob = lambda d: hashlib.sha1(b'blob %d\0' % len(d) + d).digest(); \
names = sorted(os.listdir('src')); \
body = b''.join(b'100644 ' + n.encode() + b'\0' + blob(open('src/' + n, 'rb').read()) for n in names); \
tree = hashlib.sha1(b'tree %d\0' % len(body) + body).hexdigest(); \
sys.exit(0 if tree == '${PUTNAMBENCH_SRC_TREE}' else 'lean4/src has git tree ' + tree)" \
    && test "$(cat lean-toolchain)" = "${LEAN_TOOLCHAIN}" \
    && grep -q "\"rev\": \"${MATHLIB_REV}\"" lake-manifest.json \
    && test "$(ls src | wc -l)" = "672"

RUN elan toolchain install "$(cat lean-toolchain)"

# PutnamBench CI: `lake exe cache get`, `lake build`, then the answer-inserted variant.
RUN ulimit -n "$(ulimit -Hn)" && lake exe cache get
RUN lake build
RUN python3 scripts/rewrite_solutions.py && lake build putnam_with_solutions

# Read by putnam_formal's startup check.
RUN printf '%s\n' "${PUTNAMBENCH_REVISION}" > /opt/putnamenv/.gbench-putnambench-revision

# Sanity: `import Mathlib` resolves through the project's Lake environment, which is how every
# putnam_formal compile runs (`cd /opt/putnamenv && lake env lean ...`).
RUN printf 'import Mathlib\n' > /tmp/probe.lean && lake env lean /tmp/probe.lean \
    && rm /tmp/probe.lean
