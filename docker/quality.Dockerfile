# GemmaClaw QA suite environment for gbench's quality pillar, built LOCALLY (never pulled).
#
# gbench builds this image itself on first use (gbench/runners/quality.py, ensure_quality_image)
# from a build context it assembles in a temporary directory:
#
#   src/                     `git archive <commit>` of the GemmaClaw commit
#                            (DEFAULT_GEMMACLAW_COMMIT or --gemmaclaw-commit), with gbench's two
#                            recorded edits applied
#                            (docs/quality.md, "The two edits of the checkout")
#   gemmaclaw_patches.json   the record of those edits (file, change, state, sha256)
#   quality_entrypoint.sh, quality_relay_forwarder.py, quality_npm_cache.mjs, and this file
#
# The tag is gbench-quality:<commit[:12]>-<digest[:12]>, the digest covering this file, its
# helpers and the edits, so a change to any of them builds a new image.
#
# The image holds:
#   /opt/gemmaclaw   the checkout, installed (`pnpm install --frozen-lockfile`) and built the way
#                    GemmaClaw's own Dockerfile builds it (`pnpm build`, `pnpm ui:build`,
#                    `pnpm qa:lab:build`), then `pnpm openclaw qa suite --help`, so the private-QA
#                    build and runtime sync that GemmaClaw's scripts/run-node.mjs does on first use
#                    happen here. It is owned by root and read-only at run time. Its .git is an
#                    empty directory: GemmaClaw's private QA CLI and its bundled-plugin loader run
#                    only from a source checkout, which they recognize by a .git entry, as in a
#                    clone; git itself finds no repository there, so run-node compares the build
#                    stamp with source mtimes and never rebuilds (the last step below fails the
#                    build if it would), and GIT_COMMIT names the commit where GemmaClaw shows it
#                    (its version line, its build info).
#   /opt/gbench      the entrypoint, the relay forwarder, the edits record, the commit, a
#                    metadata bundle (the QA scenarios and the plugin manifests) gbench reads
#                    before a run, and npm-cache/: the runtime dependencies of every bundled
#                    plugin (quality_npm_cache.mjs), which GemmaClaw installs with npm when it
#                    loads a plugin; the run installs them from it offline.
#   users            qa (10001) runs the suite and every tool the model under test calls;
#                    gbrelay (10002) runs the forwarder.
#
# The base image is the one GemmaClaw's Dockerfile pins at the default commit
# (OPENCLAW_NODE_BOOKWORM_IMAGE, node:24-bookworm by digest). Additions over it: python3 (the
# forwarder), the packages GemmaClaw's runtime image installs (procps hostname curl git lsof
# openssl), and Google Chrome, which the Control UI scenario opens through Playwright's
# `channel: "chrome"`, installed with the checkout's own playwright-core (`install --with-deps
# chrome`, the current stable Chrome at build time).
#
# How gbench runs it, one container per run (docs/quality.md, "The container"): --network none,
# a read-only root filesystem, tmpfs /tmp and /home/qa, anonymous volumes for the suite output,
# the checkout's .artifacts (where the suite stages each gateway's plugin tree), gbench's status
# files and the npm cache, --cap-drop ALL plus SETUID and SETGID for the
# entrypoint, and no-new-privileges. The only host paths it sees are the relay's socket
# directory and the suite's openclaw.json, both read-only.
ARG NODE_IMAGE=node:24-bookworm@sha256:3a09aa6354567619221ef6c45a5051b671f953f0a1924d1f819ffb236e520e6b
FROM ${NODE_IMAGE}

ARG GEMMACLAW_COMMIT
ARG MIN_NODE=22.19

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      procps hostname curl git lsof openssl python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 10001 qa && useradd -u 10001 -g 10001 -d /home/qa -M -s /bin/bash qa \
 && groupadd -g 10002 gbrelay \
 && useradd -u 10002 -g 10002 -d /nonexistent -M -s /usr/sbin/nologin gbrelay

ENV COREPACK_HOME=/usr/local/share/corepack \
    GIT_COMMIT=${GEMMACLAW_COMMIT}
COPY src/ /opt/gemmaclaw/
WORKDIR /opt/gemmaclaw

RUN test -n "${GEMMACLAW_COMMIT}" \
 && node -e 'const [a, b] = process.versions.node.split(".").map(Number); \
      const [x, y] = process.argv[1].split(".").map(Number); \
      if (a < x || (a === x && b < y)) { \
        console.error("node " + process.versions.node + " is older than " + process.argv[1]); \
        process.exit(1); }' "${MIN_NODE}" \
 && install -d -m 0755 "${COREPACK_HOME}" \
 && corepack enable \
 && corepack prepare "$(node -p "require('./package.json').packageManager")" --activate \
 && chmod -R a+rX "${COREPACK_HOME}"

RUN install -d -m 0755 .git \
 && GEMMACLAW_PREPARING=1 pnpm install --frozen-lockfile \
 && rm -rf /root/.local/share/pnpm /root/.cache /root/.npm

RUN pnpm build \
 && pnpm ui:build \
 && pnpm qa:lab:build \
 && pnpm openclaw qa suite --help > /dev/null \
 && rm -rf /root/.cache /root/.npm \
 && find /opt/gemmaclaw \( -type d \( ! -perm -a+rx -o -perm /022 \) \
      -o ! -type d \( ! -perm -a+r -o -perm /022 \) \) -exec chmod a+rX,go-w {} +

RUN node node_modules/playwright-core/cli.js install --with-deps chrome \
 && rm -rf /var/lib/apt/lists/* /root/.cache

# GemmaClaw installs a bundled plugin's runtime dependencies with npm when it loads the plugin
# (src/plugins/bundled-runtime-deps.ts). The container has no network, so every bundled plugin's
# dependencies are fetched here into an npm cache owned by qa; the run uses it offline, through a
# per-container copy (gbench mounts an anonymous volume on it).
COPY quality_npm_cache.mjs /opt/gbench/
RUN node /opt/gbench/quality_npm_cache.mjs /opt/gemmaclaw /opt/gbench/npm-cache \
 && rm -rf /root/.npm /root/.cache \
 && chown -R qa:qa /opt/gbench/npm-cache \
 && chmod 0644 /opt/gbench/quality_npm_cache.mjs

RUN install -d -o qa -g qa -m 0700 /opt/gemmaclaw/gbench_qa_out /opt/gemmaclaw/.artifacts \
      /home/qa \
 && install -d -o root -g gbrelay -m 0770 /gbench-status \
 && install -d -m 0755 /opt/gbench /run/gbench-relay /etc/gbench \
 && touch /etc/gbench/openclaw.json \
 && (find qa/scenarios -type f -name '*.md'; \
     find extensions -mindepth 2 -maxdepth 2 -type f \
       \( -name openclaw.plugin.json -o -name package.json \)) \
    | sort | tar -cf /opt/gbench/metadata.tar -T - \
 && printf '%s\n' "${GEMMACLAW_COMMIT}" > /opt/gbench/gemmaclaw_commit

COPY gemmaclaw_patches.json quality_entrypoint.sh quality_relay_forwarder.py /opt/gbench/
# The suite's own first step, as user qa (who cannot write the checkout): it must start without
# building or syncing anything.
RUN set -e; \
    chmod 0644 /opt/gbench/gemmaclaw_patches.json /opt/gbench/quality_relay_forwarder.py \
      /opt/gbench/metadata.tar /opt/gbench/gemmaclaw_commit; \
    chmod 0755 /opt/gbench/quality_entrypoint.sh; \
    install -d -o qa -g qa -m 0700 /tmp/gbench-verify-home; \
    out="$(setpriv --reuid=qa --regid=qa --clear-groups env HOME=/tmp/gbench-verify-home \
           pnpm openclaw qa suite --help 2>&1 > /dev/null)"; \
    rm -rf /tmp/gbench-verify-home; \
    if printf '%s' "$out" | grep -qE 'Building TypeScript|Syncing runtime artifacts'; then \
      printf '%s\n' "$out" >&2; echo "the checkout would rebuild at run time" >&2; exit 1; \
    fi

LABEL gbench.quality.gemmaclaw_commit="${GEMMACLAW_COMMIT}" \
      org.opencontainers.image.source="https://github.com/gemmaclaw/gemmaclaw"

ENTRYPOINT ["/opt/gbench/quality_entrypoint.sh"]
