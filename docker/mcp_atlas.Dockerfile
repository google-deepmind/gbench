# MCP-Atlas agent environment for gbench's `mcp_atlas` suite, built LOCALLY (never pulled).
#
# This is the release's own services/agent-environment/Dockerfile from scaleapi/mcp-atlas at the
# pinned commit (image version 1.2.7): its MCP servers (gbench runs 35 of the 36, see addition 4)
# behind one HTTP API on port 1984 (/list-tools, /call-tool, /enabled-servers). Every release
# step is kept, in order.
# The additions, each marked "gbench:" below:
#
#   1. The base image is pinned to the exact base of the published 1.2.7 image
#      (uv 0.9.30-python3.12-bookworm-slim; its layers are byte-identical to that image's first
#      five), and the source comes from a clone of the pinned commit instead of a local checkout.
#   2. Unpinned dependencies are resolved as of the published image's build time
#      (RESOLVE_CUTOFF, taken from the 1.2.7 image config: install layer created
#      2026-07-17T19:29:31Z). The release pins every MCP server package, but not what those
#      packages depend on, and uvx/npx resolve some servers when the container starts. Resolved
#      today, the Python servers get mcp 2.x (released 2026-07-28), which removed
#      `mcp.server.fastmcp` and renamed `McpError`: calculator and fetch fail to start, and
#      pubmed imports the removed module. With the cutoff all three start and answer.
#      The cutoff applies to the server installs at build time (UV_EXCLUDE_NEWER,
#      NPM_CONFIG_BEFORE), to the code-executor workspace sync, and to the servers' own
#      uvx/npx resolution at run time (/root/.config/uv/uv.toml, /root/.npmrc; the MCP client
#      starts servers with HOME, so they read these files). UV_NO_CONFIG=1 keeps the agent
#      environment's own `uv run` on its committed uv.lock (the servers do not inherit it).
#   3. A label records the commit, so gbench can refuse an image built from another one.
#   4. brave-search, the one search-engine server of the release whose tools are all search
#      tools, is removed from the server template (mcp_server_template.json), and a label
#      records it. gbench serves its two tools itself, under the same names and argument
#      schemas, with Gemini Google Search grounding; BRAVE_API_KEY is not read, and the Gemini
#      key never enters this image's containers. The release's install step still pre-installs
#      its package; nothing starts it. The other three search-engine servers stay in the
#      template and run as upstream for their page tools: ddg-search (duckduckgo-mcp-server
#      ==0.5.0, no credential) for fetch_content, exa (exa-mcp-server@3.2.1, EXA_API_KEY) for
#      web_fetch_exa, and oxylabs (oxylabs-mcp==0.4.1, OXYLABS_USERNAME / OXYLABS_PASSWORD)
#      for universal_scraper and amazon_product_scraper. gbench serves their search tools with
#      Google Search grounding and sends these servers only the page-tool calls (and exa and
#      oxylabs the release pre-flight's one check call each).
#   5. The release's Node.js install takes NodeSource's 20.x build of the image this file built
#      (20.20.2-1nodesource1, an apt pin; the build fails on another), not the newest 20.x build,
#      and the build records the Debian packages it installed (which float with the Debian
#      archive, as the release's do), Node and npm in /opt/gbench.
#
# A second target, `mongo`, is the release's MongoDB sample data in a local mongod: the dump
# the release ships for the mongodb server (data_exports/mongo_dump_video_game_store-UNZIP-FIRST.zip,
# the `video_game_store` database, 6 collections) restored with mongorestore, as
# data_exports/README.md tells the operator to restore it into their own cluster. gbench runs it
# next to the environment when MONGODB_CONNECTION_STRING is not set.
#
# Build (from the gbench repo root; the build context is not used):
#   docker build -t gbench-mcp-atlas:f24ba3fb -f docker/mcp_atlas.Dockerfile docker
#   docker build --target mongo -t gbench-mcp-atlas-mongo:f24ba3fb -f docker/mcp_atlas.Dockerfile docker
# Run: gbench starts two containers of this image per run itself, each told which servers to
# start by the environment's own ENABLED_SERVERS variable. The environment container runs the 20
# servers that need no credential (among them the model's shell, code and file tools, and
# ddg-search) and gets no credential; the credentialed-servers container runs the 15 that need
# one (exa and oxylabs among them), with their variables, on a docker network of its own, with no
# published port and the API on its loopback (gbench reaches it through docker exec). See
# docs/evals/mcp_atlas.md.
ARG UV_BASE_IMAGE=ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58
ARG MONGO_BASE_IMAGE=mongo:8.0.32@sha256:4968f22d0c6c10ef29952f3e807f62872ba22b3312f25803564fbfc08255efc2

# ---- target `mongo`: the release's MongoDB sample data (gbench addition) --------------------
FROM ${MONGO_BASE_IMAGE} AS mongo
ARG MCP_ATLAS_REF=f24ba3fb0bfa484c86acb28431fad6d7282455f9
ARG MONGO_DUMP_SHA256=83f2919e33d5237f31a5f12362038f4a23e648701d4cfc190f0dd53cfc2f7453
LABEL gbench.mcp_atlas_mongo.revision="${MCP_ATLAS_REF}" \
      gbench.mcp_atlas_mongo.dump_sha256="${MONGO_DUMP_SHA256}"
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# The image declares /data/db a volume, so the restored data lives in /data/atlas-db instead.
RUN curl -fsSL -o /tmp/dump.zip \
      "https://raw.githubusercontent.com/scaleapi/mcp-atlas/${MCP_ATLAS_REF}/data_exports/mongo_dump_video_game_store-UNZIP-FIRST.zip" \
 && echo "${MONGO_DUMP_SHA256}  /tmp/dump.zip" | sha256sum -c - \
 && mkdir -p /tmp/dump /data/atlas-db && unzip -q /tmp/dump.zip -d /tmp/dump \
 && mongod --dbpath /data/atlas-db --bind_ip 127.0.0.1 --fork --logpath /tmp/mongod.log \
 && mongorestore --quiet /tmp/dump/mongo_dump_video_game_store \
 && mongod --dbpath /data/atlas-db --shutdown \
 && chown -R mongodb:mongodb /data/atlas-db \
 && rm -rf /tmp/dump /tmp/dump.zip /tmp/mongod.log
EXPOSE 27017
CMD ["mongod", "--dbpath", "/data/atlas-db", "--bind_ip_all"]

# ---- default target: the agent environment ----------------------------------------------------
FROM ${UV_BASE_IMAGE}

ARG MCP_ATLAS_REF=f24ba3fb0bfa484c86acb28431fad6d7282455f9
ARG RESOLVE_CUTOFF=2026-07-17T19:29:31Z
LABEL gbench.mcp_atlas.revision="${MCP_ATLAS_REF}" \
      gbench.mcp_atlas.resolve_cutoff="${RESOLVE_CUTOFF}" \
      org.opencontainers.image.source="https://github.com/scaleapi/mcp-atlas"

RUN apt-get update && apt-get install -y npm bash git procps curl gettext-base && rm -rf /var/lib/apt/lists/*

# Need to install nodejs from elsewhere, because bookworm-slim is based on debian, and the available version of nodejs is 18, which is too old
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get update
# gbench: the release's `apt-get install -y nodejs` below takes NodeSource's 20.x build of the
# image this file built (NODE_VERSION, an apt pin; NodeSource keeps every build in its
# repository), not whatever 20.x build is newest on the build day.
ARG NODE_VERSION=20.20.2-1nodesource1
RUN printf 'Package: nodejs\nPin: version %s\nPin-Priority: 1001\n' "${NODE_VERSION}" \
      > /etc/apt/preferences.d/gbench-nodejs
RUN apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*
RUN test "$(dpkg-query -W -f '${Version}' nodejs)" = "${NODE_VERSION}"

# gbench: the release source at the pinned commit (the release builds from its own checkout).
RUN git clone https://github.com/scaleapi/mcp-atlas.git /tmp/mcp-atlas \
 && git -C /tmp/mcp-atlas checkout "${MCP_ATLAS_REF}" \
 && test "$(git -C /tmp/mcp-atlas rev-parse HEAD)" = "${MCP_ATLAS_REF}"

# Pre-install all MCP server packages (NPX and UVX), saves approx 1 minute of startup time
# (release: COPY dev_scripts/install_mcp_packages.sh /; gbench: resolved as of RESOLVE_CUTOFF)
RUN cp /tmp/mcp-atlas/services/agent-environment/dev_scripts/install_mcp_packages.sh / \
 && UV_EXCLUDE_NEWER="${RESOLVE_CUTOFF}" NPM_CONFIG_BEFORE="${RESOLVE_CUTOFF}" /install_mcp_packages.sh \
 && rm /install_mcp_packages.sh

# Copy project files and set up workspace
# (release: COPY . . of services/agent-environment, minus what its .dockerignore excludes)
WORKDIR /agent-environment
RUN cp -a /tmp/mcp-atlas/services/agent-environment/. /agent-environment/ \
 && rm -rf __pycache__ *.py[oc] *.egg-info build dist .venv venv tests .pytest_cache .mypy_cache \
           .ruff_cache .coverage htmlcov .vscode .idea *.swp *.swo .git .gitignore .gitattributes \
           .env .env.*
RUN uv venv --python 3.12

# Move /agent-environment/data to /data
RUN mkdir -p /data
RUN mv /agent-environment/data/* /data/ 2>/dev/null || true
RUN rm -rf /agent-environment/data

# Install git submodules from CSV to specified directories
RUN while IFS=',' read -r url sha path; do \
    echo "Installing $url at $sha to $path"; \
    rm -rf "$path" 2>/dev/null || true; \
    git clone "$url" "$path"; \
    cd "$path" && git checkout "$sha"; \
done < /data/repos/git_submodule_info.csv

# Set up MCP code executor workspace (depends on copied repo structure)
# (gbench: its unlocked transitive dependencies resolved as of RESOLVE_CUTOFF)
WORKDIR /data/repos/mcp_code_executor_workspace
RUN UV_EXCLUDE_NEWER="${RESOLVE_CUTOFF}" uv sync
WORKDIR /agent-environment

RUN rm -rf /agent-environment/dev_scripts

# Sync project dependencies (requires pyproject.toml and uv.lock from COPY)
RUN uv sync

# gbench: servers that uvx/npx resolve when the container starts use the same cutoff; the agent
# environment's own `uv run` ignores these files and keeps its committed lock.
RUN mkdir -p /root/.config/uv \
 && printf 'exclude-newer = "%s"\n' "${RESOLVE_CUTOFF}" > /root/.config/uv/uv.toml \
 && printf 'before=%s\n' "${RESOLVE_CUTOFF}" >> /root/.npmrc \
 && rm -rf /tmp/mcp-atlas

# gbench: brave-search, whose tools are all search tools, leaves the server template
# (entrypoint.sh renders the config from it, and mcp_client.py enables only servers it lists);
# gbench serves its tools. ddg-search, exa and oxylabs stay, for their page tools.
ARG REMOVED_SEARCH_SERVERS=brave-search
RUN python3 -c "import json, sys; p = '/agent-environment/src/agent_environment/mcp_server_template.json'; t = json.load(open(p)); drop = sys.argv[1].split(','); assert all(s in t['mcpServers'] for s in drop), drop; [t['mcpServers'].pop(s) for s in drop]; json.dump(t, open(p, 'w'), indent=2)" "${REMOVED_SEARCH_SERVERS}"
LABEL gbench.mcp_atlas.search_servers="removed:${REMOVED_SEARCH_SERVERS}"
ENV UV_NO_CONFIG=1

# gbench: the record of the Debian packages this build installed (they float with the Debian
# archive) and of Node and npm.
RUN mkdir -p /opt/gbench \
 && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt \
 && { node --version; npm --version; } > /opt/gbench/toolchain-versions.txt
LABEL gbench.mcp_atlas.nodejs="${NODE_VERSION}"

EXPOSE 1984

# Set the entrypoint (this always runs to generate mcp_server_config.json from template)
ENTRYPOINT ["/agent-environment/entrypoint.sh"]

# Set the default command (users can override this)
CMD ["uv", "run", "python", "-m", "uvicorn", "agent_environment.main:app", "--host", "0.0.0.0", "--port", "1984"]
