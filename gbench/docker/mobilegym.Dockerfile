# Build the local MobileGym / MobileGym++ evaluation image (`gbench-mobilegym:5fbd3f34`):
#
#   docker build -t gbench-mobilegym:5fbd3f34 -f gbench/docker/mobilegym.Dockerfile gbench/docker
#
# Upstream pins:
#   Purewhiter/mobilegym @ 5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38
#   OpenMobile2/MobileGym-plusplus @ cdde4ad06ed67176b56964dd329b8cb8e0b0001c
#   mobilegym-data-v0.1.0.tar.gz sha256:933a8499c71450f135eda70f9107af72b56b42181fc62de76ed1aa4c513a603c

# python:3.11-slim (linux/amd64 + linux/arm64 manifest list)
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ARG MOBILEGYM_REF=5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38
ARG MOBILEGYM_PP_REF=cdde4ad06ed67176b56964dd329b8cb8e0b0001c
ARG MOBILEGYM_DATA_URL=https://github.com/Purewhiter/mobilegym/releases/download/data-v0.1.0/mobilegym-data-v0.1.0.tar.gz
ARG MOBILEGYM_DATA_SHA256=0b649cd1c308c9d434717a2ce1da950c934e4cf6c71ceeb6cf8502e3e6f63150
ARG NODE_VERSION=22.14.0
ARG NODE_SHA256=69b09dba5c8dcb05c4e4273a4340db1005abeafe3927efda2bc5b249e80437ec

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fontconfig \
        fonts-liberation \
        fonts-noto-cjk \
        fonts-noto-color-emoji \
        fonts-wqy-microhei \
        git \
        iproute2 \
        iptables \
        procps \
        xz-utils \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22.14.0 LTS from official SHA256-verified archive.
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node.tar.xz \
    && echo "${NODE_SHA256}  /tmp/node.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm -f /tmp/node.tar.xz \
    && node --version && npm --version

# Install pinned Python harness dependencies and Playwright Chromium.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-unifont \
        libasound2t64 \
        libatk-bridge2.0-0t64 \
        libatk1.0-0t64 \
        libatspi2.0-0t64 \
        libcairo2 \
        libcups2t64 \
        libdbus-1-3 \
        libdrm2 \
        libgbm1 \
        libglib2.0-0t64 \
        libgtk-3-0t64 \
        libnspr4 \
        libnss3 \
        libpango-1.0-0 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxkbcommon0 \
        libxrandr2 \
        libxshmfence1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
        "openai==2.24.0" \
        "playwright==1.51.0" \
        "pytest==8.3.5" \
        "pytest-xdist==3.8.0" \
        "requests==2.32.3" \
        "pillow==11.1.0" \
        "tqdm==4.67.1" \
        "opencc-python-reimplemented==0.1.7" \
    && python -m playwright install chromium

# Clone both pinned upstream repositories and strip .git metadata in the same layer.
RUN git init /opt/mobilegym \
    && cd /opt/mobilegym \
    && git remote add origin https://github.com/Purewhiter/mobilegym.git \
    && git fetch --depth 1 origin "${MOBILEGYM_REF}" \
    && git checkout FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${MOBILEGYM_REF}" \
    && rm -rf /opt/mobilegym/.git \
    && git init /opt/mobilegym-pp \
    && cd /opt/mobilegym-pp \
    && git remote add origin https://github.com/OpenMobile2/MobileGym-plusplus.git \
    && git fetch --depth 1 origin "${MOBILEGYM_PP_REF}" \
    && git checkout FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${MOBILEGYM_PP_REF}" \
    && rm -rf /opt/mobilegym-pp/.git

# Align offline Map service worker, Vite middleware, frozen simulator clock, and offline 12306 stops fallback.
RUN rm -rf /opt/mobilegym-pp/apps/Map \
    && cp -rf /opt/mobilegym/apps/Map /opt/mobilegym-pp/apps/Map \
    && cp -f /opt/mobilegym/vite.config.ts /opt/mobilegym-pp/vite.config.ts \
    && cp -f /opt/mobilegym-pp/os/data/simulatorConfig.ts /opt/mobilegym/os/data/simulatorConfig.ts \
    && python -c 'import pathlib; t = "export async function queryTrainStops(\n  trainNo: string,\n  date: string,\n): Promise<StopInfo[]> {"; r = t + "\n  try { const mod = await import(\"../data/catalog/trainCatalog.json\"); const cat: any = (mod as any).default ?? mod; const trains = cat?.trains || {}; for (const [code, tr] of Object.entries<any>(trains)) { if (tr?.trainNo === trainNo || code === trainNo) { if (Array.isArray(tr?.stops) && tr.stops.length > 0) { return tr.stops.map((s: any) => ({ station_name: String(s.station || \"\"), arrive_time: String(s.arr || \"----\"), start_time: String(s.dep || \"----\"), running_time: \"\", station_train_code: code })); } } } } catch {}"; [pathlib.Path(d, "apps/Railway12306/services/railwayApi.ts").write_text(pathlib.Path(d, "apps/Railway12306/services/railwayApi.ts").read_text(encoding="utf-8").replace(t, r, 1), encoding="utf-8") for d in ("/opt/mobilegym", "/opt/mobilegym-pp")]'

# Install npm packages once (skipping duplicate Puppeteer Chrome download), hardlink node_modules, and build Tailwind CSS.
RUN cd /opt/mobilegym \
    && PUPPETEER_SKIP_DOWNLOAD=1 npm ci --no-audit --no-fund \
    && cp -al /opt/mobilegym/node_modules /opt/mobilegym-pp/node_modules \
    && cd /opt/mobilegym && npm run build:web-css \
    && cd /opt/mobilegym-pp && npm run build:web-css \
    && npm cache clean --force \
    && rm -rf /root/.npm

# Download and verify offline mobilegym-data-v0.1.0.tar.gz in a single layer, then symlink into both checkouts.
ARG MOBILEGYM_DATA_SHA256=933a8499c71450f135eda70f9107af72b56b42181fc62de76ed1aa4c513a603c
RUN mkdir -p /opt/mobilegym-data \
    && curl -fsSL "${MOBILEGYM_DATA_URL}" -o /tmp/mobilegym-data.tar.gz \
    && echo "${MOBILEGYM_DATA_SHA256}  /tmp/mobilegym-data.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/mobilegym-data.tar.gz -C /opt/mobilegym-data --strip-components=1 \
    && rm -f /tmp/mobilegym-data.tar.gz \
    && rm -rf /opt/mobilegym/mobilegym-data /opt/mobilegym-pp/mobilegym-data \
    && ln -s /opt/mobilegym-data /opt/mobilegym/mobilegym-data \
    && ln -s /opt/mobilegym-data /opt/mobilegym-pp/mobilegym-data

RUN mkdir -p /opt/gbench \
    && pip freeze > /opt/gbench/pip-freeze.txt \
    && dpkg-query -W > /opt/gbench/dpkg-versions.txt

COPY mobilegym_run.py /opt/gbench/mobilegym_run.py
COPY mobilegym_entrypoint.sh /opt/gbench/mobilegym_entrypoint.sh
RUN chmod +x /opt/gbench/mobilegym_entrypoint.sh

LABEL org.gbench.mobilegym-launcher=2 \
      org.gbench.mobilegym-ref=${MOBILEGYM_REF} \
      org.gbench.mobilegym-pp-ref=${MOBILEGYM_PP_REF} \
      org.gbench.mobilegym-data-sha256=${MOBILEGYM_DATA_SHA256}

WORKDIR /opt/mobilegym
ENTRYPOINT ["/opt/gbench/mobilegym_entrypoint.sh"]

