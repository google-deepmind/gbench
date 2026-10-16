# gbench-omnidocbench: LOCAL build of OmniDocBench's evaluator for the end2end composite (CDM
# formula + TEDS table + edit-distance text + reading order). gbench builds its images locally
# (it never pulls registry images), so this mirrors OmniDocBench's own repro image / README
# install: Python 3.10 + Ghostscript + TeX Live (with CJK) + ImageMagick 7 + the checkout's deps.
#
# It scores only (no model calls): gbench generates the markdown predictions in the host environment and
# runs this image's pdf_validation.py over them. OmniDocBench pins Python <3.12 + numpy 1.24.4,
# which would break the torch/vLLM serving env - hence the isolated image.
#
# Build (context = an OmniDocBench git checkout at the commit pinned in
# gbench/runners/eval_suites/omnidocbench.py, _EVALUATOR_REVISION; run from the gbench repo root):
#   git -C $GBENCH_PREREQS_DIR/OmniDocBench checkout 193627ae9e97d89188468ed1ee3b7a856ff76044
#   docker build -t gbench-omnidocbench:193627ae \
#       -f docker/omnidocbench.Dockerfile $GBENCH_PREREQS_DIR/OmniDocBench
# The COPY below must include .git: the suite reads the checkout's revision inside the image and
# refuses to score with any other commit.

FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2

# Ghostscript + TeX Live (with CJK for CDM) + latexml (LaTeX tables -> HTML) + runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ghostscript \
        texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
        texlive-lang-chinese texlive-lang-cjk latex-cjk-all \
        latexml \
        wget ca-certificates git fontconfig libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# libfribidi is an ImageMagick 7 runtime dependency the portable AppImage does NOT bundle (its
# `magick` fails with "libfribidi.so.0: cannot open shared object file" without it). Kept in its
# own layer so rebuilding for it does not re-download the TeX Live layer above.
RUN apt-get update && apt-get install -y --no-install-recommends libfribidi0 \
    && rm -rf /var/lib/apt/lists/*

# ImageMagick 7 (CDM requires 7.x; Debian ships 6.x). Portable AppImage, extracted (no FUSE),
# with PDF read/write allowed so CDM can rasterize rendered formula PDFs. The release is pinned
# and checked against the sha256 GitHub publishes for it: 7.1.2-31's clang AppImage, the build
# download.imagemagick.org/archive/binaries/magick served on 2026-09-08 (its extracted tree is the
# one the image built then holds, file for file).
ARG IMAGEMAGICK_VERSION=7.1.2-31
ARG IMAGEMAGICK_SHA256=87790f774f349089f83a138721ec3fffa0602b174a3a6c064d119c870f687c87
RUN wget -q "https://github.com/ImageMagick/ImageMagick/releases/download/${IMAGEMAGICK_VERSION}/ImageMagick-${IMAGEMAGICK_VERSION}-clang-x86_64.AppImage" -O /tmp/magick \
    && echo "${IMAGEMAGICK_SHA256}  /tmp/magick" | sha256sum -c - \
    && chmod +x /tmp/magick \
    && cd /opt && /tmp/magick --appimage-extract >/dev/null 2>&1 \
    && mv /opt/squashfs-root /opt/imagemagick7 && rm -f /tmp/magick \
    && ln -sf /opt/imagemagick7/AppRun /usr/local/bin/magick \
    && ln -sf /opt/imagemagick7/AppRun /usr/local/bin/convert \
    && for p in $(find /opt/imagemagick7 -name policy.xml 2>/dev/null); do \
         sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' "$p" || true; \
       done

# OmniDocBench + its pinned deps (numpy 1.24.4 etc. - isolated here). OmniDocBench pins its direct
# dependencies; their dependencies resolve against the constraints below (PIP_CONSTRAINT, which
# the isolated build environment follows too): the installed set of the image this file built on
# 2026-09-08 (pip freeze), written here because the build context is the OmniDocBench checkout.
COPY . /workspace
WORKDIR /workspace
RUN mkdir -p /opt/gbench && printf '%s\n' \
        'pip==23.0.1' \
        'setuptools==79.0.1' \
        'wheel==0.46.3' \
        'aiohappyeyeballs==2.7.1' \
        'aiohttp==3.14.3' \
        'aiosignal==1.4.0' \
        'anyio==4.15.1' \
        'apted==1.0.3' \
        'async-timeout==5.0.1' \
        'attrs==26.1.0' \
        'beautifulsoup4==4.11.1' \
        'certifi==2026.7.22' \
        'charset-normalizer==3.5.1' \
        'click==8.5.0' \
        'cloudpickle==3.1.2' \
        'contourpy==1.3.2' \
        'cycler==0.12.1' \
        'datasets==5.0.1' \
        'dill==0.4.1' \
        'evaluate==0.4.3' \
        'exceptiongroup==1.3.1' \
        'filelock==3.32.5' \
        'fonttools==4.64.0' \
        'frozenlist==1.8.0' \
        'fsspec==2026.6.0' \
        'func_timeout==4.3.5' \
        'h11==0.16.0' \
        'hf-xet==1.6.0' \
        'httpcore==1.0.9' \
        'httpx==0.28.1' \
        'huggingface_hub==1.30.0' \
        'idna==3.19' \
        'joblib==1.6.0' \
        'kiwisolver==1.5.1' \
        'Levenshtein==0.25.1' \
        'loguru==0.7.2' \
        'lxml==4.9.1' \
        'matplotlib==3.7.5' \
        'multidict==6.7.1' \
        'multiprocess==0.70.19' \
        'nltk==3.9.1' \
        'numpy==1.24.4' \
        'packaging==26.3' \
        'pandas==2.0.3' \
        'pillow==10.4.0' \
        'propcache==0.5.2' \
        'pyarrow==25.0.1' \
        'pylatexenc==2.10' \
        'pyparsing==3.3.2' \
        'python-dateutil==2.9.0.post0' \
        'pytz==2026.3.post1' \
        'PyYAML==6.0.2' \
        'RapidFuzz==3.14.5' \
        'regex==2026.9.3' \
        'requests==2.34.2' \
        'scipy==1.10.1' \
        'six==1.17.0' \
        'soupsieve==2.9.2' \
        'tabulate==0.9.0' \
        'tqdm==4.67.1' \
        'typing_extensions==4.16.0' \
        'tzdata==2026.3' \
        'urllib3==2.7.0' \
        'xxhash==4.0.1' \
        'yarl==1.24.5' \
        > /opt/gbench/omnidocbench.constraints.txt \
    && PIP_CONSTRAINT=/opt/gbench/omnidocbench.constraints.txt pip install --no-cache-dir -e . \
    && python -m pip freeze --exclude-editable | LC_ALL=C sort > /opt/gbench/pip-freeze.txt \
    && grep -v -e '^pip==' -e '^setuptools==' -e '^wheel==' /opt/gbench/omnidocbench.constraints.txt \
        | LC_ALL=C sort | diff - /opt/gbench/pip-freeze.txt
# NLTK data for the text metrics (evaluate / nltk): punkt and punkt_tab, laid out as
# nltk.download('punkt') and nltk.download('punkt_tab') lay them out, from the nltk_data commit
# that last changed them (4f15a3d8, the files nltk.download served on 2026-09-08), each checked
# against its sha256.
ARG NLTK_DATA_COMMIT=4f15a3d89eefe9748ec1c05be495d91289197155
RUN mkdir -p /root/nltk_data/tokenizers && cd /root/nltk_data/tokenizers \
    && wget -q "https://raw.githubusercontent.com/nltk/nltk_data/${NLTK_DATA_COMMIT}/packages/tokenizers/punkt.zip" \
    && wget -q "https://raw.githubusercontent.com/nltk/nltk_data/${NLTK_DATA_COMMIT}/packages/tokenizers/punkt_tab.zip" \
    && printf '%s\n' \
        '51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec  punkt.zip' \
        'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106  punkt_tab.zip' \
        | sha256sum -c - \
    && python -c "import zipfile; [zipfile.ZipFile(z).extractall('.') for z in ('punkt.zip', 'punkt_tab.zip')]" \
    && python -c "import nltk; nltk.data.find('tokenizers/punkt'); nltk.data.find('tokenizers/punkt_tab')"
# The record of the Debian packages this build installed (they float with the archive).
RUN dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt
