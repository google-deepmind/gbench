# gbench-livebench: isolated image for LiveBench's core-category scoring.
#
# LiveBench's base package (latex2sympy2, spacy, litellm, ...) and its coding-execution env
# (code_runner/requirements_eval.txt: tensorflow, numba, opencv, scikit-image, ...) would
# churn a torch/vLLM env, so gbench runs LiveBench's own pipeline INSIDE this image instead
# of installing it into the serving venv. gbench generates nothing here itself - it invokes
# LiveBench's run_livebench.py (inference via --api-base against the gbench-served model, then
# grading) and reads the resulting all_groups.csv.
#
# Reconciled against LiveBench commit 0baa4aa7a78dc0276c0ccf28779dba1113580e18 (LIVEBENCH_COMMIT
# in gbench/runners/eval_suites/livebench.py). Build it from the gbench repository root, with a
# LiveBench checkout at that commit as the build context (not gbench's docker/ directory), and
# tag it with the commit's first 8 characters, the image gbench runs by default
# (LIVEBENCH_IMAGE_DEFAULT; an untagged gbench-livebench, i.e. :latest, is not used):
#   git -C /path/to/LiveBench checkout 0baa4aa7a78dc0276c0ccf28779dba1113580e18
#   git -C /path/to/LiveBench lfs pull
#   docker build -t gbench-livebench:0baa4aa7 \
#       -f docker/livebench.Dockerfile /path/to/LiveBench
#
# The context must carry the checkout's LFS-pulled *.json data and its .git directory: before each
# run gbench reads the commit with `git rev-parse HEAD` in this image, and an image whose commit
# cannot be read (or is not 0baa4aa7) is reported as not leaderboard-comparable.
#
# One local change to the pinned checkout (LIVEBENCH_PATCH in livebench.py, recorded on every
# result as livebench_patch): at 0baa4aa7, livebench/gen_ground_truth_judgment.py line 480 builds
# the old-format instruction_following question list with list(set(...)) over the question dicts,
# which raises TypeError: unhashable type: 'dict', so that category is never graded. The RUN step
# after COPY checks the file's sha256 against the pinned one, replaces that one line with a
# de-duplication by question_id, and checks the patched file's sha256; the build fails if either
# check fails. Before each run gbench checks the patched sha256 in the image (infra_required
# when it differs, with these build steps).
#
# The `agentic_coding` category is intentionally NOT part of this image or the default run
# (it needs the separate ~150GB Multi-SWE-Bench container harness).

FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# System libs commonly needed by the coding-eval stack (opencv, librosa/soundfile, lxml, LaTeX
# tokenizing, git for any VCS installs). Adjust here if a requirements_eval package needs more.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/LiveBench
# Build context must be the LiveBench checkout (with its LFS *.json already pulled).
COPY . /opt/LiveBench

# The recorded instruction_following grading patch (see the header): pinned file in, one line
# replaced, patched file out, each checked by sha256.
RUN echo "dd6bdce67dafd270030980fc864a37940f09a8aa83ce6f94d8657052afbdbd37  livebench/gen_ground_truth_judgment.py" \
        | sha256sum -c - \
    && python -c "import pathlib; \
p = pathlib.Path('livebench/gen_ground_truth_judgment.py'); s = p.read_bytes(); \
old = b'if_questions = list(set([m.question for m in old_instruction_following_matches]))'; \
new = b\"if_questions = list({m.question['question_id']: m.question for m in old_instruction_following_matches}.values())\"; \
assert s.count(old) == 1, 'the line to replace is not in the file exactly once'; \
p.write_bytes(s.replace(old, new))" \
    && echo "19ed4fef82fd839acc0f558380adb4631fdbc5694948575e0c0d319f58ec3a85  livebench/gen_ground_truth_judgment.py" \
        | sha256sum -c -

# LiveBench package + the coding-execution requirements (heavy: tensorflow/numba/opencv/...).
# Their dependencies, which LiveBench leaves mostly unpinned, resolve against the constraints below
# (PIP_CONSTRAINT, for these installs and the spaCy model's): the installed set of the image this
# file built on 2026-09-26 (pip freeze --all), written here because the build context is the
# LiveBench checkout. The last step fails the build if anything else is installed.
RUN mkdir -p /opt/gbench && printf '%s\n' \
        'absl-py==2.5.0' \
        'aiohappyeyeballs==2.7.1' \
        'aiohttp==3.14.3' \
        'aiosignal==1.4.0' \
        'aniso8601==10.0.1' \
        'annotated-doc==0.0.5' \
        'annotated-types==0.8.0' \
        'anthropic==1.8.0' \
        'antlr4-python3-runtime==4.7.2' \
        'anyio==4.15.1' \
        'asgiref==3.12.1' \
        'astunparse==1.6.3' \
        'attrs==26.1.0' \
        'audioread==3.1.0' \
        'beautifulsoup4==4.15.0' \
        'blake3==1.0.9' \
        'blinker==1.9.0' \
        'blis==1.3.3' \
        'boto3==1.43.103' \
        'botocore==1.43.103' \
        'branca==0.8.2' \
        'catalogue==2.0.10' \
        'certifi==2026.7.22' \
        'cffi==2.1.1' \
        'chardet==7.6.0' \
        'charset-normalizer==3.5.1' \
        'click==8.5.0' \
        'cloudpathlib==0.25.0' \
        'cloudpickle==3.1.2' \
        'confection==1.3.3' \
        'contourpy==1.3.3' \
        'cryptography==50.0.1' \
        'cssselect==1.5.0' \
        'cycler==0.12.1' \
        'cyclopts==4.25.3' \
        'cymem==2.0.13' \
        'dataclasses-json==0.6.7' \
        'datasets==5.0.1' \
        'DateTime==6.0' \
        'decorator==5.3.1' \
        'defusedxml==0.7.1' \
        'detect_agent==0.6.0' \
        'dill==0.4.1' \
        'distro==1.9.0' \
        'Django==5.2.17' \
        'dnspython==2.8.0' \
        'docker==7.2.0' \
        'docstring_parser==0.18.0' \
        'docxtpl==0.20.2' \
        'emoji==2.16.0' \
        'en-core-web-sm==3.8.0' \
        'et_xmlfile==2.0.0' \
        'Faker==40.39.0' \
        'fastuuid==0.14.0' \
        'filelock==3.32.7' \
        'Flask-Login==0.6.3' \
        'Flask-Mail==0.10.0' \
        'Flask-RESTful==0.3.10' \
        'Flask-WTF==1.3.0' \
        'Flask==3.1.3' \
        'flatbuffers==25.12.19' \
        'folium==0.20.0' \
        'fonttools==4.66.0' \
        'formulaic==1.2.2' \
        'frozenlist==1.8.0' \
        'fsspec==2026.6.0' \
        'gast==0.7.0' \
        'gensim==4.4.0' \
        'geographiclib==2.1' \
        'geopandas==1.1.4' \
        'geopy==2.5.0' \
        'gitdb==4.0.12' \
        'GitPython==3.1.62' \
        'google-auth==2.58.1' \
        'google-genai==2.25.0' \
        'google-pasta==0.2.0' \
        'grpcio==1.84.0' \
        'h11==0.16.0' \
        'h5py==3.14.0' \
        'hf-xet==1.6.0' \
        'holidays==0.105' \
        'html5lib==1.1' \
        'httpcore2==2.13.1' \
        'httpcore==1.0.9' \
        'httpx2==2.13.1' \
        'httpx==0.28.1' \
        'huggingface_hub==1.33.0' \
        'idna==3.20' \
        'ImageIO==2.37.4' \
        'immutabledict==4.3.1' \
        'importlib_metadata==8.9.0' \
        'iniconfig==2.3.0' \
        'interface_meta==2.0.1' \
        'itsdangerous==2.2.0' \
        'Jinja2==3.1.6' \
        'jiter==0.17.0' \
        'jmespath==1.1.0' \
        'joblib==1.6.0' \
        'jsonschema-specifications==2025.9.1' \
        'jsonschema==4.26.0' \
        'keras==3.15.1' \
        'kiwisolver==1.5.1' \
        'langdetect==1.0.9' \
        'lark==1.3.1' \
        'latex2sympy2==1.9.1' \
        'lazy-loader==0.6' \
        'Levenshtein==0.27.5' \
        'libclang==18.1.1' \
        'librosa==0.11.0' \
        'libtmux==0.62.0' \
        'linkify-it-py==2.2.0' \
        'litellm==1.102.1' \
        'llvmlite==0.49.0' \
        'lxml==6.1.3' \
        'markdown-it-py==4.2.0' \
        'MarkupSafe==3.0.3' \
        'marshmallow==3.26.2' \
        'matplotlib==3.11.2' \
        'mdit-py-plugins==0.6.1' \
        'mdurl==0.1.2' \
        'mechanize==0.4.10' \
        'ml_dtypes==0.6.0' \
        'mpmath==1.3.0' \
        'msgpack==1.2.2' \
        'multidict==6.9.1' \
        'multiprocess==0.70.19' \
        'murmurhash==1.0.15' \
        'mypy_extensions==1.1.0' \
        'namex==0.1.0' \
        'narwhals==2.26.0' \
        'natsort==8.4.0' \
        'networkx==3.6.1' \
        'nltk==3.10.3' \
        'numba==0.67.0' \
        'numpy==2.4.6' \
        'openai==2.54.0' \
        'opencv-python-headless==5.0.0.93' \
        'openpyxl==3.1.5' \
        'opt_einsum==3.4.0' \
        'optree==0.20.0' \
        'outcome==1.3.0.post0' \
        'packaging==26.3' \
        'pandas==2.2.3' \
        'patsy==1.0.3' \
        'pillow==12.3.0' \
        'pip==24.0' \
        'platformdirs==4.12.0' \
        'pluggy==1.6.0' \
        'pooch==1.9.0' \
        'preshed==3.0.13' \
        'prettytable==3.18.0' \
        'prompt_toolkit==3.0.53' \
        'propcache==0.5.4' \
        'protobuf==7.36.2' \
        'psutil==7.2.2' \
        'pyarrow==25.0.1' \
        'pyasn1==0.6.4' \
        'pyasn1_modules==0.4.2' \
        'pycparser==3.0' \
        'pycryptodome==3.23.0' \
        'pydantic-settings==2.15.0' \
        'pydantic==2.13.5' \
        'pydantic_core==2.46.5' \
        'pyfakefs==6.2.0' \
        'PyGithub==2.10.0' \
        'Pygments==2.21.0' \
        'PyJWT==2.15.0' \
        'PyNaCl==1.6.2' \
        'pyogrio==0.13.0' \
        'pyparsing==3.3.3' \
        'pyproj==3.7.2' \
        'pyquery==2.1.0' \
        'PySocks==1.7.1' \
        'pytesseract==0.3.13' \
        'pytest==9.1.1' \
        'python-dateutil==2.9.0.post0' \
        'python-docx==1.2.0' \
        'python-dotenv==1.2.3' \
        'python-http-client==3.3.7' \
        'python-Levenshtein-wheels==0.13.2' \
        'pytz==2026.4' \
        'PyYAML==6.0.3' \
        'RapidFuzz==3.14.6' \
        'referencing==0.37.0' \
        'regex==2026.9.10' \
        'requests-mock==1.12.1' \
        'requests==2.34.2' \
        'rich-argparse==1.8.0' \
        'rich-rst==2.1.0' \
        'rich==14.3.4' \
        'rpds-py==2026.6.3' \
        'rsa==4.9.1' \
        'ruamel.yaml==0.19.1' \
        's3transfer==0.19.2' \
        'scikit-image==0.26.0' \
        'scikit-learn==1.9.1' \
        'scipy==1.17.1' \
        'seaborn==0.13.2' \
        'selenium==4.49.0' \
        'sendgrid==6.12.5' \
        'setuptools==79.0.1' \
        'shapely==2.1.2' \
        'shellingham==1.5.4' \
        'shortuuid==1.0.13' \
        'simple-parsing==0.1.9' \
        'six==1.17.0' \
        'smart_open==8.0.1' \
        'smmap==5.0.3' \
        'sniffio==1.3.1' \
        'sortedcontainers==2.4.0' \
        'soundfile==0.14.0' \
        'soupsieve==2.10' \
        'soxr==1.1.0' \
        'spacy-legacy==3.0.12' \
        'spacy-loggers==1.0.5' \
        'spacy==3.8.16' \
        'sqlparse==0.6.0' \
        'srsly==2.5.3' \
        'statsmodels==0.15.0' \
        'syllapy==0.8.0' \
        'sympy==1.14.0' \
        'tabulate==0.10.0' \
        'tenacity==9.1.4' \
        'tensorflow==2.21.0' \
        'termcolor==3.3.0' \
        'textblob==0.20.1' \
        'texttable==1.7.0' \
        'textual==8.2.8' \
        'thinc==8.3.13' \
        'threadpoolctl==3.7.0' \
        'tifffile==2026.3.3' \
        'tiktoken==0.14.0' \
        'together==2.37.0' \
        'tokenizers==0.23.2' \
        'toml==0.10.2' \
        'tqdm==4.70.1' \
        'trio-websocket==0.12.2' \
        'trio==0.34.0' \
        'truststore==0.10.4' \
        'typer==0.27.2' \
        'types-PyYAML==6.0.12.20260906' \
        'typing-inspect==0.9.0' \
        'typing-inspection==0.4.4' \
        'typing_extensions==4.16.0' \
        'tzdata==2026.4' \
        'unicodedata2==18.0.0' \
        'unidiff==1.0.1' \
        'urllib3==2.8.0' \
        'wasabi==1.1.3' \
        'wcwidth==0.9.1' \
        'weasel==1.0.0' \
        'webencodings==0.6.1' \
        'websocket-client==1.9.2' \
        'websockets==16.1.1' \
        'Werkzeug==3.1.8' \
        'wheel==0.46.3' \
        'wikipedia==1.4.0' \
        'wordcloud==1.9.6' \
        'wordninja==2.0.0' \
        'wrapt==2.4.1' \
        'wsproto==1.3.2' \
        'WTForms==3.2.2' \
        'xlrd==2.0.2' \
        'xlwt==1.3.0' \
        'xmltodict==1.0.4' \
        'xxhash==4.0.1' \
        'xyzservices==2026.9.1' \
        'yarl==1.25.1' \
        'zipp==4.1.0' \
        'zope.interface==8.6' \
        > /opt/gbench/livebench.constraints.txt
RUN PIP_CONSTRAINT=/opt/gbench/livebench.constraints.txt pip install --no-cache-dir -e . \
    && PIP_CONSTRAINT=/opt/gbench/livebench.constraints.txt \
        pip install --no-cache-dir -r livebench/code_runner/requirements_eval.txt

# The NLTK and spaCy data LiveBench's instruction_following grading fetches at run time, baked in
# so grading does not depend on those downloads: punkt, punkt_tab and averaged_perceptron_tagger
# (gen_ground_truth_judgment.py, before the old-format checker), punkt and stopwords
# (if_runner/ifbench/instructions_util.py) and spaCy's en_core_web_sm
# (if_runner/ifbench/instructions.py downloads it at import when it is not installed). The NLTK
# packages come from the nltk_data commit that last changed any of them (98e14261, the files
# nltk.download served on 2026-09-26), each checked against its sha256 and unpacked where
# nltk.download puts it (~/nltk_data/<category>/); a download that fails or differs fails the
# build, and every package and the spaCy model must load.
ARG NLTK_DATA_COMMIT=98e14261eaec2244af1db4ec5b282112444495aa
RUN python -c "import hashlib, os, sys, urllib.request, zipfile; \
pins = {'tokenizers/punkt': '51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec', \
        'tokenizers/punkt_tab': 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106', \
        'taggers/averaged_perceptron_tagger': 'e1f13cf2532daadfd6f3bc481a49859f0b8ea6432ccdcd83e6a49a5f19008de9', \
        'corpora/stopwords': '48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b'}; \
root = os.path.expanduser('~/nltk_data'); \
base = 'https://raw.githubusercontent.com/nltk/nltk_data/' + sys.argv[1] + '/packages/'; \
[os.makedirs(os.path.join(root, os.path.dirname(k)), exist_ok=True) for k in pins]; \
blobs = {k: urllib.request.urlopen(base + k + '.zip', timeout=120).read() for k in pins}; \
bad = [k for k, b in blobs.items() if hashlib.sha256(b).hexdigest() != pins[k]]; \
sys.exit('NLTK data differs from the pinned sha256: ' + ', '.join(bad)) if bad else None; \
[open(os.path.join(root, k + '.zip'), 'wb').write(b) for k, b in blobs.items()]; \
[zipfile.ZipFile(os.path.join(root, k + '.zip')).extractall(os.path.join(root, os.path.dirname(k))) for k in pins]" \
        "${NLTK_DATA_COMMIT}" \
    && python -c "import nltk; [nltk.data.find(p) for p in ('tokenizers/punkt', 'tokenizers/punkt_tab', \
'taggers/averaged_perceptron_tagger', 'corpora/stopwords')]" \
    && PIP_CONSTRAINT=/opt/gbench/livebench.constraints.txt python -m spacy download en_core_web_sm \
    && python -c "import spacy; spacy.load('en_core_web_sm')"

# The installed Python packages are exactly the constrained ones (LiveBench itself, installed from
# the checkout, and the spaCy model, installed from its release URL at 3.8.0, aside); the build
# records them and the Debian packages it installed (those float with the archive).
RUN python -m pip freeze --all --exclude-editable | grep -v ' @ ' | LC_ALL=C sort \
        > /opt/gbench/pip-freeze.txt \
    && grep -v '^en-core-web-sm==' /opt/gbench/livebench.constraints.txt | LC_ALL=C sort \
        | diff - /opt/gbench/pip-freeze.txt \
    && python -c "from importlib.metadata import version; assert version('en_core_web_sm') == '3.8.0'" \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt

# run_livebench.py / show_livebench_result.py resolve package-relative paths from here and
# write their result CSVs (all_tasks.csv / all_groups.csv) to the CWD.
WORKDIR /opt/LiveBench/livebench
