# BFCL harness image for gbench's bfcl, bfcl_live and bfcl_v4_agentic suites, built LOCALLY.
#
# Berkeley's own harness, the PyPI release bfcl-eval 2026.3.23 (its wheel is byte-identical to
# bfcl_eval at gorilla commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8), installed with its own
# pins (numpy==1.26.4, filelock==3.20.0, tree_sitter==0.21.3, faiss-cpu==1.11.0, ...) on the
# Python the BFCL README asks for (3.10). Those pins conflict with the main gbench environment, so
# the harness lives HERE. The suites run in this image:
#   * the single-turn checker (eval_runner._evaluate_single_ast_entry /
#     _evaluate_single_relevance_entry, ast_checker) over the replies gbench collected;
#   * the multi-turn categories (OpenAICompletionsHandler's inference loop, the executable
#     backends and the multi-turn checker), talking to the model endpoint;
#   * the agentic categories (`bfcl generate` / `bfcl evaluate` and the memory backends; the
#     web_search tool is served by Gemini Google Search grounding through gbench's search_tool,
#     installed in the `bfcl generate` process at run time, so no bfcl_eval file is changed;
#     google-search-results stays installed only because web_search.py imports it).
# gbench mounts its eval_suites directory read-only and imports only the BFCL modules from it
# (bfcl.py, bfcl_v4_agentic.py and the stdlib-only helpers they import, search_tool.py among
# them). The jobs that talk to the model endpoint (the multi-turn part and `bfcl generate`, which
# also reaches gbench's Gemini relay for the web_search categories) run on docker's bridge
# network, with the host reachable at host.docker.internal, never on the host network; every other
# job runs with `--network none`. Those jobs run the model's calls with eval() in the harness
# process, under gbench's execution guard, so no key enters a container: the endpoint key and the
# Gemini key are added by gbench's relays on the bridge gateway, and the containers hold
# placeholders. Nothing contacts a model at build time.
#
# Build (context = gbench/docker), from the gbench repository root:
#   docker build -t gbench-bfcl:2026.3.23 -f gbench/docker/bfcl.Dockerfile gbench/docker
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The release is downloaded on its own and checked against the PyPI sha256 of its wheel before it
# is installed. bfcl.constraints.txt holds the resolved set of that install (pip freeze of the
# validated build), so a rebuild gets the same versions of the dependencies bfcl-eval leaves open;
# the freeze of each build is kept at /opt/bfcl/pip-freeze.txt.
# soundfile: bfcl_eval/constants/model_config.py imports every model handler, the Qwen one imports
# qwen_agent, and qwen_agent (0.0.34, the release current when bfcl-eval 2026.3.23 came out)
# imports soundfile without declaring it, so a fresh `pip install bfcl-eval` cannot even start
# `bfcl` (gorilla issue #1338). It is installed for that import only; no BFCL code path gbench runs
# uses it.
ARG BFCL_EVAL_VERSION=2026.3.23
ARG BFCL_EVAL_WHEEL_SHA256=3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a
ARG BFCL_EVAL_COMMIT=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
COPY bfcl.constraints.txt /opt/bfcl/constraints.txt
RUN pip download --no-deps --only-binary :all: --dest /opt/bfcl/wheel \
        "bfcl-eval==${BFCL_EVAL_VERSION}" \
    && echo "${BFCL_EVAL_WHEEL_SHA256}  /opt/bfcl/wheel/bfcl_eval-${BFCL_EVAL_VERSION}-py3-none-any.whl" \
        | sha256sum -c - \
    && pip install --no-cache-dir -c /opt/bfcl/constraints.txt \
        "/opt/bfcl/wheel/bfcl_eval-${BFCL_EVAL_VERSION}-py3-none-any.whl" soundfile \
    && pip check \
    && python -c "import importlib.metadata as m, sys; sys.exit(m.version('bfcl-eval') != '${BFCL_EVAL_VERSION}')" \
    && pip freeze > /opt/bfcl/pip-freeze.txt

# memory_vector encodes with SentenceTransformer("all-MiniLM-L6-v2", device="cpu") when it is
# imported (eval_checker/multi_turn_eval/func_source_code/memory_vector.py). The model is fetched
# here at a pinned revision (the Hub's main at the time of pinning) and `main` points at it, so
# the image loads exactly that revision offline at run time.
ARG MINILM_REPO=sentence-transformers/all-MiniLM-L6-v2
ARG MINILM_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ENV HF_HOME=/opt/hf
RUN python -c "import os; from huggingface_hub import snapshot_download; \
repo, rev = os.environ['MINILM_REPO'], os.environ['MINILM_REVISION']; \
path = snapshot_download(repo, revision=rev); assert os.path.basename(path) == rev, path; \
refs = os.path.join(os.environ['HF_HOME'], 'hub', 'models--' + repo.replace('/', '--'), 'refs'); \
os.makedirs(refs, exist_ok=True); open(os.path.join(refs, 'main'), 'w').write(rev); \
print('cached', repo, 'at', path)"
RUN chmod -R a+rwX /opt/hf
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The encoder loads offline and the harness imports.
RUN python -c "from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); print('all-MiniLM-L6-v2', m.encode(['ok']).shape)" \
    && BFCL_PROJECT_ROOT=/tmp/bfcl_probe python -c "import bfcl_eval.__main__, bfcl_eval.eval_checker.eval_runner, \
bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_vector" \
    && rm -rf /tmp/bfcl_probe

LABEL gbench.bfcl.release="${BFCL_EVAL_VERSION}" \
      gbench.bfcl.commit="${BFCL_EVAL_COMMIT}" \
      gbench.bfcl.minilm="${MINILM_REPO}@${MINILM_REVISION}"

# No ENTRYPOINT: gbench runs `python -I -B -u -c <program>` with its eval_suites directory, a
# per-run exchange directory and the harness project root mounted.
