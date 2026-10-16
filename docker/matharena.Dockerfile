# gbench-matharena: MathArena's answer grader for gbench's `hmmt` and `aime` suites, built
# LOCALLY (never pulled).
#
# MathArena (github.com/eth-sri/matharena at b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b) grades
# final answers with sympy and falls back to sympy's LaTeX parser for answers sympify cannot
# read. That parser refuses any antlr runtime other than 4.11, and MathArena's pyproject pins
# antlr4-python3-runtime==4.11; the gbench environment carries 4.13 for math_verify, so the
# grader runs here instead. The image holds only the grader's Python dependencies. gbench
# streams MathArena's source into every run from its byte-for-byte vendored copy
# (gbench/third_party/matharena_src, git blob shas checked), so the image never goes
# stale when that copy is re-pinned.
#
# axiom-axle is installed only because grader.py imports matharena.tools.lean_execution,
# which imports it at module level; the final-answer path never calls it, and gbench runs the
# container with --network none.
#
# Build (context = gbench/docker; nothing is copied from it):
#   docker build -t gbench-matharena:b89f2f0a -f docker/matharena.Dockerfile docker
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Every package pinned, dependencies included (resolved on 2026-09-25 for Python 3.12).
RUN pip install \
        antlr4-python3-runtime==4.11.0 \
        sympy==1.14.0 \
        mpmath==1.3.0 \
        regex==2026.9.10 \
        loguru==0.7.3 \
        PyYAML==6.0.3 \
        axiom-axle==1.8.0 \
        aiohttp==3.14.3 \
        aiohappyeyeballs==2.7.1 \
        aiosignal==1.4.0 \
        attrs==26.1.0 \
        frozenlist==1.8.0 \
        multidict==6.9.1 \
        propcache==0.5.4 \
        yarl==1.25.1 \
        idna==3.20 \
        requests==2.34.2 \
        certifi==2026.7.22 \
        charset-normalizer==3.5.1 \
        urllib3==2.8.0 \
        tenacity==9.1.4 \
        typing_extensions==4.16.0

# Fail the build if sympy's LaTeX parser cannot use the installed antlr runtime.
RUN python -c "from sympy.parsing.latex import parse_latex; print(parse_latex(r'(3+\sqrt{6})^{-1/3}'))"

USER 65534:65534
WORKDIR /tmp
CMD ["python", "-c", "print('gbench-matharena: gbench streams the grader in; see docs/evals/hmmt.md')"]
