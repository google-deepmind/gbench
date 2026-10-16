# gbench-ojbench: isolated image for OJBench's online-judge scoring.
#
# OJBench judges a submission with the official `ojbench` library over the DMOJ sandbox
# (cptbox: ptrace + libseccomp syscall filtering), plus PyPy3 (Python-language solutions) and
# g++ (C++). DMOJ's cptbox does NOT build on Python 3.12 (its bundled Cython C accesses the
# PyLongObject.ob_digit field removed in 3.12), and the gbench serving env is 3.12 -- so the
# judge runs INSIDE this Python 3.11 image instead of the host venv. gbench does the model
# rollout itself (HTTP against the served model) and only hands the generated programs to this
# image for judging.
#
# Build (no build context needed beyond this docker/ dir, which holds the entrypoint, the
# constraints file and the check):
#   docker build -t gbench-ojbench:5e94480b -f docker/ojbench.Dockerfile docker
#
# The 7.85 GB He-Ren/OJBench_testdata is NOT baked; it is bind-mounted read-only at run time.

FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# g++ + build tools (compile C++ solutions and dmoj's cptbox), libseccomp-dev (cptbox sandbox
# header), pypy3 (run Python-language solutions), git (clone OJBench).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git g++ build-essential libseccomp-dev pypy3 \
    && rm -rf /var/lib/apt/lists/*

# Every pip install below, and the judge-server's isolated build environment (Cython), resolves
# against ojbench.constraints.txt (PIP_CONSTRAINT, set for those commands only): the installed set
# of the image this file built on 2026-09-11, so a rebuild installs those versions instead of the
# newest ones. The check at the end fails the build if anything else is installed.
COPY ojbench.constraints.txt pinned_pip_check.py /opt/gbench/

# DMOJ judge-server from the EXACT commit OJBench's README pins. OJBench's judger.py calls
# `problem.cases()` (judger.py:135), an API that exists ONLY on this commit. The PyPI
# `dmoj==4.1.0` that `pip install -e OJBench` would otherwise pull has NO `Problem.cases()`, so
# every executable submission raises AttributeError inside the judge worker and NOTHING gets
# graded - the run silently reports 0% (measured 2026-09-11). On Python 3.11 dmoj's Cython C
# compiles fine (the ob_digit break is 3.12-only). Install this FIRST so it owns the dmoj slot.
RUN git clone https://github.com/DMOJ/judge-server.git /opt/judge-server \
    && git -C /opt/judge-server checkout f098cd3a49a60186d1fadde5132329ec5f4f2213 \
    && PIP_CONSTRAINT=/opt/gbench/ojbench.constraints.txt pip install --no-cache-dir /opt/judge-server

# OJBench itself + its non-dmoj deps, installed with --no-deps so it cannot pull dmoj==4.1.0 back
# over the mandated commit (its requirements.txt otherwise pins it).
# Pinned to upstream main as of 2026-02-28 (ojbench_patch_judger.py's anchor is at judger.py:300).
RUN git clone https://github.com/He-Ren/OJBench /opt/OJBench \
    && git -C /opt/OJBench checkout 5e94480b1e135b98855cf5bc81213c256aff5b17 \
    && PIP_CONSTRAINT=/opt/gbench/ojbench.constraints.txt \
        pip install --no-cache-dir filelock loguru PyYAML tqdm \
    && PIP_CONSTRAINT=/opt/gbench/ojbench.constraints.txt \
        pip install --no-cache-dir --no-deps -e /opt/OJBench

# Guard: the whole point of the pinned commit is Problem.cases(); fail the build loudly if a
# future dep resolution ever clobbers dmoj back to a version without it (else silent 0% returns).
RUN python -c "from dmoj.problem import Problem; assert hasattr(Problem, 'cases'), \
    'dmoj lacks Problem.cases() - wrong judge-server version (need DMOJ commit f098cd3a)'"

# Fix ojbench's judge-loop deadlock: a worker that crashes hard mid-submission never queues a
# result, and the upstream unconditional result_queue.get() then hangs forever. The patch adds a
# bounded get + an all-workers-dead break (asserts it matched, so a future upstream change fails
# the build loudly rather than silently reintroducing the hang).
COPY ojbench_patch_judger.py /ojbench_patch_judger.py
RUN python /ojbench_patch_judger.py

# The installed packages are exactly the pinned ones (the two source installs aside). The build
# records the Python and Debian package versions it installed in /opt/gbench (the Debian ones float
# with the archive: compare /opt/gbench/dpkg-versions.txt across rebuilds).
RUN python /opt/gbench/pinned_pip_check.py /opt/gbench/ojbench.constraints.txt \
        --source dmoj --source ojbench --build-only Cython --freeze /opt/gbench/pip-freeze.txt \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt

COPY ojbench_judge.py /judge.py
ENTRYPOINT ["python", "/judge.py"]
