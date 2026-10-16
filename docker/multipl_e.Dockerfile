# Source stage of gbench's MultiPL-E evaluator image (`multipl_e` suite), built LOCALLY.
#
# gbench does not ship MultiPL-E's evaluator recipe (MultiPL-E is licensed BSD-3-Clause with a
# Machine Learning Restriction). This image downloads MultiPL-E's own `evaluation/` directory at
# the pinned commit, checks every file against docker/multipl_e.sha256 (and that no other file is
# present), and prints it as a tar stream. That stream is the build context of the evaluator: it
# is built with MultiPL-E's own evaluation/Dockerfile, as upstream's evaluation/Makefile builds it
# (`docker build -t multipl-e-evaluation .` in evaluation/):
#
#   docker build -t gbench-multipl-e-src:3025a531 -f docker/multipl_e.Dockerfile docker
#   docker run --rm gbench-multipl-e-src:3025a531 | docker build -t gbench-multipl-e:3025a531 -
#
# Three things in that Dockerfile are changed:
#   * its `FROM ubuntu:22.04` is pinned to the digest this file's own FROM uses (the base the
#     evaluator image was built on), so a rebuild does not take whatever ubuntu:22.04 is that day;
#   * it downloads javatuples-1.2.jar from repo.mavenlibs.com, which now redirects to an HTML
#     page, and the Java evaluator then fails to compile every program ("error reading
#     /usr/multiple/javatuples-1.2.jar"). The changed line downloads the same artifact from Maven
#     Central and checks its sha256 (the jar every earlier build got);
#   * a last step records what the build installed, in /opt/gbench of the evaluator image: the
#     Ubuntu and third-party apt packages (dpkg-versions.txt), the Python packages
#     (pip-freeze.txt), the global npm packages (npm-global.txt) and the Clojure CLI release
#     (clojure-tools.txt). Upstream's toolchain installs still resolve at build time (NodeSource's
#     current Node.js, typescript, the D, Mono and Dart repositories, the Clojure CLI, numpy);
#     the record says what a given build got.
# The changed Dockerfile is checked against its own sha256 too.
#
# A download that differs from the pinned files stops the first build. Files are extracted with
# the modes a git checkout gives them (644, or 755 when executable), not the archive's 664/775.
FROM ubuntu:22.04@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -yqq && apt-get install -yqq --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY multipl_e.sha256 /multipl_e.sha256
# nuprl/MultiPL-E commit 3025a531 (the commit the suite's chat protocol is read from).
RUN set -eu; \
    ref=3025a531af7450e7df8b96fe0440e9804480bbad; \
    mkdir /src; \
    curl -fsSL "https://codeload.github.com/nuprl/MultiPL-E/tar.gz/${ref}" \
      | tar -xz --no-same-owner --no-same-permissions -C /src --strip-components=2 \
          "MultiPL-E-${ref}/evaluation"; \
    cd /src; \
    sha256sum --strict -c /multipl_e.sha256; \
    test -z "$(find . ! -type d ! -type f)"; \
    find . -type f | sed 's|^\./||' | LC_ALL=C sort > /tmp/have; \
    awk '{print $2}' /multipl_e.sha256 | LC_ALL=C sort > /tmp/want; \
    diff /tmp/want /tmp/have; \
    sed -i \
      -e 's|^FROM ubuntu:22.04$|FROM ubuntu:22.04@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc|' \
      -e 's|wget https://repo.mavenlibs.com/maven/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O /usr/multiple/javatuples-1.2.jar$|wget https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar -O /usr/multiple/javatuples-1.2.jar \&\& echo "2eda5b19d9820e1cc2f69fcd01639a715a673c11f8507e3d1ed593cf765d5e0a  /usr/multiple/javatuples-1.2.jar" \| sha256sum -c -|' \
      Dockerfile; \
    printf '%s\n' '' '# gbench: the record of what this build installed (the toolchains above resolve at build time).' \
      "RUN mkdir -p /opt/gbench && dpkg-query -W -f '\${Package}=\${Version}\n' > /opt/gbench/dpkg-versions.txt && python3 -m pip freeze --all > /opt/gbench/pip-freeze.txt && npm ls -g --depth=0 > /opt/gbench/npm-global.txt && ls /clojure/lib/clojure/libexec > /opt/gbench/clojure-tools.txt" \
      >> Dockerfile; \
    echo "3b6b9332dd449316e0e94da6642b9d83f811333c15c328e06687183262c1be2d  Dockerfile" \
      | sha256sum --strict -c -
WORKDIR /src
ENTRYPOINT ["tar", "-c", "."]
