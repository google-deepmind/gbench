# Docker-backed evals: sandbox & network setup

Several suites run each task inside its own Docker container (and, for compose-based ones,
its own Docker **network**):

- `terminal_bench`: Harbor spins up one Docker Compose environment per trial
- `swe_bench_pro`, `swe_bench_live`, `swe_lancer`, `multi_swe_bench`: one container per
  instance, applying the model's patch and running the repo's tests

At `--sandboxes N` these run up to N containers/networks **concurrently**. That concurrency
is what makes them fast, and it is also what exhausts two finite Docker resources if the
daemon is left on its defaults: the **address pool** and **stale containers**.

## The failure this prevents

On a default daemon you will eventually hit, mid-run:

```
Error response from daemon: all predefined address pools have been fully subnetted
failed to create network <task>__env_default
```

Every affected task then dies with a bare `RuntimeError` and an empty transcript.
`terminal_bench` reports such a trial exception as a harness error (left out of the score,
counted and named, status `completed_with_errors`), never as a score; before that was so, a
`terminal_bench` run at `--sandboxes 32` lost **69 of 89 tasks** this way and reported 6.74% (a
floor, not a measurement). Either way the tasks were not measured.

Docker's default `default-address-pools` yield only 31 networks in total: `172.17.0.0/16`,
`172.18.0.0/16`, `172.19.0.0/16`, `172.20.0.0/14`, `172.24.0.0/14` and `172.28.0.0/14` carved
at /16 (15 networks), and `192.168.0.0/16` carved at /20 (16 networks); the default bridge takes
one of them. One compose network per concurrent trial, plus the default bridge, plus any
leftover containers, overruns that almost immediately.

`gbench --calibrate` prints `Docker networks: N free of M` (the free networks of the daemon's
pools, or of the defaults above when `daemon.json` sets none), and the pre-run guard refuses a
`--sandboxes` above the free networks when `terminal_bench`, `swe_bench_pro` or `swe_lancer`
(the suites that create one network per concurrent task) is selected, printing the
`daemon.json` remedy below; `GBENCH_SKIP_CALIBRATION_GUARD=1` overrides it.

## Recommended daemon configuration

Give Docker a large, **bounded** pool carved into small subnets, in a range you have
confirmed is free on the host. This is a best practice for any box that runs these suites at
concurrency.

```json
{
  "default-address-pools": [
    { "base": "172.28.0.0/14", "size": 24 }
  ]
}
```

`172.28.0.0/14` size 24 = 2^(24-14) = **1024 networks**, far more than any `--sandboxes`
value needs, while leaving the rest of the RFC-1918 space free for other workloads and
manually-created networks.

Apply it (requires root; the restart kills all running containers, so do it while nothing
important is running):

```bash
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
  "default-address-pools": [
    { "base": "172.28.0.0/14", "size": 24 }
  ]
}
EOF
sudo systemctl restart docker
docker system info --format '{{json .DefaultAddressPools}}'   # verify
```

### Pick a range that does not collide with your host

**Do not blindly copy the base above.** Before applying, confirm the range is unused on
*your* host; a pool that overlaps the host's own NIC, a VPN, or a corporate subnet will
break connectivity, not just Docker.

```bash
ip -o -4 addr show | awk '{print $2, $4}'        # host interfaces
docker network ls -q | xargs -r docker network inspect \
    -f '{{.Name}} {{range .IPAM.Config}}{{.Subnet}}{{end}}'   # subnets already in use
ip route | grep -E '172\.(2[89]|3[01])\.'        # anything in 172.28-172.31? (empty = clear)
```

In particular, **`192.168.0.0/16` is a common host/LAN range**; Docker's *default* pool
includes it, so on a box whose own address is `192.168.x.y` the stock daemon can already
hand out a colliding network. The recommended config deliberately omits `192.168/16` for
this reason. If `172.28.0.0/14` is not free on your host, pick another cleanly-unused block
(e.g. a slice of `10.0.0.0/8`, checked the same way) and keep `size: 24`.

## Also: prune before a run

The pool holds live *and* recently-exited containers' addresses. Clear stale state before a
sweep (safe when nothing you need is running):

```bash
docker container prune -f && docker network prune -f
```

Do this **before** the sweep, never during; pruning races networks your own in-flight tasks
are still attaching.

`swe_bench_live` labels every SWE-agent task container `org.gbench.sweagent_batch=<batch>` and
removes them and its `gbench-sweagent-<hex>` batch container on every exit path gbench can catch.
After a SIGKILL, clean up with
`docker rm -f $(docker ps -aq --filter label=org.gbench.sweagent_batch)` and
`docker rm -f $(docker ps -aq --filter name=gbench-sweagent-)`.

## Concurrency without the pool fix

If you cannot edit the daemon (no root), keep `--sandboxes` at or below ~12-16 for these
suites so concurrent networks stay under the ~31 default limit, and prune first. It is
slower, but it avoids the silent 0-score failures above.


## What a rebuild of a local image reproduces

gbench builds every image it runs locally (never pulled), each FROM a base image pinned by
digest. On top of the base, each image pins what it installs as listed below, and the build
fails when a pinned install resolves to anything else. The rest resolves when the image is
built (a build-time float): a rebuild on another day can get newer versions of it. Where the
table says "recorded", the image keeps the installed versions in the file named, so two builds
can be compared (`docker run --rm --entrypoint cat IMAGE FILE`).

| Image (Dockerfile directory) | Pinned, and checked by the build | Build-time floats |
|---|---|---|
| `gbench-mini-swe-agent-v2`, `gbench-swe-bench-pro-v2`, `gbench-wildclawbench`, `gbench-skillsbench` (gbench/docker) | the Ubuntu packages, from the Ubuntu archive snapshot `20260925T120000Z`; the Docker CLI and, where installed, its compose and buildx plugins at exact versions (labels `org.gbench.docker_cli`, `org.gbench.docker_compose`, `org.gbench.docker_buildx`); the Python packages, against `<image>.constraints.txt` (the set of the checked image) or, for skillsbench, the checkout's `uv.lock` (and the deepagents agent BenchFlow installs in each task container, pinned to `deepagents==0.7.23` / `langchain-openai==1.7.0` with `--exclude-newer 2026-10-10T19:31:32Z` by `skillsbench_benchflow_transport.patch`) | none (recorded: `/opt/gbench/dpkg-versions.txt`, and the Python set) |
| `gbench-swebench-v5`, `gbench-sweagent-for-eval`, `gbench-swe-bench-live`, `gbench-swe-lancer-harness` (gbench/docker) | the same Ubuntu snapshot and exact Docker CLI (where installed); the Python packages against their constraints file or `uv.lock` | none (recorded: `/opt/swebench-dpkg-versions.txt`, `/opt/sweagent-for-eval-dpkg-versions.txt`, `/opt/swe-bench-live-dpkg-versions.txt`, `/opt/swe-lancer-dpkg-versions.txt`) |
| `gbench-mswe-agent` (gbench/docker) | the Debian packages, from the Debian archive snapshot `20260925T120000Z`; the Docker CLI at an exact version; the Python packages against `mswe_agent.constraints.txt` | none (recorded: `/opt/mswe-agent-dpkg-versions.txt`) |
| `gbench-gaia2`, `gbench-ui-control-osworld` (gbench/docker) | the Python packages, against `gaia2.constraints.txt` / `ui_control_osworld.constraints.txt` (the set of the checked image) | the Debian packages (recorded: `/opt/gbench/dpkg-versions.txt`) |
| `gbench-mcp-bench` (gbench/docker) | the system interpreter's packages, against `mcp_bench_constraints.txt`, plus the four servers installed editable from the checkout; Node.js (`22.23.3-1nodesource1`), pip, setuptools, wheel and uv (`0.12.19`); the managed CPython builds uv downloads (3.13.15, 3.14.7); each `uv run` server's `uv.lock` | the Ubuntu packages (recorded: `/opt/gbench/dpkg-versions.txt`); the build's npm packages resolve as of `NPM_CONFIG_BEFORE` (`2026-09-26T00:10:00Z`; recorded: `/opt/gbench/toolchain-versions.txt` and the changes `install.sh` makes to the checkout, `/opt/gbench/mcp-bench-checkout.diff`) |
| `gbench-toolbench`, `gbench-ojbench`, `gbench-bigcodebench`, `gbench-omnidocbench`, `gbench-livebench` (docker) | the Python packages, against a constraints or pins file, checked with `docker/pinned_pip_check.py` or a freeze comparison | the Debian and Ubuntu packages (recorded: `/opt/gbench/dpkg-versions.txt`) |
| `aider-benchmark` (docker) | the Python packages against `aider_polyglot.constraints.txt`; Node.js (`20.20.2-1nodesource1`); jest and the exercises' other npm packages at their versions, with their own dependencies resolved as of `NPM_CONFIG_BEFORE` (`2026-09-27T05:30:00Z`) | Rust's stable toolchain (rustup), the Ubuntu and deadsnakes packages (recorded: `/opt/gbench/dpkg-versions.txt`, `/opt/gbench/toolchain-versions.txt`) |
| `gbench-mcp-atlas` (docker) | the release's server packages and their dependencies, resolved as of the published release image's build time (`RESOLVE_CUTOFF`); Node.js (`20.20.2-1nodesource1`, an apt pin) | the Debian packages (recorded: `/opt/gbench/dpkg-versions.txt`); the Debian packages of `gbench-mcp-atlas-mongo` (not recorded) |
| `gbench-multipl-e` (MultiPL-E's own `evaluation/Dockerfile`, which `docker/multipl_e.Dockerfile` fetches and checks) | the recipe's base image (`ubuntu:22.04` by digest) and its javatuples jar (sha256) | upstream's toolchain installs: the Ubuntu packages, NodeSource's current Node.js, typescript, the D, Mono and Dart repositories, the Clojure CLI, numpy (recorded: `/opt/gbench/dpkg-versions.txt`, `pip-freeze.txt`, `npm-global.txt`, `clojure-tools.txt`) |
| `gbench-matharena` (docker) | every Python package, dependencies included | none |
| `gbench-putnam-formal` (docker) | the Lean toolchain and Mathlib, at PutnamBench's `lean-toolchain` and `lake-manifest.json` | the Debian packages `python3` and `ca-certificates` (not recorded) |
| `gbench-tau2` (docker) | the Python packages, from tau2-bench's frozen `uv.lock` | the Debian packages (ca-certificates, git, ripgrep, bubblewrap, socat, nodejs, npm; not recorded) |
| `gbench-api-bank`, `gbench-bfcl`, `gbench-cyberseceval`, `gbench-terminal-bench` (gbench/docker) | the Python packages against their constraints files; terminal_bench's compose and buildx binaries by release and sha256 | the Debian packages terminal_bench installs (`ca-certificates`, `curl`; not recorded) |

The quality pillar's image is built by gbench itself; see [quality.md](../quality.md).

---
For the `/tmp` inode limit (the other resource a big sweep exhausts) and a full pre-sweep checklist, see [large-sweep-setup.md](large-sweep-setup.md).
