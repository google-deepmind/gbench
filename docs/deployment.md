# Deployment modes

How gBench runs today (the CLI) and the deployment modes on the roadmap.

[Back to the README](../README.md)

## Deployment modes

### CLI mode (local workstation)
Run directly from terminal for local development or remote endpoint testing.

### Container mode (Docker): roadmap, not shipped
> **Not shipped.** This repository builds no gBench CLI image, so there is no `gbench:latest` to
> run. The Dockerfiles under `docker/` and `gbench/docker/` build per-suite sandboxes and the
> quality pillar's `gbench-quality` image, not the CLI. Install gBench (see
> [Installation](installation.md)) and use the CLI. The image is tracked in
> [section 7 of the roadmap](evals_roadmap.md#7-packaging--operational-parked).

Intended shape:

```bash
docker run --gpus all -v ./results:/results \
  gbench:latest \
  --models google/gemma-4-E4B-it \
  --preset default
```

### Kubernetes mode (`BenchmarkJob` CRD): roadmap, not implemented
> **Not shipped.** There is no `gbench.io/v1` controller in this repository and no CRD is
> installed by any manifest here. The schema below is a design sketch for a future
> operator; applying it today does nothing. Use the CLI.

Intended shape:

```yaml
apiVersion: gbench.io/v1
kind: BenchmarkJob
spec:
  models: ["google/gemma-4-E4B-it"]
  remoteEndpoint: "http://vllm-service.default.svc.cluster.local:8000/v1"
  evaluations: ["gpqa", "mmlu"]
  goldenOnly: true
  output:
    type: gcs
    bucket: gs://gbench-results
```

### Web dashboard and Cloud Run deployment: roadmap, not shipped
> **Not shipped.** This release has no web dashboard, no REST API service, no run launcher and no
> Cloud Run or terraform deployment. gBench is driven from the CLI, and its results are the files
> described in [Output directory structure and result schemas](results.md#output-directory-structure-and-result-schemas).
> A model server that speaks the OpenAI-compatible `/v1` API can still be measured with
> `--remote-endpoint` (on Cloud Run, only a service that allows unauthenticated calls; see
> [Remote Endpoint Mode](remote-endpoints.md#remote-endpoint-mode---remote-endpoint)). The plan for bringing back a
> results viewer and a hosted launcher is
> [section 8 of the roadmap](evals_roadmap.md#8-web-dashboard-and-hosted-deployment-not-shipped).
