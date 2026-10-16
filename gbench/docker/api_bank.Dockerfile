# googletrans image for gbench's api_bank suite, built LOCALLY (gbench never pulls it).
#
# API-Bank's Translate API (api-bank/apis/translate.py at AlibabaResearch/DAMO-ConvAI f30ccf22)
# calls the synchronous googletrans API of the release its requirements.txt pins,
# googletrans==3.1.0a0, which pins httpx==0.13.3. Installing it next to gbench would downgrade
# httpx and its dependencies, so it lives HERE, with its own pins (api_bank.constraints.txt).
# gbench's process runs upstream's translate.py unchanged and hands each googletrans call to
# api_bank_translate_worker.py in a fresh container of this image, on docker's bridge network (the
# worker reaches Google Translate, translate.googleapis.com, itself or through the proxy
# variables gbench passes on; never the host network), read-only, as an unprivileged user, with
# no key or token in its environment. Nothing contacts a model at build or run time.
#
# Build (context = gbench/docker), from the gbench repository root:
#   docker build -t gbench-api-bank:googletrans-3.1.0a0 -f gbench/docker/api_bank.Dockerfile gbench/docker
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY api_bank.constraints.txt /opt/api_bank/constraints.txt
RUN pip install --no-cache-dir -c /opt/api_bank/constraints.txt googletrans==3.1.0a0 \
    && pip freeze > /opt/api_bank/pip-freeze.txt \
    && python -c "import googletrans, httpx; assert httpx.__version__ == '0.13.3', httpx.__version__"

COPY api_bank_translate_worker.py /opt/api_bank/api_bank_translate_worker.py

LABEL org.gbench.suite="api_bank" \
      org.gbench.api-bank.googletrans="3.1.0a0" \
      org.gbench.api-bank.httpx="0.13.3"

USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "/opt/api_bank/api_bank_translate_worker.py"]
