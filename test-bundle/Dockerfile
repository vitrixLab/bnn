FROM gcc:14-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENTRYPOINT ["/workspace/test-bundle/run_nanolm_sandbox_tests.sh"]
