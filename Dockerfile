# AuditGym: offline verification image. Build once (needs network), then run with --network none.
FROM python:3.11.15-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv
ENV PYTHONTZPATH="" PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=0
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY auditgym ./auditgym
RUN uv sync --frozen --group test --extra gym && rm -rf /root/.cache
COPY tests ./tests
COPY configs ./configs
COPY data ./data
COPY schema ./schema
COPY examples ./examples
RUN useradd --create-home auditor && mkdir -p /app/runs && chown -R auditor /app/runs
USER auditor
ENV PATH="/app/.venv/bin:$PATH" AUDITGYM_LAUNCHER="docker run --rm -i --network none auditgym:1.0.0"
ENTRYPOINT ["auditgym"]
CMD ["check"]
