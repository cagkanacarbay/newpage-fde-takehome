FROM python:3.12-slim-bookworm AS builder

COPY --from=astral/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable
RUN uv run --no-sync python -c "from live_long_rnd.retrieve import prepare_flashrank_model; prepare_flashrank_model()"


FROM node:22-bookworm-slim AS web-builder

WORKDIR /web
RUN corepack enable && corepack prepare pnpm@10.17.1 --activate
COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY web ./
ENV NEXT_PUBLIC_API_BASE_URL=.
RUN pnpm build


FROM python:3.12-slim-bookworm AS runtime

RUN useradd --system --uid 10001 --create-home app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/data/models/flashrank /app/data/models/flashrank
COPY --from=index --chown=app:app / /app/data/index/
COPY --from=corpus --chown=app:app / /app/data/corpus/longevity/
COPY --from=web-builder --chown=app:app /web/out /app/web

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LIVE_LONG_RETRIEVER=lancedb \
    LIVE_LONG_LLM=openai \
    LIVE_LONG_WEB_DIR=/app/web \
    LIVE_LONG_CORPUS_DIR=/app/data/corpus/longevity

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs', timeout=2)"]

CMD ["uvicorn", "live_long_rnd.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
