FROM python:3.12-slim

# uv binary (pin tag if you want fully reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# copy (not hardlink) into the image, precompile bytecode, use the project venv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependency layer: cached until pyproject.toml / uv.lock change.
# The project is a uv virtual package (source = "."), so sync installs
# only the locked dependencies, not the app itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# App source
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
