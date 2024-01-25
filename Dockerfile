# builder image
FROM python:3.11-slim AS builder
WORKDIR /usr/local
RUN  pip install poetry==1.5.1
COPY ./pyproject.toml ./poetry.lock ./README.md /usr/local/
COPY ./chemtsv2 /usr/local/chemtsv2
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR="/tmp/poetry_cache"
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR
RUN /usr/local/.venv/bin/pip install --no-deps .

# runtime image
FROM python:3.11-slim AS runtime
ENV VENV_PATH="/usr/local/.venv" \
    PATH="/usr/local/.venv/bin:$PATH"
COPY --from=builder $VENV_PATH $VENV_PATH
WORKDIR /app
CMD ["/bin/bash"]

