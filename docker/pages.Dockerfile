# The page service for justelesdocs: server/pages.py under Python.
#
# It cuts one page out of a restricted document per request (server/pages.py has
# the why). The documents are bind-mounted read-only at /restricted by
# docker-compose.yml and never copied in: they are the material this site may
# not redistribute, so the fewer copies of them exist the better, and an image
# that carried them could be pushed to a registry by accident.
#
# The build context is docker/ ONLY, unlike search.Dockerfile: this service
# imports nothing from src/, so there is no reason to hand the daemon the repo
# root. server/pages.py is copied from the parent by the compose file's context,
# which is `..` for the same reason the search image uses it.
#
# pymupdf is the one dependency, and it is the same library scripts/chunk.py
# used to produce the page geometry the highlights are drawn from. Pinned, and
# installed from a wheel: it ships manylinux wheels, so there is no compiler in
# this image and nothing is built at install time.
#
# Written by Claude Code (Opus 5).
FROM python:3.12-slim

# Runs as the host UID (docker-compose.yml sets `user:`), so nothing here may
# depend on being root at runtime. Read-only rootfs: the service writes nothing
# but its log lines, which go to stderr.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# The SAME pin as every PEP 723 header that names pymupdf (the build scripts,
# tests/run.py, server/pages.py itself), which tests/test_pins.py enforces: the
# tests then exercise the version that serves, and `uv run server/pages.py`
# locally runs it too. It used to be 1.24.14 here while everything else floated
# to the latest, so the suite was testing a library production did not run.
# To bump: change every pin together, then run tests/run.py and the build chain.
ARG PYMUPDF_VERSION=1.28.2
RUN pip install --no-cache-dir --no-compile pymupdf==${PYMUPDF_VERSION}

COPY server/pages.py /app/server/pages.py

# Same port as PAGES_PORT's default in server/pages.py and PAGES_UPSTREAM's
# default in the Caddyfile. Never published on the host: Caddy reaches it over
# the compose network by service name.
EXPOSE 8651
ENTRYPOINT ["python", "server/pages.py"]
