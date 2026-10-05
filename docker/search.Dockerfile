# The search service for justelesdocs: server/service.mjs under Node.
#
# What it holds is the shipped ranker, src/search.js, run against the index on
# disk instead of in the browser (DESIGN.md, "Ranking on the server"). The image
# is therefore nothing but a Node runtime plus the handful of ES modules the
# service imports; the index itself is bind-mounted read-only at /index by
# docker-compose.yml, never copied in, so a rebuilt index never needs a rebuilt
# image and the image stays a few dozen megabytes.
#
# The build context is the REPO ROOT (the service imports src/search.js, which
# lives outside docker/), which is exactly the case the root .dockerignore was
# written for: it denies everything and re-admits the few paths named below, so
# the 657 MB of documents under data/ can never end up in a build context.
# Anyone adding an import to the ranker, or to a module it imports, has to add
# it to the COPY below AND re-admit it there. Forgetting the COPY does NOT fail
# the build: the image builds and the container dies at start on a missing
# module (it happened when i18n.js began importing store.js).
# tests/search_image.test.mjs rebuilds this file set in a temporary directory
# and imports the ranker from it, so the suite catches it before a deploy.
#
# No npm install: the service has no dependencies beyond node:*, so there is no
# lockfile to audit and nothing to fetch at build time. Pinned to the current
# LTS major; bump deliberately, and re-run tests/run.py under the new one first.
#
# Written by Claude Code (Fable 5.1).
FROM node:22-alpine

# Runs as the host UID (docker-compose.yml sets `user:`), so nothing here may
# depend on being root at runtime. Read-only rootfs, no writable path needed:
# the service writes nothing, not even a log file (stdout only).
ENV NODE_ENV=production
WORKDIR /app
COPY server/ /app/server/
COPY src/search.js src/i18n.js src/paths.js src/store.js /app/src/

# Same port as SEARCH_PORT's default in server/service.mjs and SEARCH_UPSTREAM's
# default in the Caddyfile. Never published on the host: Caddy reaches it over
# the compose network by service name.
EXPOSE 8650
# The heap cap is load-bearing on a small VPS. Every answer parses a few
# per-document JSON files (up to 5 MB each) into short-lived objects, and V8 left
# to itself lets the heap grow past 1 GB of garbage before collecting under
# that churn: measured at 0.9 to 1.4 GB RSS after a hundred questions with no
# cap, 350 to 400 MB with this one, at the same latency (DESIGN.md, "Measured:
# the ranking on the server"). The live set is far smaller: the index is about
# 70 MB and the text cache holds at most SEARCH_TEXT_CACHE documents.
ENTRYPOINT ["node", "--max-old-space-size=256", "server/service.mjs"]
