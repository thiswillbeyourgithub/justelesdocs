#!/bin/sh
# Container entrypoint for the justelesdocs static site. Three jobs before
# handing off to Caddy:
#
#   1. Stamp the container start time into STARTED_AT (epoch seconds) so the
#      optional DEV banner can show "restarted X ago".
#
#   2. Fail fast on a half-configured ANALYTICS_URL. If it is set it MUST be
#      reachable AND actually serve JavaScript (the umami tracker, e.g.
#      .../script.js). The common mistake is pointing it at the umami *instance*
#      base URL: reachable, but it returns HTML, so the injected <script> loads
#      no tracker and records zero events while looking configured. Refusing to
#      start beats silently tracking nothing. An empty ANALYTICS_URL disables
#      analytics and skips the check.
#
#   3. Render the runtime app config (window.__APP_CONFIG__) from the environment
#      into /gen/app-config.js, which Caddy serves for /app-config.js. The site
#      root is mounted read-only, so this writes to /gen, a writable tmpfs.
#
# Adapted from ../justelesRCP/docker/entrypoint.sh. The difference is what goes
# into the rendered config: this site has no refresh button, and it does carry
# the search knobs the UI needs (see the keys below).
set -eu

export STARTED_AT="$(date +%s)"

if [ -n "${ANALYTICS_URL:-}" ]; then
  echo "entrypoint: validating ANALYTICS_URL=$ANALYTICS_URL"

  # busybox wget: -S dumps response headers to stderr, body discarded. A non-zero
  # exit means unreachable or an HTTP error (DNS failure, refused, 4xx, 5xx).
  if ! headers="$(wget -S -O /dev/null "$ANALYTICS_URL" 2>&1)"; then
    echo "entrypoint: FATAL ANALYTICS_URL is not reachable:" >&2
    echo "$headers" >&2
    exit 1
  fi

  # It resolved, but it must be JavaScript, not the instance's HTML page. Take the
  # LAST Content-Type, which is the final response after any redirects.
  ctype="$(printf '%s\n' "$headers" | grep -i 'content-type:' | tail -n1 | tr 'A-Z' 'a-z')"
  case "$ctype" in
    *javascript*)
      echo "entrypoint: ANALYTICS_URL OK (reachable, serves JavaScript)"
      ;;
    *)
      echo "entrypoint: FATAL ANALYTICS_URL did not return JavaScript." >&2
      echo "entrypoint: point it at the umami tracker script (e.g. .../script.js)," >&2
      echo "entrypoint: not the instance base URL." >&2
      if [ -n "$ctype" ]; then
        echo "entrypoint: got -> $ctype" >&2
      else
        echo "entrypoint: got -> (no Content-Type header)" >&2
      fi
      exit 1
      ;;
  esac
fi

# Origin (scheme://host[:port]) of ANALYTICS_URL, exported for the
# Content-Security-Policy in docker/Caddyfile (read there as {$ANALYTICS_ORIGIN:}),
# so the tracker script and its beacon are allowed (script-src / connect-src).
# Empty when analytics is disabled, which leaves the CSP with no extra origin.
# BRE sed (no -E) so it works under busybox.
export ANALYTICS_ORIGIN=""
if [ -n "${ANALYTICS_URL:-}" ]; then
  ANALYTICS_ORIGIN="$(printf '%s' "$ANALYTICS_URL" | sed 's#^\([a-zA-Z][a-zA-Z0-9+.-]*://[^/]*\).*#\1#')"
  echo "entrypoint: CSP will allow ANALYTICS_ORIGIN=$ANALYTICS_ORIGIN"
fi

# Render the app config from the environment into the /gen tmpfs. An unset var
# becomes an empty string and the consumers treat empty as "disabled" or "use the
# built-in default", so a bare deploy with no docker/.env still works.
#
# SEARCH_DIM and SEARCH_FLOOR are NOT rendered here any more: ranking happens in
# the search container, which reads both from the same docker/.env and refuses
# to start when SEARCH_DIM disagrees with the index it was given. The page
# learns the floor from every /api/search answer.
cat > /gen/app-config.js <<CONFIG
// Generated from the container environment by docker/entrypoint.sh at startup.
window.__APP_CONFIG__ = {
  url: '${ANALYTICS_URL:-}',
  websiteId: '${ANALYTICS_WEBSITE_ID:-}',
  sri: '${ANALYTICS_SRI:-}',
  dnt: '${ANALYTICS_DNT:-}',
  dev: '${DEV:-}',
  startedAt: '${STARTED_AT:-}',
  sourceUrl: '${SOURCE_URL:-}',
};
CONFIG
echo "entrypoint: rendered /gen/app-config.js (dev='${DEV:-}')"

exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
