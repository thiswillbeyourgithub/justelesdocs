# Custom Caddy image for justelesdocs.
#
# Identical in shape to ../justelesRCP/docker/web.Dockerfile, and for the same
# reason: the stock caddy:2-alpine ships no rate-limit module, but /api/search,
# /api/page and /pdf/* need per-IP throttling (see the per_ip_limit snippet in
# docker/Caddyfile). So build Caddy with the caddy-ratelimit plugin via xcaddy,
# then drop the single static binary onto the SAME alpine base the stock image
# uses, so everything else (the entrypoint, the wget healthcheck, the file
# layout, the NET_BIND_SERVICE file capability) is byte-for-byte stock behaviour.
#
# The throttle matters more here than it does in the sibling. There, a flood of
# searches costs that project its own CPU. Here every /api/search costs the
# SHARED encoder one embedding, so an unthrottled flood against this site
# degrades justelesRCP too.
#
# There is NO COPY from the build context, so compose points the context at this
# docker/ directory rather than the repo root. That is not merely tidy: the repo
# root holds data/, 657 MB of documents of which 384 may never leave this
# machine, and a build context is uploaded to the Docker daemon wholesale. The
# root .dockerignore is the second line of defence for the same reason.
#
# The build needs network access to fetch the Go module (github.com plus the Go
# module proxy), so it runs on whatever host does `docker compose up --build`.
# That network access is the fragile part, and it is why deploy.sh does NOT
# rebuild this image on an ordinary deploy (only with --rebuild-web): everything
# this container serves is a bind mount, so the image is only the binary and a
# deploy never needs a fresh one. On a VPS whose Docker bridge advertises IPv6
# without a route, xcaddy dies with
#   go: module github.com/mholt/caddy-ratelimit: Get "https://proxy.golang.org/...":
#   dial tcp [2a00:...]:443: connect: network is unreachable
# If a rebuild there is unavoidable, either make the module proxy reachable
# (add `network: host` under the web service's `build:`) or build on a machine
# with working egress and ship the image (`sudo docker save <SITE_ID>-web |
# ssh VPS 'sudo docker load'`).
ARG CADDY_VERSION=2

FROM caddy:${CADDY_VERSION}-builder-alpine AS builder
# Pinned to a released tag. Unpinned, `go get` resolves to the latest SemVer tag
# at build time, which today is this same v0.1.0 (it is the module's only tag),
# so the pin changes nothing about what is built and everything about whether two
# builds a year apart agree. The module has commits after the tag; taking them
# means naming a commit here on purpose, not drifting onto them by default.
RUN xcaddy build \
	--with github.com/mholt/caddy-ratelimit@v0.1.0

FROM caddy:${CADDY_VERSION}-alpine
# Replace the stock binary with the plugin-enabled one; same path, so the base
# image's ENTRYPOINT/CMD and our entrypoint.sh keep working unchanged.
COPY --from=builder /usr/bin/caddy /usr/bin/caddy
