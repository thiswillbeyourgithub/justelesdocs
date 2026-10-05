#!/bin/sh
# Ask the RUNNING site whether search actually works, from the machine
# that runs it.
#
# Why this exists, and why it is a separate file rather than more lines in
# deploy.sh: the first deploy came up perfectly and answered every query with
# "Le service de recherche est indisponible". Nothing in the build chain could
# have caught that. verify_chunks.py checks the corpus, check_search.mjs checks
# the ranking against the local index, check_ui.mjs checks the DOM and the
# highlight geometry, and all three were green. The thing that was broken lived
# in none of them: whether the deployed Caddy could reach the encoder, a
# property of the host and of how another project bound its port.
#
# Search has since moved out of the browser into a container of its own
# (server/service.mjs, container <SITE_ID>-search), so there are now two
# such layers, and this gate is still the only thing that can see either:
#   (a) whether Caddy reaches the search container, and
#   (b) whether the search container reaches the shared encoder.
# The page itself downloads no vectors any more; it asks the service.
#
# So this gate covers the layers the others cannot see, and it runs where those
# layers exist. deploy.sh pipes it over ssh (`ssh ... "sh -s -- <url>" <
# scripts/smoke_deployed.sh`), so nothing has to be installed or kept in sync on
# the server. It is POSIX sh and curl and nothing else for the same reason.
#
# It is equally valid against the local dev_server.py:
#   sh scripts/smoke_deployed.sh http://127.0.0.1:8649
#
# Exit codes: 0 all good, 1 something is wrong (the output says what).
#
# Written by Claude Code.
set -u

BASE="${1:-http://127.0.0.1:8648}"
# A question with an apostrophe and accents, because a query that survives the
# whole path (shell, JSON, Caddy, encoder) is the only proof the whole path works.
# The default names no corpus; a deploy passes its own corpus's question as $2
# (scenarios.json smoke.question), so the search also has something to find.
QUESTION="${2:-qu'est-ce qu'une recommandation à jour}"
# The compose SITE_ID (docker/.env), which prefixes the container names the
# diagnostic hints below tell you to inspect. Optional: left out, the hints print
# a <SITE_ID> placeholder to substitute by hand.
SITE="${3:-${SITE_ID:-<SITE_ID>}}"
failures=0

say() { printf '%s\n' "$*"; }
ok() { printf '  ok   %s\n' "$*"; }
bad() { printf '  FAIL %s\n' "$*" >&2; failures=$((failures + 1)); }

# Status code only. Judged on the code rather than on curl's exit status so that
# connection refused (000) and an HTTP error are reported the same way.
code() { curl -s -o /dev/null -m 10 -w '%{http_code}' "$1" 2>/dev/null; }
body() { curl -s -m 10 "$1" 2>/dev/null || true; }

printf '\nsmoke test against %s\n' "$BASE"

# --- the static site ------------------------------------------------------
c="$(code "$BASE/index.html")"
[ "$c" = 200 ] && ok "index.html ($c)" || bad "index.html returned $c"
# The second page of the site. A deploy that stages only what it thinks changed,
# or a Caddy root pointing somewhere stale, shows up here as a 404 on the newer
# page while the older one answers perfectly.
c="$(code "$BASE/browse.html")"
[ "$c" = 200 ] && ok "browse.html ($c)" || bad "browse.html returned $c"

c="$(code "$BASE/MANIFEST.tsv")"
[ "$c" = 200 ] && ok "MANIFEST.tsv ($c)" || bad "MANIFEST.tsv returned $c"

c="$(code "$BASE/index/meta.json")"
[ "$c" = 200 ] && ok "the document table is served ($c)" || bad "index/meta.json returned $c"

# The index's own width. Everything else about search is meaningless if the two
# sides disagree about that number, so it is read from the shipped artefact
# rather than assumed.
meta_json="$(body "$BASE/index/meta.json")"
meta_dims="$(printf '%s' "$meta_json" | sed -n 's/.*"dims"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
[ -n "$meta_dims" ] && ok "the index declares $meta_dims dims" || bad "could not read dims from index/meta.json"

# The per-document boxes the viewer draws highlights from. They are still served
# to the browser; the vectors are not.
c="$(code "$BASE/index/doc/0.json")"
[ "$c" = 200 ] && ok "the per-document highlight boxes are served ($c)" \
	|| bad "index/doc/0.json returned $c, so the viewer can draw no highlight"

# The access tier, which is the one thing about a deployed site that a healthy
# build says nothing about. 53 documents may be READ a page at a time and never
# downloaded whole, and that is enforced by stage.py putting them outside the web
# root and by the page service cutting a page per request. Three ways for it to be
# wrong on the server and right here: a deploy that rsynced an older dist/www, a
# Caddy root pointing at a tree staged before the split, or a pages container that
# is up but was never given its mount. All three are invisible until asked.
#
# The document is picked out of the shipped meta.json rather than named here, so
# this survives any change to the corpus. sed over JSON is enough because the
# fields wanted are adjacent in the object and neither holds a quote.
restricted="$(printf '%s' "$meta_json" | tr '{' '\n' \
	| grep '"access":[[:space:]]*"restricted"' | head -1)"
if [ -z "$restricted" ]; then
	say "  ..  no restricted document in the index, so the tier is not exercised"
else
	r_id="$(printf '%s' "$restricted" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
	r_file="$(printf '%s' "$restricted" | sed -n 's/.*"file"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
	c="$(code "$BASE/pdf/$r_file")"
	if [ "$c" = 404 ]; then
		ok "a restricted document is not downloadable (/pdf/ -> $c)"
	else
		bad "/pdf/$r_file returned $c: a restricted document is being served WHOLE."
		say "     Restage (scripts/stage.py), run scripts/check_served.py, redeploy."
	fi
	c="$(code "$BASE/api/page?doc=$r_id&p=1")"
	if [ "$c" = 200 ]; then
		ok "but one page of it is (/api/page -> $c)"
	else
		bad "/api/page?doc=$r_id&p=1 returned $c: the restricted documents cannot be read at all."
		say "     sudo docker logs ${SITE}-pages, and check PAGES_UPSTREAM in docker/.env"
		say "     (the default, pages:8651, is the compose service name)."
	fi
fi

# The vector files, by the names meta.json gives them. They must NOT be
# reachable: they live in dist/index, outside the web root, and only the search
# container reads them. A 200 here means stage.py copied them into dist/www or
# the Caddy root points at the wrong directory, and the site is then handing
# every visitor tens of megabytes it has no use for.
for field in vectors_file chunks_file pages_file sections_file refs_file; do
	name="$(printf '%s' "$meta_json" | sed -n "s/.*\"$field\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" | head -1)"
	[ -z "$name" ] && continue
	c="$(code "$BASE/index/$name")"
	if [ "$c" = 404 ]; then
		ok "the vectors are not served ($name -> 404)"
	else
		bad "index/$name returned $c, but the vectors must not be served"
		say "     They belong in dist/index, outside the web root. A 200 here means"
		say "     stage.py put them in dist/www or the Caddy root is wrong."
	fi
done

# app-config.js is rendered by the entrypoint from the container environment, so
# a non-empty answer also proves /gen is being served and not a stale copy of
# nothing. Its contents are not checked: the width lives in meta.json and in the
# search container, and the page never sees it.
cfg="$(body "$BASE/app-config.js")"
if [ -n "$cfg" ]; then
	ok "app-config.js is served and non-empty"
else
	bad "app-config.js is empty or missing, so /gen is not being served"
fi

# The notes and the version stamp travel together out of stage.py, so a
# disagreement between them means half a deploy arrived: a site stamped with a
# new version whose notes are the previous one's, which is the exact failure the
# compiler refuses locally. Worth re-checking here because rsync, not the
# compiler, is what puts them on the server.
ver_js="$(body "$BASE/app-version.js")"
site_version="$(printf '%s' "$ver_js" | sed -n 's/.*__APP_VERSION__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
notes="$(body "$BASE/changelog.json")"
notes_version="$(printf '%s' "$notes" | sed -n 's/.*"current"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
if [ -z "$site_version" ]; then
	bad "app-version.js names no version, so the release-notes popup can never open"
elif [ -z "$notes_version" ]; then
	bad "changelog.json is missing or names no current version (the site says $site_version)"
elif [ "$site_version" = "$notes_version" ]; then
	ok "the site and its release notes agree on the version ($site_version)"
else
	bad "the site is stamped $site_version but changelog.json says $notes_version:"
	say "     half a deploy, so returning readers would be shown the wrong notes."
fi

# --- the search service, through the site's own proxy ---------------------
# Through the proxy on purpose. Reaching the search container directly would
# prove the service is up while saying nothing about the hop that actually
# breaks: Caddy's route to it from inside its own container.
health_code="$(code "$BASE/api/search/health")"
if [ "$health_code" = 200 ]; then
	ok "the search service answers /api/search/health through the proxy (200)"
else
	bad "/api/search/health returned $health_code"
fi

health="$(body "$BASE/api/search/health")"
svc_dims="$(printf '%s' "$health" | sed -n 's/.*"dims"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
n_chunks="$(printf '%s' "$health" | sed -n 's/.*"n_chunks"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
rss_mb="$(printf '%s' "$health" | sed -n 's/.*"rss_mb"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
if [ -z "$svc_dims" ]; then
	[ "$health_code" = 200 ] && bad "/api/search/health answered 200 but named no dims"
elif [ -n "$meta_dims" ] && [ "$svc_dims" != "$meta_dims" ]; then
	bad "the search container serves an index of $svc_dims dims but the page's"
	say "     meta.json says $meta_dims: the two halves of dist/ came from different builds."
else
	ok "the service holds ${n_chunks:-?} chunks at $svc_dims dims, rss ${rss_mb:-?} MB"
fi

# The query itself. This is the one request the site cannot answer on its own,
# and the only check here that exercises every hop at once: Caddy, the search
# container, and from there the shared encoder.
# The question is a JSON string, so a backslash or a double quote in a question
# passed as $2 must be escaped first: interpolated raw, `say "no"` would send
# invalid JSON and the 400 would read as a broken service. Backslashes first, or
# the escaping of quotes would itself be escaped. Control characters are not
# handled: a question on one shell argument has none worth supporting.
question_json="$(printf '%s' "$QUESTION" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"
answer="$(curl -s -m 30 -w '\n%{http_code}' \
	-H 'Content-Type: application/json' \
	--data "$(printf '{"q":"%s","filters":{},"bm25":false}' "$question_json")" \
	"$BASE/api/search" 2>/dev/null || true)"
search_code="$(printf '%s' "$answer" | tail -1)"
search_body="$(printf '%s' "$answer" | sed '$d')"

if [ "$search_code" = 200 ]; then
	ok "/api/search answered the question (200)"
	case "$search_body" in
		*'"doc":'*)
			# Only the FIRST result. sed's .* is greedy, so matching against the
			# whole body would report the LAST hit while calling it the best one:
			# cut the body down to one result object before reading any field.
			first="$(printf '%s' "$search_body" | sed -n 's/.*"results"[[:space:]]*:[[:space:]]*\[[[:space:]]*{\([^}]*\)}.*/\1/p')"
			doc="$(printf '%s' "$first" | sed -n 's/.*"doc"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
			page="$(printf '%s' "$first" | sed -n 's/.*"page"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
			cos="$(printf '%s' "$first" | sed -n 's/.*"cosine"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p')"
			ok "the best hit is document $doc page $page at cosine $cos"
			;;
		*)
			bad "/api/search returned 200 with no results for a question the corpus answers."
			say "     Either the index is not the one this corpus was built from, or"
			say "     SEARCH_FLOOR in docker/.env is set far too high."
			;;
	esac
	ans_dims="$(printf '%s' "$search_body" | sed -n 's/.*"dims"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)"
	if [ -n "$meta_dims" ] && [ -n "$ans_dims" ] && [ "$ans_dims" != "$meta_dims" ]; then
		bad "the answer is stamped $ans_dims dims but the page's meta.json says $meta_dims:"
		say "     the two halves of dist/ came from different builds."
	fi
elif [ "$search_code" = 502 ]; then
	bad "/api/search returned 502: the search container cannot reach the encoder."
	say ""
	say "     This is the failure the site reports as 'Le service de recherche est"
	say "     indisponible', and it is a property of the host, not of the site."
	say "     On this machine, in this order:"
	say ""
	say "       sudo docker ps --filter name=justelesrcp-embed"
	say "       sudo docker network inspect justeles-embed \\"
	say "         --format '{{range .Containers}}{{.Name}} {{end}}'"
	say "       sudo docker exec ${SITE}-search wget -qO- http://justelesrcp-embed:8461/api/sem/health"
	say ""
	say "     The encoder is reached by CONTAINER NAME over the shared justeles-embed"
	say "     network. Nothing is published on the host, so a curl from the host tells"
	say "     you nothing about what a container can reach."
	say ""
	say "     First one empty -> the encoder is not running. Deploy justelesRCP."
	say "     Second one missing either justelesrcp-embed or ${SITE}-search"
	say "       -> that container is not on the shared network. Re-deploy the project"
	say "       it belongs to; both compose files attach their container to it, and"
	say "       both deploy.sh create the network."
	say "     Third one failing while the first two are fine -> EMBED_URL in"
	say "       docker/.env names something other than the encoder's container."
elif [ "$search_code" = 503 ]; then
	bad "/api/search returned 503: the service's queue is full."
	say ""
	say "     The site says 'too many searches at once' for this. On an idle server"
	say "     that means something is hammering it, so look at who:"
	say ""
	say "       sudo docker logs --tail 50 ${SITE}-search"
else
	bad "/api/search returned $search_code: Caddy cannot reach the search container,"
	say "     or the container is not up. A refused connection (000), a 404 and a 502"
	say "     raised by Caddy rather than by the service cannot be told apart by the"
	say "     code alone, so check all three:"
	say ""
	say "       sudo docker ps --filter name=${SITE}-search"
	say "       sudo docker logs --tail 20 ${SITE}-search"
	say ""
	say "     A service that refuses to start prints"
	say "       index is N dims, SEARCH_DIM asks for M"
	say "     which is fixed by SEARCH_DIM in docker/.env or by rebuilding the index."
	say "     If the container is up and healthy, check SEARCH_UPSTREAM in docker/.env"
	say "     (default search:8650): that is the name Caddy proxies to."
fi

# The encoder used to be reachable through this site, back when the browser did
# the embedding. It must not be any more: the search container talks to it over
# the private network, and a path from the public site onto justelesRCP's
# service is a way to spend that project's capacity from here.
c="$(code "$BASE/api/sem/embed")"
if [ "$c" = 404 ]; then
	ok "the encoder is not exposed through this site any more (/api/sem/embed -> 404)"
else
	bad "/api/sem/embed returned $c: the Caddyfile still proxies /api/sem/embed,"
	say "     and this site must not open a path onto justelesRCP's service."
fi

# The cross-encoder resort was removed on 2026-10-01 after it exhausted the VPS's
# memory. A 200 or a 5xx here means a stale Caddyfile or a stale search image is
# still serving it, and a stale rerank container may be eating that memory again.
c="$(code "$BASE/api/search/rerank")"
if [ "$c" = 404 ]; then
	ok "the removed cross-encoder resort is gone (/api/search/rerank -> 404)"
else
	bad "/api/search/rerank returned $c: a stale deploy still routes the removed resort."
	say "     Check docker/Caddyfile and: sudo docker ps --filter name=${SITE}-rerank"
fi

if [ "$failures" -eq 0 ]; then
	printf '\nsmoke test passed: the deployed site can search.\n'
	exit 0
fi
printf '\n%s check(s) failed.\n' "$failures" >&2
exit 1
