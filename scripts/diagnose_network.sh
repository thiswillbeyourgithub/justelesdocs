#!/bin/sh
# Collect everything needed to explain why a container on this host has no
# outbound network, and print it to stdout as one report.
#
# Why this exists. This site and justelesRCP share a VPS, and naming the two
# Compose projects (so they stopped replacing each other's web container) moved
# both stacks off the old `docker_default` bridge onto NEW networks with NEW
# subnets. Immediately after that, justelesRCP's web container could no longer
# reach its analytics host, and docker/entrypoint.sh refuses to start when
# ANALYTICS_URL is unreachable, so the site went down. Everything below is aimed
# at one question: which layer drops the packet.
#
# Run it on the VPS, or pipe it over ssh from a workstation:
#
#   sh scripts/diagnose_network.sh > diagnostics.log 2>&1
#   ssh -p <port> <user>@<host> 'sh -s -- <SITE_ID>' < scripts/diagnose_network.sh > diagnostics.log 2>&1
#
# The optional argument (else $SITE_ID, else justelesdocs) is this site's compose
# SITE_ID from docker/.env, which names its container and its default network.
#
# It needs passwordless `sudo docker`, which deploy.sh already relies on. It does
# not touch any running service. What it creates, it removes: `docker run --rm`
# probe containers that exit immediately, and one throwaway network (see the
# discriminator in section 4, which is the only write in the script).
#
# PRIVACY. The report is meant to be pasted to someone else, so every line goes
# through a redactor before it is printed: public IPv4 and IPv6 addresses become
# placeholders, and the hostname, username and home directory are masked. RFC1918
# and loopback addresses are kept, because the Docker bridge subnets ARE the
# subject and masking them would make the report useless. Read it before sending
# it anyway.
#
# Written by Claude Code.

DOCKER="sudo docker"
SITE="${1:-${SITE_ID:-justelesdocs}}"
# Any public HTTPS URL, fetched from inside the containers to tell a broken TLS
# path (MTU, proxy, CA bundle) from a broken upstream. Override it with one the
# VPS normally talks to, so the probe exercises the same route as real traffic.
TLS_PROBE_URL="${TLS_PROBE_URL:-https://example.com/}"

# ---------------------------------------------------------------- redaction
# Applied to the whole report, including anything a command writes to stderr.
# Kept as one awk program rather than a chain of seds so that the decision about
# which addresses are safe lives in exactly one place.
redact() {
  awk -v home="${HOME:-/nonexistent}" -v user="${USER:-nonexistent}" \
      -v host="$(hostname 2>/dev/null || echo nonexistent)" '
    function is_private(ip,   a) {
      split(ip, a, ".")
      if (a[1] == "10" || a[1] == "127" || a[1] == "0" || a[1] == "255") return 1
      if (a[1] == "192" && a[2] == "168") return 1
      if (a[1] == "169" && a[2] == "254") return 1
      if (a[1] == "172" && a[2] + 0 >= 16 && a[2] + 0 <= 31) return 1
      return 0
    }
    {
      line = $0
      # Identity first: a hostname can contain digits and dots, so masking it
      # before the address pass keeps it from being half-eaten by that pass.
      #
      # Order matters and is not obvious. The hostname often CONTAINS the
      # username, and on this kind of machine it also carries the hardware model
      # (a login "sam" on host "sam-MS-7758"). Masking the username first would
      # leave "<user>-MS-7758" standing, so the most specific string goes first:
      # hostname, then home directory, then bare username.
      if (host != "nonexistent" && length(host) > 2) gsub(host, "<host>", line)
      if (home != "/nonexistent") gsub(home, "<home>", line)
      if (user != "nonexistent" && length(user) > 2) gsub(user, "<user>", line)
      # Global-unicast IPv6 (2000::/3) is always public, so it always goes.
      gsub(/[23][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]:[0-9a-fA-F:][0-9a-fA-F:]+/, "<public-ipv6>", line)
      out = ""
      while (match(line, /[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/)) {
        ip = substr(line, RSTART, RLENGTH)
        out = out substr(line, 1, RSTART - 1) (is_private(ip) ? ip : "<public-ip>")
        line = substr(line, RSTART + RLENGTH)
      }
      print out line
    }
  '
}

section() { printf '\n\n===== %s =====\n' "$1"; }
run()     { printf '\n$ %s\n' "$*"; sh -c "$*" 2>&1 || printf '(exit %s)\n' "$?"; }

# ------------------------------------------------------------------ probes
# One egress attempt, reported as a single OK/FAIL line plus the reason.
#
# Every probe uses wget and never ping: these containers run `cap_drop: ALL`,
# which removes CAP_NET_RAW, so ping fails with EPERM regardless of whether the
# network works. A ping result here would be a false negative and would send the
# next hour in the wrong direction.
probe() {
  label="$1"; network="$2"; url="$3"
  net_arg=""
  [ -n "$network" ] && net_arg="--network $network"
  # shellcheck disable=SC2086
  output="$($DOCKER run --rm $net_arg --entrypoint sh "$IMG" -c \
    "wget -T 8 -q -O /dev/null '$url' 2>&1" 2>&1)"
  if [ $? -eq 0 ]; then
    printf 'OK    %s\n' "$label"
  else
    printf 'FAIL  %s  ->  %s\n' "$label" "$(printf '%s' "$output" | tr '\n' ' ' | cut -c1-160)"
  fi
}

main() {
printf 'justeles host network diagnostics, %s\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')"
printf 'Report is redacted: public addresses, hostname, username and home are masked.\n'

section "1. Versions and daemon"
# Only the network-relevant lines of `docker info`: the rest carries CPU count
# and total memory, which have nothing to do with this and should not travel.
run "$DOCKER --version"
run "$DOCKER compose version"
run "$DOCKER info 2>/dev/null | grep -Ei 'storage driver|cgroup|iptables|ipv6|network|server version'"
run "cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION)='"
run "grep -E '\"(mtu|bip|default-address-pools|base|size|iptables|ip-forward|dns|fixed-cidr|ipv6|userland-proxy)\"' /etc/docker/daemon.json 2>/dev/null || echo '(no /etc/docker/daemon.json, or none of the addressing keys are set)'"

section "2. Docker networks, subnets and members"
# The heart of it: which subnet each network got, and who is attached. The old
# `docker_default` subnet (if the network still exists) is what any hand-written
# firewall rule on this host would have been written against.
run "$DOCKER network ls"
for n in $($DOCKER network ls --format '{{.Name}}' 2>/dev/null); do
  printf '\n--- network %s\n' "$n"
  $DOCKER network inspect "$n" --format \
    'driver={{.Driver}} internal={{.Internal}} subnets={{range .IPAM.Config}}{{.Subnet}} gw={{.Gateway}} {{end}}
options={{.Options}}
members={{range .Containers}}{{.Name}}({{.IPv4Address}}) {{end}}' 2>&1
done

section "3. Containers"
run "$DOCKER ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"
for c in justelesrcp justelesrcp-embed justelesrcp-refresh "$SITE"; do
  if $DOCKER inspect "$c" >/dev/null 2>&1; then
    printf '\n--- %s\n' "$c"
    $DOCKER inspect "$c" --format \
      'state={{.State.Status}} exit={{.State.ExitCode}} restarts={{.RestartCount}}
project={{index .Config.Labels "com.docker.compose.project"}}
networks={{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}' 2>&1
    printf '\nlast 25 log lines:\n'
    $DOCKER logs --tail 25 "$c" 2>&1 | sed 's/^/    /'
  fi
done

section "4. Egress probes"
# Which image to probe with. Reusing one that is already on the host means no
# pull, so this works even when the host's own egress is the broken thing.
IMG="$($DOCKER inspect justelesrcp --format '{{.Config.Image}}' 2>/dev/null)"
[ -z "$IMG" ] && IMG="$($DOCKER images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>' | head -1)"
printf 'probe image: %s\n\n' "${IMG:-NONE FOUND}"

if [ -z "$IMG" ]; then
  printf 'No usable image on this host, skipping the probes.\n'
else
  # Raw IP before DNS, on purpose: if the IP probe passes and the name probe
  # fails, the problem is resolution and not routing, and those two have nothing
  # in common. http://1.1.1.1/ answers without needing a name or a certificate.
  printf 'From the host itself:\n'
  if wget -T 8 -q -O /dev/null http://1.1.1.1/ 2>/dev/null; then
    printf 'OK    host -> http://1.1.1.1/ (raw IP)\n'
  else
    printf 'FAIL  host -> http://1.1.1.1/ (raw IP)\n'
  fi

  printf '\nFrom containers:\n'
  probe "default bridge -> raw IP   " ""                   "http://1.1.1.1/"
  probe "default bridge -> DNS+TLS  " ""                   "${TLS_PROBE_URL}"
  for n in justelesrcp_default justeles-embed "${SITE}_default" docker_default; do
    if $DOCKER network inspect "$n" >/dev/null 2>&1; then
      probe "$n -> raw IP  " "$n" "http://1.1.1.1/"
      probe "$n -> DNS+TLS " "$n" "${TLS_PROBE_URL}"
    fi
  done

  # The shared-encoder route, checked from where it actually has to work. This is
  # a different question from egress and is included because it is the reason the
  # networks were rearranged in the first place.
  if $DOCKER network inspect justeles-embed >/dev/null 2>&1; then
    printf '\nShared encoder route:\n'
    probe "justeles-embed -> justelesrcp-embed:8461/api/sem/health" \
          justeles-embed "http://justelesrcp-embed:8461/api/sem/health"
  fi

  # THE DISCRIMINATOR. When some networks have egress and others do not, there are
  # two candidate explanations and they look identical from the outside: the
  # broken ones are NEW (so whatever installs the forward rules did not run, and a
  # daemon restart fixes it), or the broken ones sit in a SUBNET RANGE the host's
  # firewall does not cover (so the fix is to pin them where it does).
  #
  # One throwaway network separates them, because it is new AND in the range the
  # working networks use. If it has egress, the range is the problem. If it does
  # not, being new is the problem. Without this the report can only report the
  # symptom, which is what happened the first time it ran.
  #
  # This is the one place the script writes anything. The network is created,
  # probed and removed, it carries no containers of its own, and the subnet is
  # picked from 172.16.0.0/16, which Docker's own pool hands out last.
  printf '\nDiscriminator (throwaway network, new but inside 172.16.0.0/12):\n'
  if $DOCKER network inspect diag-probe-net >/dev/null 2>&1; then
    printf 'SKIP  a network called diag-probe-net already exists, not touching it.\n'
  elif $DOCKER network create --subnet 172.16.250.0/24 diag-probe-net >/dev/null 2>&1; then
    probe "diag-probe-net (172.16.250.0/24) -> raw IP" diag-probe-net "http://1.1.1.1/"
    $DOCKER network rm diag-probe-net >/dev/null 2>&1 \
      || printf 'WARN  could not remove diag-probe-net, do it by hand.\n'
  else
    printf 'SKIP  could not create the probe network (172.16.250.0/24 may be taken).\n'
  fi
fi

section "5. Host routing and interfaces"
run "ip -4 route"
run "ip -4 -brief addr"
run "sysctl net.ipv4.ip_forward net.ipv4.conf.all.forwarding 2>/dev/null"
run "cat /etc/resolv.conf | grep -v '^#'"

section "6. Firewall"
# A rule written against the OLD subnet is the single best fit for the symptom:
# a DROP gives a timeout rather than a refusal, which is exactly what the failing
# wget reported. So these are printed in full rather than summarised.
run "iptables -V 2>/dev/null; sudo iptables -V 2>/dev/null"
run "sudo iptables -S FORWARD"
run "sudo iptables -S DOCKER-USER 2>/dev/null || echo '(no DOCKER-USER chain)'"
run "sudo iptables -t nat -S POSTROUTING"
# Docker 28+ moved its own per-bridge ACCEPT rules out of FORWARD and into these
# chains. With a DROP policy on FORWARD, a bridge that has a MASQUERADE rule in
# nat but NO accept rule here is exactly the shape of the bug being chased: the
# container's packets are silently dropped on the way out and the caller sees a
# timeout rather than a refusal. The first report did not dump them, which is why
# it could show the symptom without showing the cause.
run "sudo iptables -S DOCKER-FORWARD 2>/dev/null || echo '(no DOCKER-FORWARD chain, older Docker)'"
run "sudo iptables -S DOCKER-BRIDGE 2>/dev/null || echo '(no DOCKER-BRIDGE chain)'"
run "sudo iptables -S DOCKER-ISOLATION-STAGE-1 2>/dev/null || echo '(no DOCKER-ISOLATION-STAGE-1 chain)'"
run "sudo nft list ruleset 2>/dev/null | head -120 || echo '(nft not present or empty)'"
run "sudo ufw status verbose 2>/dev/null || echo '(ufw not present)'"
run "systemctl is-active firewalld 2>/dev/null || echo '(firewalld not active)'"

section "7. MTU"
# Only relevant if the raw-IP probe passes and the TLS one hangs: a small uplink
# MTU lets the handshake start and then stalls on the first large packet. Listed
# last because it explains a narrow symptom and nothing else.
run "ip -o link show | sed -n 's/.*: \\([a-z0-9@.-]*\\): .*mtu \\([0-9]*\\).*/\\1 \\2/p'"

section "8. Findings"
# The report above is long, and the two failures that actually strand a container
# are both single facts buried in it. So they are decided here rather than left
# to be spotted by eye.
#
# The one that prompted this script: Docker hands each new bridge network a
# subnet from its default pool, 172.17.0.0/16 through 172.31.0.0/16 and then some
# 192.168 /20s. A host running many stacks exhausts that, and a network created
# after the exhaustion can land somewhere that collides with a route the host
# needs, or outside RFC1918 altogether. Renaming the two Compose projects created
# two new networks on a host in exactly that state, which is when the egress
# broke, so this is the first thing to rule in or out.
{
  dsub="$(mktemp)"; hsub="$(mktemp)"
  for n in $($DOCKER network ls --format '{{.Name}}' 2>/dev/null); do
    $DOCKER network inspect "$n" --format \
      '{{range .IPAM.Config}}{{printf "%s %s\n" .Subnet "'"$n"'"}}{{end}}' 2>/dev/null
  done | awk 'NF == 2 && $1 ~ /\//' | sort -u > "$dsub"

  # Routes that are NOT a docker bridge: the host's own uplink and any tunnel.
  ip -4 route 2>/dev/null | grep -vE 'dev (docker0|br-)' \
    | awk '/^[0-9]/ {print $1}' | grep / | sort -u > "$hsub"

  # Fall back to the routing table when the daemon cannot be queried (no
  # passwordless sudo, daemon down). The kernel's view of the bridges is enough
  # for both checks below, and a findings section that goes quiet because one
  # command failed is worse than useless: it reads as "nothing wrong".
  if [ ! -s "$dsub" ]; then
    printf '(could not query the daemon; reading the bridges from the routing table)\n'
    ip -4 route 2>/dev/null | awk '/dev (docker0|br-)/ {for (i = 1; i <= NF; i++) if ($i == "dev") print $1, $(i + 1)}' \
      | awk 'NF == 2 && $1 ~ /\//' | sort -u > "$dsub"
  fi

  printf 'docker bridge subnets in use: %s\n' "$(wc -l < "$dsub" | tr -d ' ')"

  awk '{
    split($1, p, "/"); split(p[1], o, ".")
    private = (o[1] == "10") \
           || (o[1] == "192" && o[2] == "168") \
           || (o[1] == "172" && o[2] + 0 >= 16 && o[2] + 0 <= 31)
    if (!private)
      printf "PROBLEM  network %s has subnet %s, which is NOT RFC1918. Containers on\n         it will try to reach the real owner of that range instead of the\n         internet. Pin it, or set default-address-pools in daemon.json.\n", $2, $1
  }' "$dsub"

  # Real prefix arithmetic, because the cheap version (compare the first two
  # octets) calls 192.168.112.0/20 an overlap of 192.168.3.0/24, which it is not,
  # and a findings section that cries wolf gets ignored on the day it is right.
  #
  # Bit operations are avoided: mawk has no and(), so the network base is derived
  # by integer division instead. A 32-bit address is exact in a double, so this
  # is not an approximation.
  awk '
    function ip2int(s,   o) { split(s, o, "."); return ((o[1] * 256 + o[2]) * 256 + o[3]) * 256 + o[4] }
    function base(ip, bits,   size) { size = 2 ^ (32 - bits); return int(ip / size) * size }
    function last(ip, bits) { return base(ip, bits) + 2 ^ (32 - bits) - 1 }
    NR == FNR {
      split($1, c, "/"); hb[FNR] = base(ip2int(c[1]), c[2]); he[FNR] = last(ip2int(c[1]), c[2])
      hn[FNR] = $1; hcount = FNR; next
    }
    {
      split($1, c, "/"); b = base(ip2int(c[1]), c[2]); e = last(ip2int(c[1]), c[2])
      for (i = 1; i <= hcount; i++)
        if (b <= he[i] && hb[i] <= e)
          printf "PROBLEM  docker network %s (%s) overlaps the host route %s. Traffic that\n         should leave the machine is delivered to the bridge instead.\n", $2, $1, hn[i]
    }' "$hsub" "$dsub"

  fwd="$(cat /proc/sys/net/ipv4/ip_forward 2>/dev/null)"
  [ "$fwd" = 1 ] || printf 'PROBLEM  net.ipv4.ip_forward is %s, not 1. No container can reach anything\n         off this host until it is.\n' "${fwd:-unreadable}"

  if sudo iptables -S FORWARD 2>/dev/null | grep -q '^-P FORWARD DROP'; then
    printf 'NOTE     the FORWARD policy is DROP, so egress depends entirely on the\n         per-network ACCEPT rules Docker installs. A rule written by hand\n         against the OLD subnet stops matching when a network is recreated,\n         and a DROP shows up as a TIMEOUT, which is the reported symptom.\n'
  fi

  rm -f "$dsub" "$hsub"
  printf '\n(If nothing is flagged above, read section 4: an OK on the default bridge with\n a FAIL on a compose network points at the firewall, and an OK on raw IP with a\n FAIL on DNS+TLS points at resolution or MTU rather than at routing.)\n'
} 2>&1

printf '\n\n===== end of report =====\n'
}

main 2>&1 | redact
