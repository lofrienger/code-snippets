#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: preflight.sh <ssh-host>

Run a read-only inspection of a Debian/Ubuntu VPS. <ssh-host> may be an SSH
config alias or user@host. No configuration files are read or printed.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

ssh_host=$1
if [[ $ssh_host == -* || ! $ssh_host =~ ^[A-Za-z0-9._@:-]+$ ]]; then
  echo "Invalid SSH host or alias: $ssh_host" >&2
  exit 2
fi

ssh -o BatchMode=yes -o ConnectTimeout=10 -- "$ssh_host" 'bash -s' <<'REMOTE'
set -u

section() {
  printf '\n[%s]\n' "$1"
}

section identity
printf 'user='; id -un
printf 'uid='; id -u
printf 'hostname='; hostname
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  printf 'os=%s %s\n' "${ID:-unknown}" "${VERSION_ID:-unknown}"
fi
printf 'kernel='; uname -r
printf 'arch='; uname -m

section resources
free -h 2>/dev/null || true
printf '\nroot filesystem:\n'
df -hP / 2>/dev/null || true
printf '\nswap devices:\n'
swapon --show 2>/dev/null || true

section network
printf 'public_ipv4='
curl -4fsS --max-time 5 https://api.ipify.org 2>/dev/null || printf 'unavailable'
printf '\n'
printf 'tcp_listeners:\n'
ss -H -lntp 2>/dev/null || ss -H -lnt 2>/dev/null || true

section ssh
if command -v sshd >/dev/null 2>&1; then
  printf 'effective_ports='
  sshd -T 2>/dev/null | awk '$1 == "port" {ports = ports sep $2; sep = ","} END {print ports ? ports : "unknown"}'
else
  printf 'effective_ports=unknown\n'
fi

section firewall
if command -v ufw >/dev/null 2>&1; then
  ufw status verbose 2>/dev/null || true
else
  printf 'ufw=not-installed\n'
fi
if command -v nft >/dev/null 2>&1; then
  printf 'nftables_rules='; nft list ruleset 2>/dev/null | wc -l | tr -d ' '
  printf '\n'
fi

section congestion_control
printf 'active='; sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null || printf 'unknown\n'
printf 'available='; sysctl -n net.ipv4.tcp_available_congestion_control 2>/dev/null || printf 'unknown\n'
printf 'qdisc='; sysctl -n net.core.default_qdisc 2>/dev/null || printf 'unknown\n'

section xray
if command -v xray >/dev/null 2>&1; then
  xray version 2>/dev/null | head -n 1
elif [[ -x /usr/local/bin/xray ]]; then
  /usr/local/bin/xray version 2>/dev/null | head -n 1
else
  printf 'binary=not-installed\n'
fi
if command -v systemctl >/dev/null 2>&1; then
  printf 'service_active='; systemctl is-active xray 2>/dev/null || true
  printf 'service_enabled='; systemctl is-enabled xray 2>/dev/null || true
fi

section clock
if command -v timedatectl >/dev/null 2>&1; then
  timedatectl show -p NTPSynchronized -p TimeUSec -p Timezone 2>/dev/null || true
else
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
fi
REMOTE
