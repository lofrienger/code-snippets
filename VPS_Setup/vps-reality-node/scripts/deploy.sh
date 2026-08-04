#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage:
  deploy.sh --ssh-host HOST --reality-target HOSTNAME [options]

Required:
  --ssh-host HOST          Root SSH alias or root@host
  --reality-target NAME    TLS 1.3 hostname used as REALITY target and SNI

Options:
  --server-ip ADDRESS      Public IPv4/hostname written to the client profile
  --port PORT              Server TCP port (default: 443)
  --name NAME              Safe ASCII node name (default: VPS-Reality)
  --output-dir DIR         Local secret output directory (default: user config directory)
  --xray-version VERSION   Official release tag, for example v25.7.26 (default: latest)
  --swap-gb N              Create swap only when none exists (default: 1; 0 disables)
  --enable-ufw             Preserve SSH access, allow node port, and enable UFW
  --no-bbr                 Do not configure fq + BBR
  --force                  Back up and replace an existing Xray configuration
  --dry-run                Perform read-only checks and print the plan
  -h, --help               Show this help

The REALITY private key never leaves the VPS. Generated client files contain
credentials and are created with mode 0600; do not commit or paste them.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

ssh_host=
reality_target=
server_ip=
port=443
node_name=VPS-Reality
if [[ -n ${XDG_CONFIG_HOME:-} ]]; then
  output_dir=$XDG_CONFIG_HOME/vps-reality-node
elif [[ -n ${HOME:-} ]]; then
  output_dir=$HOME/.config/vps-reality-node
else
  output_dir=./vps-reality-output
fi
xray_version=latest
swap_gb=1
enable_ufw=0
enable_bbr=1
force=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --ssh-host) [[ $# -ge 2 ]] || die "--ssh-host requires a value"; ssh_host=$2; shift 2 ;;
    --reality-target) [[ $# -ge 2 ]] || die "--reality-target requires a value"; reality_target=$2; shift 2 ;;
    --server-ip) [[ $# -ge 2 ]] || die "--server-ip requires a value"; server_ip=$2; shift 2 ;;
    --port) [[ $# -ge 2 ]] || die "--port requires a value"; port=$2; shift 2 ;;
    --name) [[ $# -ge 2 ]] || die "--name requires a value"; node_name=$2; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || die "--output-dir requires a value"; output_dir=$2; shift 2 ;;
    --xray-version) [[ $# -ge 2 ]] || die "--xray-version requires a value"; xray_version=$2; shift 2 ;;
    --swap-gb) [[ $# -ge 2 ]] || die "--swap-gb requires a value"; swap_gb=$2; shift 2 ;;
    --enable-ufw) enable_ufw=1; shift ;;
    --no-bbr) enable_bbr=0; shift ;;
    --force) force=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ -n $ssh_host ]] || die "--ssh-host is required"
[[ -n $reality_target ]] || die "--reality-target is required"
[[ $ssh_host != -* && $ssh_host =~ ^[A-Za-z0-9._@:-]+$ ]] || die "invalid SSH host or alias"
[[ $reality_target =~ ^([A-Za-z0-9][A-Za-z0-9-]*\.)+[A-Za-z]{2,63}$ ]] || die "REALITY target must be a DNS hostname"
[[ $port =~ ^[0-9]+$ && $port -ge 1 && $port -le 65535 ]] || die "port must be 1-65535"
[[ $swap_gb =~ ^[0-9]+$ && $swap_gb -le 16 ]] || die "swap size must be an integer from 0 to 16"
[[ $node_name =~ ^[A-Za-z0-9._\ -]{1,48}$ ]] || die "node name must be 1-48 safe ASCII characters"
[[ $xray_version == latest || $xray_version =~ ^v[0-9][A-Za-z0-9._-]*$ ]] || die "invalid Xray release tag"
if [[ -n $server_ip ]]; then
  [[ $server_ip =~ ^[A-Za-z0-9.-]+$ ]] || die "server address contains unsupported characters"
fi
command -v ssh >/dev/null 2>&1 || die "ssh is required"
if [[ $dry_run -eq 0 && $force -ne 1 ]]; then
  [[ ! -e $output_dir/flclash-reality.yaml ]] || die "local client YAML already exists; move it or explicitly use --force"
  [[ ! -e $output_dir/vless-share-link.txt ]] || die "local share-link file already exists; move it or explicitly use --force"
fi

echo "Running read-only connection and host checks..."
precheck_file=$(mktemp "${TMPDIR:-/tmp}/vps-reality-precheck.XXXXXX")
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 -- "$ssh_host" "bash -s -- '$port' '$reality_target'" >"$precheck_file" <<'REMOTE_PRECHECK'
set -Eeuo pipefail
port=$1
target=$2
[[ $(id -u) -eq 0 ]] || { echo 'CHECK_ERROR=deployment requires root SSH access'; exit 1; }
[[ -r /etc/os-release ]] || { echo 'CHECK_ERROR=/etc/os-release is missing'; exit 1; }
. /etc/os-release
case ${ID:-} in
  debian|ubuntu)
    :
    ;;
  *)
    echo "CHECK_ERROR=unsupported OS: ${ID:-unknown}"
    exit 1
    ;;
esac
printf 'CHECK_OS=%s %s\n' "$ID" "${VERSION_ID:-unknown}"
printf 'CHECK_ARCH=%s\n' "$(uname -m)"
printf 'CHECK_MEMORY_MB=%s\n' "$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)"
printf 'CHECK_SWAP_MB=%s\n' "$(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo)"
printf 'CHECK_DISK_AVAILABLE_MB=%s\n' "$(df -Pm / | awk 'NR==2 {print $4}')"
printf 'CHECK_SSH_PORTS='
if command -v sshd >/dev/null 2>&1; then
  sshd -T 2>/dev/null | awk '$1=="port" {v=v s $2; s=","} END {print v ? v : "22"}'
else
  printf '22\n'
fi
if command -v ufw >/dev/null 2>&1; then
  printf 'CHECK_UFW=%s\n' "$(ufw status 2>/dev/null | awk 'NR==1 {print tolower($2)}')"
else
  printf 'CHECK_UFW=not-installed\n'
fi
listener=$(ss -H -lntp "sport = :$port" 2>/dev/null || true)
if [[ -n $listener ]]; then
  if grep -qi xray <<<"$listener"; then
    printf 'CHECK_PORT_%s=occupied-by-xray\n' "$port"
  else
    printf 'CHECK_PORT_%s=occupied-by-other\n' "$port"
  fi
else
  printf 'CHECK_PORT_%s=free\n' "$port"
fi
if [[ -e /usr/local/etc/xray/config.json || -e /etc/xray/config.json ]]; then
  printf 'CHECK_XRAY_CONFIG=present\n'
else
  printf 'CHECK_XRAY_CONFIG=absent\n'
fi
printf 'CHECK_TARGET_TLS13='
if timeout 12 openssl s_client -connect "$target:443" -servername "$target" -tls1_3 -verify_return_error </dev/null 2>/dev/null | grep -q 'Verify return code: 0 (ok)'; then
  printf 'ok\n'
else
  printf 'failed\n'
fi
REMOTE_PRECHECK
then
  cat "$precheck_file" >&2
  rm -f -- "$precheck_file"
  die "preflight failed; inspect SSH access and the messages above"
fi
precheck=$(<"$precheck_file")
rm -f -- "$precheck_file"
printf '%s\n' "$precheck"

grep -q '^CHECK_TARGET_TLS13=ok$' <<<"$precheck" || die "REALITY target failed TLS 1.3 certificate validation from the VPS"

if [[ $dry_run -eq 1 ]]; then
  cat <<EOF

DRY RUN: no VPS or local files were changed.
Planned protocol: VLESS + TCP + XTLS Vision + REALITY
Planned listen port: $port/tcp
REALITY target/SNI: $reality_target
Official Xray release: $xray_version with SHA-256 verification
BBR configuration: $enable_bbr
Swap request: ${swap_gb} GiB, only when no swap is active
UFW changes: $enable_ufw
Local output directory: $output_dir
EOF
  exit 0
fi

grep -q "^CHECK_PORT_${port}=occupied-by-other$" <<<"$precheck" && die "TCP port $port belongs to another service; choose another port or stop it explicitly"
if grep -q "^CHECK_PORT_${port}=occupied-by-xray$" <<<"$precheck" && [[ $force -ne 1 ]]; then
  die "TCP port $port is already used by Xray; inspect it or explicitly approve replacement with --force"
fi
if grep -q '^CHECK_XRAY_CONFIG=present$' <<<"$precheck" && [[ $force -ne 1 ]]; then
  die "an Xray config already exists; inspect it or explicitly approve backup/replacement with --force"
fi

echo "Deploying checksum-verified Xray and hardened service..."
deploy_log=$(mktemp "${TMPDIR:-/tmp}/vps-reality-deploy.XXXXXX")
cleanup_local() { rm -f -- "$deploy_log"; }
trap cleanup_local EXIT

ssh -- "$ssh_host" "bash -s -- '$reality_target' '$port' '$xray_version' '$swap_gb' '$enable_ufw' '$enable_bbr' '$force' '$server_ip'" <<'REMOTE_DEPLOY' | tee "$deploy_log" | sed '/^RESULT_.*=/d'
set -Eeuo pipefail
umask 077

target=$1
port=$2
requested_version=$3
swap_gb=$4
enable_ufw=$5
enable_bbr=$6
force=$7
requested_server_ip=$8
config_dir=/usr/local/etc/xray
config_file=$config_dir/config.json
service_file=/etc/systemd/system/xray.service
backup_dir=

if [[ -e $config_file || -e /etc/xray/config.json || -e $service_file ]]; then
  [[ $force -eq 1 ]] || { echo "existing Xray installation found; rerun only with explicit --force approval" >&2; exit 1; }
  backup_dir="/var/backups/xray-reality-$(date -u +%Y%m%dT%H%M%SZ)"
  install -d -m 0700 "$backup_dir"
  for old_file in "$config_file" /etc/xray/config.json "$service_file" /etc/sysctl.d/99-vps-reality-node.conf; do
    if [[ -f $old_file ]]; then
      cp -a --parents "$old_file" "$backup_dir"
    fi
  done
  printf 'Existing Xray files backed up to %s\n' "$backup_dir"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl openssl unzip >/dev/null
if [[ $enable_ufw -eq 1 ]]; then
  apt-get install -y -qq ufw >/dev/null
fi

. /etc/os-release
case ${ID:-} in
  debian|ubuntu)
    :
    ;;
  *)
    echo "unsupported OS: ${ID:-unknown}" >&2
    exit 1
    ;;
esac

case $(uname -m) in
  x86_64|amd64) asset_arch=linux-64 ;;
  aarch64|arm64) asset_arch=linux-arm64-v8a ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

if [[ $requested_version == latest ]]; then
  version=$(curl -fsSL --retry 3 https://api.github.com/repos/XTLS/Xray-core/releases/latest |
    awk -F'"' '/"tag_name"/ && !tag {tag=$4} END {print tag}')
else
  version=$requested_version
fi
[[ $version =~ ^v[0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid release tag received: $version" >&2; exit 1; }

work_dir=$(mktemp -d /tmp/xray-release.XXXXXX)
cleanup_remote() { rm -rf -- "$work_dir"; }
trap cleanup_remote EXIT
asset="Xray-${asset_arch}.zip"
base_url="https://github.com/XTLS/Xray-core/releases/download/${version}"
curl -fL --retry 3 -o "$work_dir/$asset" "$base_url/$asset"
curl -fL --retry 3 -o "$work_dir/$asset.dgst" "$base_url/$asset.dgst"
expected_sha=$(awk -F'= ' '/SHA2-256/ {print tolower($2); exit}' "$work_dir/$asset.dgst" | tr -d '[:space:]')
actual_sha=$(sha256sum "$work_dir/$asset" | awk '{print $1}')
[[ $expected_sha =~ ^[0-9a-f]{64}$ && $actual_sha == "$expected_sha" ]] || {
  echo "official release checksum verification failed" >&2
  exit 1
}
unzip -q "$work_dir/$asset" -d "$work_dir/unpacked"

getent group xray >/dev/null 2>&1 || groupadd --system xray
if ! id xray >/dev/null 2>&1; then
  useradd --system --gid xray --home-dir /var/lib/xray --create-home --shell /usr/sbin/nologin xray
fi
install -m 0755 "$work_dir/unpacked/xray" /usr/local/bin/xray
install -d -m 0755 /usr/local/share/xray
install -m 0644 "$work_dir/unpacked/geoip.dat" /usr/local/share/xray/geoip.dat
install -m 0644 "$work_dir/unpacked/geosite.dat" /usr/local/share/xray/geosite.dat
install -d -o root -g xray -m 0750 "$config_dir"
if ! timeout 20 /usr/local/bin/xray tls ping "$target:443" >/dev/null 2>&1; then
  echo "WARNING: xray tls ping did not succeed; OpenSSL TLS 1.3 certificate validation passed earlier" >&2
fi

uuid=$(/usr/local/bin/xray uuid)
key_output=$(/usr/local/bin/xray x25519)
private_key=$(awk -F': *' '/PrivateKey|Private key/ {print $2; exit}' <<<"$key_output")
public_key=$(awk -F': *' '/Password|PublicKey|Public key/ {print $2; exit}' <<<"$key_output")
short_id=$(openssl rand -hex 8)
[[ -n $uuid && -n $private_key && -n $public_key && $short_id =~ ^[0-9a-f]{16}$ ]] || {
  echo "failed to generate Xray credentials" >&2
  exit 1
}

cat >"$config_file" <<JSON
{
  "log": {"loglevel": "warning"},
  "inbounds": [{
    "tag": "vless-reality-in",
    "listen": "0.0.0.0",
    "port": $port,
    "protocol": "vless",
    "settings": {
      "clients": [{"id": "$uuid", "flow": "xtls-rprx-vision"}],
      "decryption": "none"
    },
    "streamSettings": {
      "network": "tcp",
      "security": "reality",
      "realitySettings": {
        "show": false,
        "dest": "$target:443",
        "xver": 0,
        "serverNames": ["$target"],
        "privateKey": "$private_key",
        "shortIds": ["$short_id"]
      }
    },
    "sniffing": {
      "enabled": true,
      "destOverride": ["http", "tls", "quic"],
      "routeOnly": true
    }
  }],
  "outbounds": [
    {"tag": "direct", "protocol": "freedom"},
    {"tag": "block", "protocol": "blackhole"}
  ],
  "routing": {
    "domainStrategy": "IPIfNonMatch",
    "rules": [
      {"type": "field", "ip": ["geoip:private"], "outboundTag": "block"},
      {"type": "field", "protocol": ["bittorrent"], "outboundTag": "block"}
    ]
  }
}
JSON
chown root:xray "$config_file"
chmod 0640 "$config_file"

cat >"$service_file" <<'SYSTEMD'
[Unit]
Description=Xray VLESS REALITY Service
Documentation=https://github.com/XTLS/Xray-core
After=network-online.target nss-lookup.target
Wants=network-online.target

[Service]
Type=simple
User=xray
Group=xray
ExecStart=/usr/local/bin/xray run -config /usr/local/etc/xray/config.json
Restart=on-failure
RestartSec=5s
LimitNOFILE=1048576
MemoryMax=384M
TasksMax=512
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
PrivateDevices=true
PrivateTmp=true
ProtectClock=true
ProtectControlGroups=true
ProtectHome=true
ProtectHostname=true
ProtectKernelLogs=true
ProtectKernelModules=true
ProtectKernelTunables=true
ProtectSystem=strict
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
RestrictNamespaces=true
RestrictRealtime=true
LockPersonality=true
StateDirectory=xray
UMask=0077

[Install]
WantedBy=multi-user.target
SYSTEMD
chmod 0644 "$service_file"

runuser -u xray -- /usr/local/bin/xray run -test -config "$config_file"
systemctl stop xray 2>/dev/null || true

if [[ $enable_bbr -eq 1 ]]; then
  modprobe tcp_bbr 2>/dev/null || true
  if sysctl -n net.ipv4.tcp_available_congestion_control 2>/dev/null | grep -qw bbr; then
    cat >/etc/sysctl.d/99-vps-reality-node.conf <<'SYSCTL'
net.core.default_qdisc=fq
net.ipv4.tcp_congestion_control=bbr
SYSCTL
    sysctl --system >/dev/null
  else
    echo "WARNING: this kernel does not advertise BBR; leaving congestion control unchanged" >&2
  fi
fi

if [[ $swap_gb -gt 0 ]] && ! swapon --show=NAME --noheadings 2>/dev/null | grep -q .; then
  required_mb=$((swap_gb * 1024 + 512))
  available_mb=$(df -Pm / | awk 'NR==2 {print $4}')
  if [[ $available_mb -ge $required_mb ]]; then
    if [[ ! -e /swapfile-xray ]]; then
      fallocate -l "${swap_gb}G" /swapfile-xray 2>/dev/null || dd if=/dev/zero of=/swapfile-xray bs=1M count=$((swap_gb * 1024)) status=none
      chmod 0600 /swapfile-xray
      mkswap /swapfile-xray >/dev/null
    fi
    swapon /swapfile-xray
    grep -qE '^/swapfile-xray[[:space:]]' /etc/fstab || printf '/swapfile-xray none swap sw 0 0\n' >>/etc/fstab
  else
    echo "WARNING: insufficient free disk for requested swap; swap was not created" >&2
  fi
fi

if [[ $enable_ufw -eq 1 ]]; then
  ufw_was_active=0
  ufw status 2>/dev/null | grep -qi '^Status: active' && ufw_was_active=1
  mapfile -t ssh_ports < <(sshd -T 2>/dev/null | awk '$1=="port" {print $2}' | sort -nu)
  [[ ${#ssh_ports[@]} -gt 0 ]] || ssh_ports=(22)
  for ssh_port in "${ssh_ports[@]}"; do
    ufw limit "${ssh_port}/tcp" comment 'SSH preserved by vps-reality-node' >/dev/null
  done
  ufw allow "${port}/tcp" comment 'Xray VLESS REALITY' >/dev/null
  if [[ $ufw_was_active -eq 0 ]]; then
    ufw default deny incoming >/dev/null
    ufw default allow outgoing >/dev/null
  fi
  ufw --force enable >/dev/null
fi

systemctl daemon-reload
systemctl enable --now xray
systemctl is-active --quiet xray
ss -H -lnt "sport = :$port" | grep -q .

test_socks_port=
for _ in {1..20}; do
  candidate=$((20000 + RANDOM % 20000))
  if [[ $candidate -ne $port ]] && ! ss -H -lnt "sport = :$candidate" 2>/dev/null | grep -q .; then
    test_socks_port=$candidate
    break
  fi
done
[[ -n $test_socks_port ]] || { echo "could not allocate a local port for authenticated testing" >&2; exit 1; }
client_test=/run/xray-reality-client-test.json
client_log=/run/xray-reality-client-test.log
client_pid=
cleanup_client() {
  if [[ -n ${client_pid:-} ]]; then kill "$client_pid" 2>/dev/null || true; fi
  rm -f -- "$client_test" "$client_log"
}
trap 'cleanup_client; cleanup_remote' EXIT
cat >"$client_test" <<JSON
{
  "log": {"loglevel": "warning"},
  "inbounds": [{
    "listen": "127.0.0.1",
    "port": $test_socks_port,
    "protocol": "socks",
    "settings": {"auth": "noauth", "udp": false}
  }],
  "outbounds": [{
    "protocol": "vless",
    "settings": {"vnext": [{
      "address": "127.0.0.1",
      "port": $port,
      "users": [{"id": "$uuid", "encryption": "none", "flow": "xtls-rprx-vision"}]
    }]},
    "streamSettings": {
      "network": "tcp",
      "security": "reality",
      "realitySettings": {
        "fingerprint": "chrome",
        "serverName": "$target",
        "password": "$public_key",
        "shortId": "$short_id"
      }
    }
  }]
}
JSON
chmod 0600 "$client_test"
/usr/local/bin/xray run -config "$client_test" >"$client_log" 2>&1 &
client_pid=$!
for _ in {1..20}; do
  ss -H -lnt "sport = :$test_socks_port" | grep -q . && break
  sleep 0.25
done
ss -H -lnt "sport = :$test_socks_port" | grep -q . || { echo "authenticated client test failed to start" >&2; exit 1; }
test_ip=$(curl -4fsS --max-time 15 --proxy "socks5h://127.0.0.1:$test_socks_port" https://api.ipify.org)
[[ $test_ip =~ ^[0-9a-fA-F:.]+$ ]] || { echo "authenticated proxy test returned an invalid IP" >&2; exit 1; }

if [[ -n $requested_server_ip ]]; then
  resolved_server_ip=$requested_server_ip
else
  resolved_server_ip=$(curl -4fsS --max-time 10 https://api.ipify.org)
fi
[[ $resolved_server_ip =~ ^[A-Za-z0-9.-]+$ ]] || { echo "could not determine a safe public server address" >&2; exit 1; }

printf 'RESULT_VERSION=%s\n' "$version"
printf 'RESULT_SERVER_IP=%s\n' "$resolved_server_ip"
printf 'RESULT_PORT=%s\n' "$port"
printf 'RESULT_UUID=%s\n' "$uuid"
printf 'RESULT_PUBLIC_KEY=%s\n' "$public_key"
printf 'RESULT_SHORT_ID=%s\n' "$short_id"
printf 'RESULT_SNI=%s\n' "$target"
printf 'RESULT_TEST_EGRESS_IP=%s\n' "$test_ip"
printf 'RESULT_BACKUP_DIR=%s\n' "${backup_dir:-none}"
printf 'RESULT_BBR=%s\n' "$(sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null || printf unknown)"
printf 'RESULT_SWAP_MB=%s\n' "$(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo)"
REMOTE_DEPLOY

result_value() {
  local key=$1 value
  value=$(awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$deploy_log")
  [[ -n $value ]] || die "deployment completed without $key; client output was not generated"
  printf '%s' "$value"
}

resolved_server_ip=$(result_value RESULT_SERVER_IP)
resolved_port=$(result_value RESULT_PORT)
uuid=$(result_value RESULT_UUID)
public_key=$(result_value RESULT_PUBLIC_KEY)
short_id=$(result_value RESULT_SHORT_ID)
sni=$(result_value RESULT_SNI)
version=$(result_value RESULT_VERSION)

mkdir -p -- "$output_dir"
chmod 0700 "$output_dir"
yaml_file="$output_dir/flclash-reality.yaml"
link_file="$output_dir/vless-share-link.txt"

cat >"$yaml_file" <<YAML
mixed-port: 7890
allow-lan: false
mode: rule
log-level: info
ipv6: true

dns:
  enable: true
  ipv6: true
  enhanced-mode: fake-ip
  respect-rules: true
  proxy-server-nameserver:
    - https://dns.alidns.com/dns-query
  nameserver-policy:
    "geosite:cn":
      - https://dns.alidns.com/dns-query
    "geosite:geolocation-!cn":
      - "https://1.1.1.1/dns-query#Proxy"
  nameserver:
    - https://dns.alidns.com/dns-query

proxies:
  - name: "$node_name"
    type: vless
    server: "$resolved_server_ip"
    port: $resolved_port
    uuid: "$uuid"
    network: tcp
    tls: true
    udp: true
    flow: xtls-rprx-vision
    servername: "$sni"
    client-fingerprint: chrome
    reality-opts:
      public-key: "$public_key"
      short-id: "$short_id"

proxy-groups:
  - name: "Proxy"
    type: select
    proxies:
      - "$node_name"
      - DIRECT

rules:
  - GEOSITE,private,DIRECT
  - GEOIP,private,DIRECT,no-resolve
  - GEOSITE,cn,DIRECT
  - GEOIP,cn,DIRECT,no-resolve
  - MATCH,Proxy
YAML

uri_name=${node_name// /%20}
printf 'vless://%s@%s:%s?encryption=none&flow=xtls-rprx-vision&security=reality&sni=%s&fp=chrome&pbk=%s&sid=%s&type=tcp#%s\n' \
  "$uuid" "$resolved_server_ip" "$resolved_port" "$sni" "$public_key" "$short_id" "$uri_name" >"$link_file"
chmod 0600 "$yaml_file" "$link_file"

cat <<EOF

Deployment and authenticated proxy test succeeded.
Xray release: $version
FLClash/Mihomo YAML: $yaml_file
Share link: $link_file
These files contain client credentials and are mode 0600. Keep them out of Git.
EOF
