---
name: vps-reality-node
description: Deploy, harden, verify, and troubleshoot a self-hosted VPS proxy node for FLClash or Mihomo. Use for Ubuntu or Debian VPS setup with VLESS, TCP, XTLS Vision, and REALITY; secure Xray installation; BBR, swap, systemd, or firewall tuning; China-direct routing; China Telecom connectivity; client YAML generation; and diagnosing an existing REALITY node.
---

# VPS Reality Node

Deploy a minimal Xray VLESS + TCP + XTLS Vision + REALITY node and generate a FLClash/Mihomo configuration that sends mainland China traffic directly. Prefer this TCP/443 profile when the client is on China Telecom or UDP quality is unknown.

## Safety rules

- Confirm that the user owns or administers the VPS before changing it.
- Treat SSH passwords, UUIDs, short IDs, share links, and client YAML as secrets. Never commit generated client files.
- Never print or copy the REALITY private key off the VPS. Only the public key belongs in client output.
- Back up an existing Xray configuration before replacing it. Refuse replacement unless the user explicitly approves it.
- Preserve the effective SSH port before enabling a firewall. Never reset an existing firewall.
- Use official Xray releases and verify the release SHA-256 digest. Do not pipe third-party installation scripts into a shell.
- Run read-only inspection first. Describe mutations and get confirmation before deploying to a live VPS.

## Choose the transport

Use the following default unless evidence points elsewhere:

- Use **VLESS + TCP + XTLS Vision + REALITY on 443** for China Telecom, uncertain UDP paths, no domain, or a low-memory VPS.
- Consider **NaiveProxy** when the user controls a real domain and certificate and prioritizes conventional browser-like HTTPS behavior.
- Consider **Hysteria2** only after testing that UDP is stable and not rate-limited on the actual client network.

Read [references/design-and-troubleshooting.md](references/design-and-troubleshooting.md) before changing the default, choosing a REALITY target, or diagnosing a failed node.

## Workflow

### 1. Inspect without changing state

Run:

```bash
scripts/preflight.sh <ssh-host>
```

Use an SSH config alias when possible. Check OS support, architecture, free memory, swap, disk, port conflicts, SSH port, firewall state, Xray state, time synchronization, and available congestion-control algorithms.

Stop if the host is not Debian/Ubuntu, port 443 is occupied by an unrelated service, SSH access is uncertain, or disk/memory are critically low.

### 2. Select and verify the REALITY target

Select a stable TLS 1.3 site that:

- is reachable from the VPS;
- presents a valid certificate for the chosen SNI;
- is preferably near the VPS or in the same ASN;
- is not a user-controlled origin that could be harmed by fallback traffic.

Validate it from the VPS with OpenSSL and, after Xray is installed, `xray tls ping`. Do not guess an SNI merely because it is popular.

### 3. Preview the deployment

Run the deployer with `--dry-run`. This connects to the VPS and performs inspection but makes no changes:

```bash
scripts/deploy.sh \
  --ssh-host <ssh-host> \
  --reality-target <tls-hostname> \
  --dry-run
```

Review the detected SSH port, existing services, firewall, and planned output paths with the user.

### 4. Deploy after confirmation

For a new Ubuntu/Debian VPS:

```bash
scripts/deploy.sh \
  --ssh-host <ssh-host> \
  --reality-target <tls-hostname> \
  --server-ip <public-ip> \
  --enable-ufw
```

The deployer installs a checksum-verified official Xray binary, creates an unprivileged service account, generates credentials on the VPS, writes a hardened systemd service, enables BBR when supported, optionally creates swap, and optionally configures UFW. It validates the server config and performs an authenticated loopback proxy test before generating local client files.

Use `--force` only after the user explicitly approves replacing an existing Xray configuration. The script creates a timestamped backup under `/var/backups/` before replacement.

### 5. Verify the result

Require all of the following:

- `xray run -test` succeeds as the service user;
- `xray.service` is enabled and active;
- the configured TCP port is listening;
- an authenticated temporary client can proxy a request through the node;
- the FLClash/Mihomo YAML parses and imports successfully;
- mainland China rules precede `MATCH` and route to `DIRECT`.

From the user's actual Shenzhen connection, separately measure TCP latency, packet loss, and real download performance. A remote VPS self-test cannot measure the mainland access path.

### 6. Hand off securely

Report the protocol, port, SNI, public key fingerprint fields, firewall changes, BBR status, swap status, backup path, and verification results. Point to the generated local files without pasting their secrets into chat. Recommend rotating the UUID and short ID if a share link was exposed.

## Included tools

- `scripts/preflight.sh`: read-only VPS inspection.
- `scripts/deploy.sh`: guarded deployment and FLClash/Mihomo configuration generation.
- `references/design-and-troubleshooting.md`: protocol tradeoffs, REALITY target criteria, field mapping, rollback, and fault isolation.
