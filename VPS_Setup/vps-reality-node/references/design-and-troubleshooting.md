# Design and troubleshooting reference

## Protocol selection

| Profile | Best fit | Main tradeoff |
| --- | --- | --- |
| VLESS + TCP + XTLS Vision + REALITY | China Telecom, no domain, unknown UDP quality, 1 GB VPS | Depends on a suitable TLS target; TCP retransmission can amplify loss |
| NaiveProxy | A real domain and certificate are available; conventional HTTPS behavior is desired | Requires domain/certificate lifecycle and a web-facing service |
| Hysteria2 | Stable UDP path with measurable loss or bandwidth variation | UDP may be throttled or blocked and should be tested from the real client network |

Do not claim that one transport is universally fastest. Routing, congestion, time of day, server location, and the mainland carrier usually dominate protocol micro-benchmarks.

## REALITY target checklist

The target and `serverName` are security and reliability inputs, not cosmetic settings.

1. Test from the VPS, not only from the operator's laptop.
2. Require a successful TLS 1.3 handshake with the intended SNI.
3. Verify that the certificate SAN covers that SNI and certificate verification succeeds.
4. Prefer a stable origin close to the VPS, ideally in the same ASN.
5. Avoid arbitrary CDN front doors and fragile or politically sensitive targets.
6. Re-test the target when an otherwise healthy node suddenly stops handshaking.

Useful checks:

```bash
openssl s_client -connect example.com:443 -servername example.com -tls1_3 -verify_return_error </dev/null
/usr/local/bin/xray tls ping example.com:443
```

The first command is authoritative for certificate and TLS validation. `xray tls ping` is an additional compatibility signal and may vary across Xray versions.

## Server-to-client field mapping

| Xray server | Mihomo/FLClash client | Handling |
| --- | --- | --- |
| `settings.clients[].id` | `uuid` | Secret client credential |
| `settings.clients[].flow` | `flow` | Must be `xtls-rprx-vision` on both sides |
| `realitySettings.serverNames[]` | `servername` | Exact SNI hostname |
| `realitySettings.privateKey` | Never exported | Server secret |
| Client password/public-key material derived by `xray x25519` | Xray: `realitySettings.password`; Mihomo: `reality-opts.public-key` | Client credential; keep it inside the client profile |
| `realitySettings.shortIds[]` | `reality-opts.short-id` | Client credential; rotate if exposed |
| Inbound TCP port | `port` | Usually 443 |

## Mainland-direct policy

Keep private and mainland rules before the catch-all rule:

```yaml
rules:
  - GEOSITE,private,DIRECT
  - GEOIP,private,DIRECT,no-resolve
  - GEOSITE,cn,DIRECT
  - GEOIP,cn,DIRECT,no-resolve
  - MATCH,Proxy
```

Use a mainland resolver for mainland domains. When `respect-rules` is enabled, send non-mainland DNS through the proxy group and define `proxy-server-nameserver` so the proxy hostname can be resolved without a loop.

## Fault isolation

### SSH works but the node port is closed

1. Check `systemctl is-active xray` and `journalctl -u xray --since -10min`.
2. Check `ss -lntp` for the configured port.
3. Check provider firewall/security-group rules as well as UFW/nftables.
4. Confirm that another service did not take port 443.

### REALITY handshake fails

1. Confirm client `uuid`, `flow`, `servername`, public key, and short ID exactly match.
2. Confirm the VPS clock is synchronized with `timedatectl` or `chronyc`.
3. Re-run the TLS target checks from the VPS.
4. Test with the VPS IP rather than an unverified DNS record.
5. Inspect warning-level Xray logs without enabling verbose logs for long periods.

### Process is killed or restarts

Check `journalctl -k` and `dmesg` for OOM-killer messages. On a 1 GB VPS, retain at least 1 GB swap, keep Xray's memory limit conservative, and avoid running package upgrades concurrently with memory-heavy tools.

### Speed is poor from China Telecom

Measure from the actual client:

- repeated TCP ping or `mtr -T` to port 443;
- packet loss at busy and quiet hours;
- a single-stream and multi-stream download through the proxy;
- direct comparison with another VPS location or carrier-optimized route.

BBR can improve sender behavior but cannot repair a congested or badly routed international path. A US West Coast route often has lower propagation delay than central/eastern US, but the provider's China routes matter more than map distance.

## Rollback

The deployer stores replaced files under a timestamped `/var/backups/xray-reality-*` directory. To roll back, stop Xray, restore the recorded config/service/sysctl files, run `systemctl daemon-reload`, validate the restored config, and restart the previous service. Do not delete swap or firewall rules until their prior ownership and purpose are known.

## Primary documentation

- Xray core releases and source: <https://github.com/XTLS/Xray-core>
- Xray official installer reference: <https://github.com/XTLS/Xray-install>
- REALITY transport: <https://xtls.github.io/en/config/transports/reality.html>
- VLESS inbound: <https://xtls.github.io/en/config/inbounds/vless.html>
- Mihomo VLESS fields: <https://wiki.metacubex.one/en/config/proxies/vless/>
- Mihomo DNS: <https://wiki.metacubex.one/en/config/dns/>
