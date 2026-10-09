#!/usr/bin/env python3
"""Render private multi-device profiles; never deploy or overwrite profiles."""

import argparse
import base64
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import ssl
import sys
import uuid


DEVICES = ("mac", "windows", "android", "iphone")
AI_DOMAINS = (
    "openai.com", "chatgpt.com", "oaistatic.com", "oaiusercontent.com",
    "anthropic.com", "claude.ai", "gemini.google.com",
    "generativelanguage.googleapis.com", "aistudio.google.com",
)
DOMESTIC_DOMAINS = (
    "douyin.com", "douyinvod.com", "douyinpic.com", "douyinstatic.com",
    "douyincdn.com", "idouyinvod.com", "doubao.com", "doubaocdn.com",
    "zijieapi.com", "zijieapi.net", "amemv.com", "snssdk.com", "pstatp.com",
    "byteimg.com", "ixigua.com", "ixiguavideo.com", "toutiao.com",
)
LOCAL_SUFFIXES = ("lan", "local", "ts.net")
DIRECT_DNS = (
    "https://dns.alidns.com/dns-query#DIRECT",
    "https://doh.pub/dns-query#DIRECT",
)
REMOTE_DNS = "https://1.1.1.1/dns-query#PROXY"
REQUIRED_FIELDS = (
    "server", "port", "sni", "certificate", "fingerprint",
    "reality_server_name", "reality_public_key", "short_id", "devices",
)


def in_git_tree(path):
    resolved = Path(path).expanduser().resolve()
    return any((parent / ".git").exists() for parent in (resolved, *resolved.parents))


def validate_access(access):
    if not isinstance(access, dict) or any(key not in access for key in REQUIRED_FIELDS):
        raise ValueError("Access file is missing required client fields.")

    def reject_private_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if "private" in key.lower().replace("_", "").replace("-", ""):
                    raise ValueError("Access file must not contain private-key fields.")
                reject_private_keys(item)
        elif isinstance(value, list):
            for item in value:
                reject_private_keys(item)
        elif isinstance(value, str) and "PRIVATE KEY-----" in value:
            raise ValueError("Access file must not contain PEM private keys.")

    reject_private_keys(access)
    try:
        ipaddress.ip_address(access["server"])
    except (ValueError, TypeError):
        raise ValueError("server must be an IP address.") from None
    if type(access["port"]) is not int or not 1 <= access["port"] <= 65535:
        raise ValueError("port must be an integer between 1 and 65535.")
    for field in ("sni", "reality_server_name"):
        value = access[field]
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9.-]+", value):
            raise ValueError(f"{field} must be a hostname.")
    try:
        der = ssl.PEM_cert_to_DER_cert(access["certificate"])
        # Parse the trust material as well as hashing it; malformed PEM is not enough.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(cadata=access["certificate"])
    except (ValueError, TypeError, ssl.SSLError):
        raise ValueError("certificate must contain a valid PEM certificate.") from None
    pin = access["fingerprint"]
    if not isinstance(pin, str) or not re.fullmatch(r"[A-Fa-f0-9]{64}", pin):
        raise ValueError("fingerprint must be 64 hex characters without colons.")
    if hashlib.sha256(der).hexdigest() != pin.lower():
        raise ValueError("Certificate SHA256 does not match fingerprint.")
    public_key = access["reality_public_key"]
    if not isinstance(public_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", public_key):
        raise ValueError("reality_public_key must be a 32-byte base64url public key.")
    if len(base64.urlsafe_b64decode(public_key + "=")) != 32:
        raise ValueError("Invalid REALITY public key length.")
    sid = access["short_id"]
    if not isinstance(sid, str) or not re.fullmatch(r"(?:[0-9A-Fa-f]{2}){1,8}", sid):
        raise ValueError("short_id must be 2 to 16 hex characters in byte pairs.")
    devices = access["devices"]
    if not isinstance(devices, dict) or set(devices) != set(DEVICES):
        raise ValueError("devices must contain mac, windows, android and iphone.")
    if any(not isinstance(routes, dict) for routes in devices.values()):
        raise ValueError("Each device must define credential routes.")
    has_warp = "warp" in devices["mac"]
    expected_routes = {"direct", "warp"} if has_warp else {"direct"}
    uuids, passwords = set(), set()
    for device in DEVICES:
        if set(devices[device]) != expected_routes:
            raise ValueError("All devices must use the same direct/optional-warp layout.")
        for credentials in devices[device].values():
            if not isinstance(credentials, dict):
                raise ValueError("Route credentials must be objects.")
            value = credentials.get("uuid")
            try:
                normalized = str(uuid.UUID(value))
            except (ValueError, TypeError, AttributeError):
                raise ValueError("Invalid device UUID.") from None
            password = credentials.get("hy2_password")
            if not isinstance(password, str) or len(password) < 24:
                raise ValueError("HY2 passwords must have at least 24 characters.")
            if normalized in uuids or password in passwords:
                raise ValueError("Credentials must be unique across devices and routes.")
            uuids.add(normalized)
            passwords.add(password)


def clash_profile(access, device, arxiv_direct=True, preferred="hy2"):
    nodes = []
    for route, credentials in access["devices"][device].items():
        nodes.extend([
            {
                "name": "HY2-" + route, "type": "hysteria2",
                "server": access["server"], "port": access["port"],
                "password": credentials["hy2_password"], "sni": access["sni"],
                "skip-cert-verify": False, "fingerprint": access["fingerprint"].lower(),
                "alpn": ["h3"],
            },
            {
                "name": "REALITY-" + route, "type": "vless",
                "server": access["server"], "port": access["port"],
                "uuid": credentials["uuid"], "network": "tcp", "tls": True,
                "flow": "xtls-rprx-vision", "servername": access["reality_server_name"],
                "client-fingerprint": "chrome",
                "reality-opts": {"public-key": access["reality_public_key"], "short-id": access["short_id"]},
                "udp": route == "direct",
            },
        ])
    original_nodes = ["HY2-direct", "REALITY-direct"]
    if preferred == "reality":
        original_nodes.reverse()
    ai_nodes = ["PROXY"]
    if "warp" in access["devices"][device]:
        ai_nodes.extend(["HY2-warp", "REALITY-warp"])
    rules = [f"IP-CIDR,{cidr},DIRECT,no-resolve" for cidr in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10",
    )]
    rules.extend(f"DOMAIN-SUFFIX,{domain},DIRECT" for domain in LOCAL_SUFFIXES)
    if arxiv_direct:
        rules.append("DOMAIN-SUFFIX,arxiv.org,DIRECT")
    rules.extend(f"DOMAIN-SUFFIX,{domain},AI" for domain in AI_DOMAINS)
    rules.extend([
        "RULE-SET,domestic-critical,DIRECT", "DOMAIN-SUFFIX,cn,DIRECT",
        "RULE-SET,cn,DIRECT", "GEOIP,CN,DIRECT", "MATCH,PROXY",
    ])
    policy = {"+.cn": list(DIRECT_DNS)}
    policy.update({f"rule-set:{tag}": list(DIRECT_DNS) for tag in ("domestic-critical", "cn")})
    policy.update({f"+.{domain}": ["system"] for domain in LOCAL_SUFFIXES})
    policy.update({f"+.{domain}": [REMOTE_DNS] for domain in AI_DOMAINS})
    if arxiv_direct:
        policy["+.arxiv.org"] = list(DIRECT_DNS)
    return {
        "mixed-port": 7890, "allow-lan": False, "mode": "rule",
        "log-level": "warning", "ipv6": False,
        "proxies": nodes,
        "proxy-groups": [
            {"name": "PROXY", "type": "select", "proxies": original_nodes},
            {"name": "AI", "type": "select", "proxies": ai_nodes},
        ],
        "rule-providers": {
            "domestic-critical": {"type": "inline", "behavior": "domain", "payload": ["+." + d for d in DOMESTIC_DOMAINS]},
            "cn": {
                "type": "http", "behavior": "domain", "format": "mrs",
                "path": "./providers/dmit/cn.mrs",
                "url": "https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/meta/geo/geosite/cn.mrs",
                "proxy": "PROXY", "interval": 86400,
            },
        },
        "rules": rules,
        "dns": {
            "enable": True, "listen": "127.0.0.1:1053", "ipv6": False,
            "enhanced-mode": "fake-ip", "fake-ip-filter": ["*." + d for d in LOCAL_SUFFIXES],
            "default-nameserver": ["223.5.5.5"], "nameserver": [REMOTE_DNS],
            "nameserver-policy": policy, "direct-nameserver": list(DIRECT_DNS),
            "direct-nameserver-follow-policy": True, "respect-rules": True,
            "proxy-server-nameserver": ["223.5.5.5"],
        },
        "tun": {"enable": False, "stack": "mixed", "auto-route": True, "auto-detect-interface": True, "dns-hijack": ["any:53"]},
    }


def singbox_profile(access, arxiv_direct=True, preferred="hy2"):
    clash = clash_profile(access, "iphone", arxiv_direct, preferred)
    nodes = []
    for route, credentials in access["devices"]["iphone"].items():
        hy = {
            "type": "hysteria2", "tag": "HY2-" + route,
            "server": access["server"], "server_port": access["port"],
            "password": credentials["hy2_password"],
            "tls": {"enabled": True, "server_name": access["sni"], "insecure": False, "certificate": [access["certificate"]], "alpn": ["h3"]},
        }
        reality = {
            "type": "vless", "tag": "REALITY-" + route,
            "server": access["server"], "server_port": access["port"],
            "uuid": credentials["uuid"], "flow": "xtls-rprx-vision",
            "tls": {
                "enabled": True, "server_name": access["reality_server_name"],
                "utls": {"enabled": True, "fingerprint": "chrome"},
                "reality": {"enabled": True, "public_key": access["reality_public_key"], "short_id": access["short_id"]},
            },
        }
        if route == "warp":
            hy["network"] = reality["network"] = "tcp"
        nodes.extend([hy, reality])
    rules = [{"action": "sniff"}, {"protocol": "dns", "action": "hijack-dns"}]
    dns_rules = []
    if arxiv_direct:
        rules.extend([{"domain_suffix": ["arxiv.org"], "action": "resolve"}, {"domain_suffix": ["arxiv.org"], "outbound": "direct"}])
        dns_rules.append({"domain_suffix": ["arxiv.org"], "server": "local"})
    rules.extend([
        {"ip_is_private": True, "outbound": "direct"},
        {"ip_cidr": ["100.64.0.0/10"], "outbound": "direct"},
        {"domain_suffix": list(LOCAL_SUFFIXES), "action": "resolve"},
        {"domain_suffix": list(LOCAL_SUFFIXES), "outbound": "direct"},
        {"domain_suffix": list(AI_DOMAINS), "outbound": "AI"},
    ])
    # AI domain rules run before country-IP routing so they keep the chosen AI exit.
    for match in ({"rule_set": ["domestic-critical"]}, {"domain_suffix": ["cn"]}, {"rule_set": ["cn"]}):
        rules.extend([{**match, "action": "resolve"}, {**match, "outbound": "direct"}])
    rules.extend([{"action": "resolve"}, {"rule_set": ["cn-ip"], "outbound": "direct"}])
    dns_rules.extend([
        {"domain_suffix": list(LOCAL_SUFFIXES), "server": "system"},
        {"domain_suffix": list(AI_DOMAINS), "server": "remote"},
        {"domain_suffix": ["cn"], "server": "local"},
        {"rule_set": ["domestic-critical", "cn"], "server": "local"},
    ])
    groups = [{"type": "selector", "tag": g["name"], "outbounds": g["proxies"], "default": g["proxies"][0]} for g in clash["proxy-groups"]]
    remote_sets = [{
        "type": "remote", "tag": tag, "format": "binary",
        "url": f"https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/{kind}/cn.srs",
        "http_client": {"detour": "PROXY"}, "update_interval": "1d",
    } for tag, kind in (("cn", "geosite"), ("cn-ip", "geoip"))]
    return {
        "log": {"level": "warn"},
        "dns": {
            "servers": [
                {"type": "https", "tag": "remote", "server": "1.1.1.1", "detour": "PROXY"},
                {"type": "https", "tag": "local", "server": "223.5.5.5", "tls": {"enabled": True, "server_name": "dns.alidns.com"}},
                {"type": "local", "tag": "system"},
            ],
            "rules": dns_rules, "final": "remote", "strategy": "ipv4_only",
        },
        "inbounds": [{"type": "tun", "tag": "tun-in", "address": ["172.19.0.1/30"], "auto_route": True, "stack": "mixed"}],
        "outbounds": groups + [{"type": "direct", "tag": "direct"}] + nodes,
        "route": {
            "auto_detect_interface": True, "default_domain_resolver": "remote",
            "rules": rules,
            "rule_set": [{"type": "inline", "tag": "domestic-critical", "rules": [{"domain_suffix": list(DOMESTIC_DOMAINS)}]}] + remote_sets,
            "final": "PROXY",
        },
    }


def render(access_path, output, arxiv_direct=True, preferred="hy2"):
    access_path = Path(access_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if in_git_tree(access_path) or in_git_tree(output):
        raise ValueError("Private access and output paths must be outside Git repositories.")
    if os.name == "posix" and access_path.stat().st_mode & 0o077:
        raise ValueError("Access file must not be readable by group or other users.")
    access = json.loads(access_path.read_text(encoding="utf-8"))
    validate_access(access)
    profiles = {device + "-flclash.yaml": clash_profile(access, device, arxiv_direct, preferred) for device in DEVICES}
    profiles["iphone-singbox.json"] = singbox_profile(access, arxiv_direct, preferred)
    previous_umask = os.umask(0o077)
    try:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, profile in profiles.items():
            with (output / name).open("x", encoding="utf-8") as handle:
                json.dump(profile, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
    finally:
        os.umask(previous_umask)
    return len(profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access", required=True, type=Path, help="Private client-only access.json, outside Git")
    parser.add_argument("--output", required=True, type=Path, help="New private directory, outside Git")
    parser.add_argument("--arxiv-route", choices=("direct", "proxy"), default="direct")
    parser.add_argument("--preferred-transport", choices=("hy2", "reality"), default="hy2")
    args = parser.parse_args()
    try:
        count = render(args.access, args.output, args.arxiv_route == "direct", args.preferred_transport)
    except ValueError as exc:
        # JSON decoding errors may contain excerpts; do not emit underlying data.
        message = "Invalid access JSON." if isinstance(exc, json.JSONDecodeError) else str(exc)
        print("Error: " + message, file=sys.stderr)
        return 1
    except OSError:
        print("Error: cannot read access file or create a new private output directory; nothing is overwritten.", file=sys.stderr)
        return 1
    print(f"Generated {count} private profiles. No credentials printed; no client or server state changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
