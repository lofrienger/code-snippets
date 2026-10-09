#!/usr/bin/env python3
"""Offline invariants; optional installed-core checks use isolated test fixtures."""

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import ssl
import subprocess
import tempfile
import unittest
import uuid

import render_profiles as profiles


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="dmit-skill-tests-")
        cls.root = Path(cls.workspace.name)
        result = subprocess.run([
            "openssl", "req", "-x509", "-newkey", "ec",
            "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-days", "1",
            "-subj", "/CN=dmit-proxy.internal",
            "-addext", "subjectAltName=DNS:dmit-proxy.internal,IP:203.0.113.10",
            "-keyout", str(cls.root / "fixture.key"),
            "-out", str(cls.root / "fixture.crt"),
        ], capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError("OpenSSL failed to create an isolated test certificate.")
        certificate = (cls.root / "fixture.crt").read_text()
        cls.access = {
            "server": "203.0.113.10", "port": 443, "sni": "dmit-proxy.internal",
            "certificate": certificate,
            "fingerprint": hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate)).hexdigest(),
            "reality_server_name": "example.com",
            "reality_public_key": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("="),
            "short_id": secrets.token_hex(8),
            "devices": {device: {route: {
                "uuid": str(uuid.uuid4()), "hy2_password": secrets.token_urlsafe(32),
            } for route in ("direct", "warp")} for device in profiles.DEVICES},
        }

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def new_access(self, directory):
        path = directory / "access.json"
        path.write_text(json.dumps(self.access))
        path.chmod(0o600)
        return path

    def test_access_accepts_fixture(self):
        profiles.validate_access(self.access)

    def test_wrong_certificate_pin_rejected(self):
        access = copy.deepcopy(self.access)
        access["fingerprint"] = "0" * 64
        with self.assertRaises(ValueError):
            profiles.validate_access(access)

    def test_malformed_certificate_rejected(self):
        access = copy.deepcopy(self.access)
        access["certificate"] = "-----BEGIN CERTIFICATE-----\naGVsbG8=\n-----END CERTIFICATE-----"
        with self.assertRaises(ValueError):
            profiles.validate_access(access)

    def test_duplicate_uuid_rejected(self):
        access = copy.deepcopy(self.access)
        access["devices"]["iphone"]["warp"]["uuid"] = access["devices"]["mac"]["direct"]["uuid"]
        with self.assertRaises(ValueError):
            profiles.validate_access(access)

    def test_duplicate_password_rejected(self):
        access = copy.deepcopy(self.access)
        access["devices"]["iphone"]["warp"]["hy2_password"] = access["devices"]["mac"]["direct"]["hy2_password"]
        with self.assertRaises(ValueError):
            profiles.validate_access(access)

    def test_private_keys_rejected(self):
        for field in ("privateKey", "certificate_private_key"):
            access = copy.deepcopy(self.access)
            access[field] = "do-not-export"
            with self.assertRaises(ValueError):
                profiles.validate_access(access)
        access = copy.deepcopy(self.access)
        access["extra"] = "-----BEGIN PRIVATE KEY-----"
        with self.assertRaises(ValueError):
            profiles.validate_access(access)

    def test_invalid_public_key_and_short_id_rejected(self):
        for key, value in (("reality_public_key", "invalid"), ("short_id", "abc")):
            access = copy.deepcopy(self.access)
            access[key] = value
            with self.assertRaises(ValueError):
                profiles.validate_access(access)

    def test_partial_warp_layout_rejected(self):
        access = copy.deepcopy(self.access)
        del access["devices"]["android"]["warp"]
        with self.assertRaises(ValueError):
            profiles.validate_access(access)

    def test_optional_warp_omits_all_warp_nodes(self):
        access = copy.deepcopy(self.access)
        for routes in access["devices"].values():
            del routes["warp"]
        profiles.validate_access(access)
        clash = profiles.clash_profile(access, "mac")
        self.assertEqual(clash["proxy-groups"][1]["proxies"], ["PROXY"])
        self.assertEqual(len(clash["proxies"]), 2)
        self.assertFalse(any(n["tag"].endswith("-warp") for n in profiles.singbox_profile(access)["outbounds"]))

    def test_clash_route_dns_and_device_bindings(self):
        for device in profiles.DEVICES:
            config = profiles.clash_profile(self.access, device)
            self.assertEqual(config["rules"][-1], "MATCH,PROXY")
            self.assertIn("DOMAIN-SUFFIX,arxiv.org,DIRECT", config["rules"][:-1])
            self.assertEqual(config["dns"]["nameserver-policy"]["+.arxiv.org"], list(profiles.DIRECT_DNS))
            self.assertIn("RULE-SET,cn,DIRECT", config["rules"])
            self.assertEqual(config["dns"]["nameserver-policy"]["+.ts.net"], ["system"])
            self.assertFalse(config["allow-lan"])
            self.assertFalse(config["tun"]["enable"])
            self.assertNotIn("external-controller", config)
            for node in config["proxies"]:
                route = node["name"].split("-")[1]
                expected = self.access["devices"][device][route]
                if node["type"] == "hysteria2":
                    self.assertEqual(node["password"], expected["hy2_password"])
                    self.assertFalse(node["skip-cert-verify"])
                    self.assertEqual(node["fingerprint"], self.access["fingerprint"])
                else:
                    self.assertEqual(node["uuid"], expected["uuid"])
                    self.assertEqual(node["udp"], route == "direct")

    def test_singbox_routing_and_warp_tcp_only(self):
        config = profiles.singbox_profile(self.access)
        rules = config["route"]["rules"]
        self.assertEqual(rules[2:4], [
            {"domain_suffix": ["arxiv.org"], "action": "resolve"},
            {"domain_suffix": ["arxiv.org"], "outbound": "direct"},
        ])
        self.assertEqual(config["dns"]["rules"][0], {"domain_suffix": ["arxiv.org"], "server": "local"})
        self.assertEqual(config["route"]["final"], "PROXY")
        for node in config["outbounds"]:
            if node["tag"].endswith("-warp"):
                self.assertEqual(node["network"], "tcp")
            if node["type"] == "hysteria2":
                self.assertFalse(node["tls"]["insecure"])
                self.assertEqual(node["tls"]["certificate"], [self.access["certificate"]])
        country_ip = next(i for i, rule in enumerate(rules) if rule.get("rule_set") == ["cn-ip"])
        ai = next(i for i, rule in enumerate(rules) if rule.get("outbound") == "AI")
        self.assertLess(ai, country_ip)

    def test_proxy_arxiv_and_preferred_reality(self):
        clash = profiles.clash_profile(self.access, "mac", False, "reality")
        self.assertNotIn("DOMAIN-SUFFIX,arxiv.org,DIRECT", clash["rules"])
        self.assertNotIn("+.arxiv.org", clash["dns"]["nameserver-policy"])
        self.assertEqual(clash["proxy-groups"][0]["proxies"][0], "REALITY-direct")
        config = profiles.singbox_profile(self.access, False, "reality")
        self.assertFalse(any("arxiv.org" in rule.get("domain_suffix", []) for rule in config["route"]["rules"]))
        self.assertEqual(config["outbounds"][0]["default"], "REALITY-direct")

    def test_render_permissions_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            directory = Path(directory)
            source = self.new_access(directory)
            output = directory / "bundle"
            self.assertEqual(profiles.render(source, output), 5)
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            for path in output.iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                json.loads(path.read_text())
            with self.assertRaises(FileExistsError):
                profiles.render(source, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_git_output_rejected_without_files(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            directory = Path(directory)
            source = self.new_access(directory)
            repository = directory / "repo"
            repository.mkdir()
            (repository / ".git").write_text("gitdir: elsewhere")
            with self.assertRaises(ValueError):
                profiles.render(source, repository / "bundle")
            self.assertFalse((repository / "bundle").exists())

    def test_insecure_source_permissions_rejected(self):
        if os.name != "posix":
            self.skipTest("Unix mode check")
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            directory = Path(directory)
            source = self.new_access(directory)
            source.chmod(0o644)
            with self.assertRaises(ValueError):
                profiles.render(source, directory / "bundle")

    def test_cli_does_not_print_credentials(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            directory = Path(directory)
            source = self.new_access(directory)
            result = subprocess.run([
                os.sys.executable, str(Path(profiles.__file__)),
                "--access", str(source), "--output", str(directory / "bundle"),
            ], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            for routes in self.access["devices"].values():
                for credential in routes.values():
                    for value in credential.values():
                        self.assertNotIn(value, result.stdout + result.stderr)

    def test_mihomo_installed_core(self):
        binary = os.environ.get("DMIT_TEST_MIHOMO")
        if not binary:
            self.skipTest("Set DMIT_TEST_MIHOMO to check an installed core.")
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            directory = Path(directory)
            profiles.render(self.new_access(directory), directory / "bundle")
            runtime = directory / "runtime"
            runtime.mkdir()
            result = subprocess.run([binary, "-t", "-d", str(runtime), "-f", str(directory / "bundle/mac-flclash.yaml")], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_singbox_installed_core(self):
        binary = os.environ.get("DMIT_TEST_SINGBOX")
        if not binary:
            self.skipTest("Set DMIT_TEST_SINGBOX to check an installed core.")
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            directory = Path(directory)
            profiles.render(self.new_access(directory), directory / "bundle")
            result = subprocess.run([binary, "check", "-c", str(directory / "bundle/iphone-singbox.json")], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
