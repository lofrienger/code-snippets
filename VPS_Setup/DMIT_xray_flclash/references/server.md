# 服务端部署与维护

## 只读预检

在用户指定并核对过的 SSH 主机上执行；不默认使用旧会话中的 `vps` 别名：

```bash
hostname
cat /etc/os-release
uname -m
free -h
df -h /
timedatectl status
ss -lntup
ip -4 route
ip -6 route
sudo sshd -T
sudo ufw status verbose
systemctl status dmit-edge warp-svc --no-pager
```

关注实际 SSH 监听端口、TCP 和 UDP 的 443 占用、是否有已有 Xray 配置、WARP 是否接管默认路由。上面有命令不可用时记录并用系统对应工具检查，不为“预检”擅自安装软件。

授权前汇报：要安装哪些包/版本，新增账户/目录/服务，TCP/UDP 开放范围，可选 WARP 的影响，客户端是否需切换，以及回滚路径。备份现有配置、服务单元、路由与防火墙状态；只读检查不需要重启服务。

## 安装与隔离

- 从 [Xray 官方发行版](https://github.com/XTLS/Xray-core/releases)选适合 OS/架构的包，并核对该发行版发布的摘要。不要复制其他版本或架构的 SHA256；不要直接执行不明来源的一键脚本。
- 使用 `/opt/dmit-edge/releases/<version>/xray`，保留旧二进制以便回滚。把所选版本的绝对路径填入服务模板。
- 专用系统账户 `dmit-edge`，配置目录 `/usr/local/etc/dmit-edge`，状态目录 `/var/lib/dmit-edge`。现有路径若已存在，先检查和备份，不能当作空白新装。
- 配置与证书私钥用 `root:dmit-edge 0640`，配置目录 `0750`；客户端资料只存管理员私有目录。先检查服务账户可读性。
- 不默认添加 swap、改拥塞控制或停用其他服务；有实际资源/网络证据且获得授权后单独处理。

服务单元见 [assets/dmit-edge.service](../assets/dmit-edge.service)。它是参数化模板，不可直接启动；替换 `__XRAY_BIN__` 后检查权限和资源限制适合目标系统。历史小内存设置不保证适合所有机器。

## 凭据与 TLS

每个 `mac/windows/android/iphone` 分别创建 `direct/warp` 两组凭据：VLESS UUID、HY2 强随机密码、用于服务端路由的唯一用户标签，如 `mac-warp`。UUID 可用 `uuid.uuid4()`，密码可用 `secrets.token_urlsafe(32)`；输出直接写 `0600` 文件，不打印。

REALITY 用目标版本的 `xray x25519` 生成密钥；私钥留在 VPS，公钥进入客户端资料。不同版本 CLI 的公钥输出标签可能不同，不要把 `Hash32` 等字段误当公钥。short ID 用符合该版本长度限制的随机十六进制值。

选 REALITY 目标前从 VPS 检查 DNS、TLS 1.3、证书/SNI、连通性和实际代理握手；配置的 target、serverNames 和客户端 servername 必须一致。不要固定复制某个热门站点，目标变更需同步所有客户端。

HY2 私有证书示例（占位 IP 必须替换；在 VPS 的私有目录中生成）：

```bash
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
  -nodes -days 365 -subj /CN=dmit-proxy.internal \
  -addext 'subjectAltName=DNS:dmit-proxy.internal,IP:203.0.113.10' \
  -keyout server.key -out server.crt
openssl x509 -in server.crt -noout -fingerprint -sha256
```

证书的 SHA256 指纹规范化为无冒号的十六进制。Mihomo 使用 `fingerprint` 与 `skip-cert-verify: false`；sing-box 显式信任证书并保留 `insecure: false`。轮换证书时同步更新各设备 pin/证书，旧证书到期前预留更新窗口。

## Xray 配置契约

以下是 **26.3.27** 的结构示意，不是含完整凭据的可直接部署文件。生成完整配置时填入每设备凭据、REALITY 私钥、已验证目标和证书绝对路径；不把含私钥配置导出到客户端。

```json
{
  "log": {"loglevel": "warning", "access": "none"},
  "inbounds": [
    {
      "tag": "reality", "listen": "0.0.0.0", "port": 443,
      "protocol": "vless",
      "settings": {
        "clients": [{"id": "DEVICE_UUID", "flow": "xtls-rprx-vision", "email": "mac-direct"}],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "raw", "security": "reality",
        "realitySettings": {
          "show": false, "target": "VERIFIED_TLS_HOST:443",
          "serverNames": ["VERIFIED_TLS_HOST"],
          "privateKey": "SERVER_ONLY_PRIVATE_KEY", "shortIds": ["SHORT_ID_HEX"]
        }
      },
      "sniffing": {"enabled": true, "destOverride": ["http", "tls", "quic"], "routeOnly": true}
    },
    {
      "tag": "hy2", "listen": "0.0.0.0", "port": 443,
      "protocol": "hysteria",
      "settings": {"version": 2, "clients": [{"auth": "DEVICE_HY2_PASSWORD", "email": "mac-direct"}]},
      "streamSettings": {
        "network": "hysteria", "security": "tls",
        "tlsSettings": {
          "alpn": ["h3"],
          "certificates": [{"certificateFile": "/usr/local/etc/dmit-edge/server.crt", "keyFile": "/usr/local/etc/dmit-edge/server.key"}]
        },
        "hysteriaSettings": {"version": 2, "udpIdleTimeout": 60}
      },
      "sniffing": {"enabled": true, "destOverride": ["http", "tls", "quic"], "routeOnly": true}
    }
  ],
  "outbounds": [
    {"tag": "direct", "protocol": "freedom", "settings": {}},
    {"tag": "warp", "protocol": "socks", "settings": {"address": "127.0.0.1", "port": 40000}},
    {"tag": "block", "protocol": "blackhole", "settings": {}}
  ],
  "routing": {
    "domainStrategy": "IPOnDemand",
    "rules": [
      {"type": "field", "ip": ["0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.168.0.0/16", "224.0.0.0/4", "::/128", "::1/128", "fc00::/7", "fe80::/10", "ff00::/8"], "outboundTag": "block"},
      {"type": "field", "user": ["mac-warp", "windows-warp", "android-warp", "iphone-warp"], "network": "udp", "outboundTag": "block"},
      {"type": "field", "user": ["mac-warp", "windows-warp", "android-warp", "iphone-warp"], "network": "tcp", "outboundTag": "warp"}
    ]
  }
}
```

完整部署必须把所有用户加入两类入站，而不只是上例的 `mac-direct`。不启用 WARP 时省略 WARP 用户、出站和对应规则，仅保留原生路径。检查私有/保留地址阻止范围符合本机安全要求；这份列表不是完整云环境 SSRF 防护承诺。

**字段陷阱**：[26.3.27 源码](https://github.com/XTLS/Xray-core/blob/v26.3.27/infra/conf/hysteria.go)明确使用 JSON `clients`；[在线 Hysteria 文档](https://xtls.github.io/config/inbounds/hysteria.html)当前示例为 `users`。未知字段可能被忽略：语法检查之外，必须验证有效凭据成功、错误密码失败。新版本应重新检查协议、传输层、认证字段和 SOCKS 格式，不能照抄旧配置。

## 可选 WARP：只影响选定流量

按 [Cloudflare 官方软件源](https://pkg.cloudflareclient.com/)安装匹配当前系统的包，仓库代号从实际系统读取，不写死 `resolute`。先查看已安装版本的 `warp-cli --help`、`mode --help`、`proxy --help`；下列为本次使用的命令顺序：

```bash
warp-cli --accept-tos registration new
warp-cli --accept-tos mode proxy
warp-cli --accept-tos proxy port 40000
warp-cli --accept-tos connect
warp-cli --accept-tos status
ss -lntp
ip -4 route
ip -6 route
curl --socks5-hostname 127.0.0.1:40000 https://www.cloudflare.com/cdn-cgi/trace
```

注册/接受条款需要在用户授权范围内；已注册的机器不要盲目重新注册。只允许 loopback 监听；确认默认路由、SSH 和既有服务未被接管。看到 `warp=on` 才证明该次测试请求经 WARP；不能由“已安装 WARP”推导所有节点都走它。[Local proxy 模式说明](https://developers.cloudflare.com/warp-client/warp-modes/)。

WARP 凭据的 UDP 在 Xray 路由中先阻止，TCP 才转 SOCKS。上例不是全机隧道，不保证 UDP、QUIC 或地区解锁。

## 启动、变更与回滚

检查服务单元与配置后，授权范围内执行：

```bash
sudo -u dmit-edge /opt/dmit-edge/releases/CHOSEN_VERSION/xray run -test -config /usr/local/etc/dmit-edge/config.json
systemctl daemon-reload
systemctl enable --now dmit-edge
systemctl status dmit-edge --no-pager
journalctl -u dmit-edge -n 30 --no-pager
```

防火墙只追加必要的 TCP/UDP 代理端口，先保留实际 SSH 端口；若防火墙未启用，不擅自全局启用。还需检查提供商侧规则。从另一条 SSH 连接验证可登录后再结束原连接。

升级前备份二进制路径、配置、服务单元和凭据；验证新版本后才切换并重启本服务。失败时恢复对应版本与配置，重新测试。不要仅回滚服务端证书而忘记客户端 pin。

只停 WARP 可 `warp-cli --accept-tos disconnect`，原生路径应仍可用；恢复后检查状态和出口。全新部署需要停用时，先核对服务归属，再 `systemctl disable --now dmit-edge`；只有 WARP 也由本次专用安装且获准停用时才停 `warp-svc`。不删除 SSH、其他服务或已有共享 WARP。
