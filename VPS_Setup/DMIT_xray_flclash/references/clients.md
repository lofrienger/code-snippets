# 多平台配置与同步

## 私有客户端资料契约

`render_profiles.py` 只读取客户端所需材料，不读取 REALITY 或证书私钥。它不创建服务端用户；每组凭据必须已经配置在对应的服务端入站，并通过用户标签绑定原生/WARP 路径。

`access.json` 的结构如下；这是字段说明，不是有效配置，不要把真实文件复制到仓库：

```text
server                 目标服务器 IP
port                   代理端口，示例为 443
sni                    HY2 证书中的主机名
certificate            HY2 公共证书 PEM
fingerprint            上述证书 DER 的 SHA256，64 位十六进制
reality_server_name    已验证的 REALITY SNI
reality_public_key     REALITY X25519 公钥，base64url
short_id               REALITY short ID，偶数个十六进制字符，最多 16 位
devices
  mac/windows/android/iphone
    direct
      uuid             该设备 VLESS UUID
      hy2_password     该设备 HY2 密码
    warp               可选，但若启用必须四台设备都有
      uuid             独立 WARP VLESS UUID
      hy2_password     独立 WARP HY2 密码
```

私有资料包含公共证书，但 UUID/密码仍是敏感信息。通过 SSH 从 VPS 导出经过白名单挑选的字段；确认文件没有 `privateKey`、`private_key` 或证书私钥。不要直接下载完整服务端配置。

## 生成新资料包

```bash
python3 scripts/render_profiles.py \
  --access /private/path/access.json \
  --output /private/path/new-bundle
```

脚本使用标准库，支持 Python 3.10+；先检查证书指纹、UUID、公钥形状、设备凭据唯一性、WARP 配套和输出位置，再创建文件。默认保留本次选择的 arXiv 直连；网络环境不适合直连时用 `--arxiv-route proxy`，应在实测后选择。默认首选 HY2；UDP 不适合时可 `--preferred-transport reality`。

输出：`mac-flclash.yaml`、`windows-flclash.yaml`、`android-flclash.yaml`、`iphone-flclash.yaml`、`iphone-singbox.json`。Clash 文件使用 JSON 序列化，兼容 YAML；不要因为扩展名是 `.yaml` 就用不适配的 JSON/YAML 处理方式。

`iphone-flclash.yaml` 是 Clash 格式备用文件，不声称 iPhone 存在或已安装 FlClash；iPhone 主交付为 sing-box。不同 iOS 客户端的格式不可互换。

脚本生成的是本次规则方案的新配置，不保留任意第三方 profile 的自定义设置。已有 profile 只需局部更新时，应备份后针对该文件改动，不直接用生成器覆盖。

## 节点与分组

- `PROXY`：仅原生 `HY2-direct`、`REALITY-direct`。
- `AI`：默认跟随 `PROXY`；配置了 WARP 才额外包含 `HY2-warp`、`REALITY-warp`。
- 国内、私网、Tailscale 和本次选择的 arXiv 走终端 `DIRECT`；其余走 `PROXY`。
- 名为 `HY2-direct` 的节点是 **DMIT 原生出口**，仍经过代理；规则 `DIRECT` 是 **终端直连**。不要混淆。

AI 规则覆盖本次列出的 OpenAI/ChatGPT、Anthropic/Claude 和 Gemini 域名，不是完整服务依赖清单。Antigravity 或其他未列域名默认经过 `PROXY`，若需要 AI 专属出口，先观察真实请求再补相应域名，不能只加官网域名就宣称登录/模型接口全部覆盖。

## 规则和 DNS 必须配套

Mihomo 规则顺序：私网/Tailscale → arXiv（若直连）→ 明确 AI 域名 → 国内关键域名 → `.cn` → 国内域名规则集 → `GEOIP,CN` → `MATCH,PROXY`。

DNS 采用 fake-IP；外网 DoH 明确走 `PROXY`，国内与直连 arXiv 的 DoH 走 `DIRECT`，Tailscale/`.lan`/`.local` 使用系统解析。设置 `respect-rules` 时提供独立 `proxy-server-nameserver` 避免解析引导循环。DNS 走向不能只靠流量规则推断。[Mihomo DNS 参考](https://wiki.metacubex.one/config/dns/)。

国内补丁覆盖这次涉及的抖音/豆包等域名，同时使用 MetaCubeX 的国内域名规则库；维护时依据实际请求更新，不假定清单永远完整。规则库是第三方维护数据，需确认用户接受下载来源、可用性与缓存。

sing-box 使用对应的 DNS server/rules 和 route rules；对需要直连的域名先 `resolve` 再路由，避免域名规则命中后未按预期解析。`cn` 域名和 `cn-ip` 二进制规则集分别配置。生成器面向本次 `1.14.2` 结构（含 `http_client.detour`），其他版本以实际 `sing-box check` 为准。[sing-box 配置](https://sing-box.sagernet.org/configuration/)。

Tailscale 系统解析只在目标设备已经具备正确的 tailnet/MagicDNS 配置时有效；绕过代理不等于自动安装或登录 Tailscale。

## 校验、备份、导入、应用

1. 在仓库外备份所有待更新源文件及 active imported profile，保留原来节点选择。不要同时改写归档副本。
2. 用实际或对应内核检查配置；JSON parse 只是基础检查：

   ```bash
   jq empty /private/path/new-bundle/mac-flclash.yaml
   /path/to/mihomo -t -d /private/path/check-runtime -f /private/path/new-bundle/mac-flclash.yaml
   /path/to/sing-box check -c /private/path/new-bundle/iphone-singbox.json
   ```

   `check-runtime` 是独立目录；规则库/GeoIP 的下载失败要单独处理，不能删掉必要规则来伪造“校验成功”。

3. FlClash 新增本地配置并明确选择目标 profile，保留旧 profile。源文件、导入副本、运行时生成配置可能不同；不要只改生成的 `config.yaml`，它可能被应用重新生成。
4. 按当前客户端 UI 保存/应用或重启核心；“强制重启核心”不等于重启电脑。不能把点击按钮本身当作生效证据。
5. 核对节点选择、规则模式、系统代理/TUN 状态，并观察一条新的真实连接。已有长连接可能继续使用旧路由。
6. 其他设备通过用户选择的可信方式传输，分别导入并验收。Windows 远程目录看到文件只证明传输；本机 sing-box 校验不证明 iPhone VPN 权限和实际网络。

FlClash 系统代理适合遵循系统设置的应用；终端、IDE 后台进程或部分 AI 工具可能不遵循。按实际客户端文档配置显式代理，或在获得授权后使用 TUN/VPN 并重新验收；不要擅自全局接管网络。

HY2 的 `fingerprint` 是证书 pin；REALITY 的 `client-fingerprint: chrome` 是 TLS 客户端特征，两者不是同一个概念。[Mihomo HY2 TLS 字段](https://wiki.metacubex.one/config/proxies/hysteria2/)、[sing-box HY2](https://sing-box.sagernet.org/configuration/outbound/hysteria2/)。
