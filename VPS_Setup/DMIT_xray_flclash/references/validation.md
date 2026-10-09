# 验收与故障排查

## 先定义“成功”

分清：语法正确 → 内核接受 → 服务运行 → 客户端实际连接 → 正确出口/路由 → 账号和模型可用。较早一层通过不能替代较晚一层。

记录目标设备、网络、时间、版本、节点和测试 URL。服务状态、网站地区要求、IP 地理库与速度结果都可能变化；下列历史证据不替代实时检查。

## 四条路径和负向测试

可选 WARP 已部署时，分别验证 HY2/REALITY × 原生/WARP。测试配置强制 `MATCH,TEST`，TEST 组单独选择目标节点；临时监听只在 loopback，使用独立端口与数据目录，不更改正式 profile。

```bash
curl --proxy http://127.0.0.1:17891 --connect-timeout 8 --max-time 20 \
  https://www.cloudflare.com/cdn-cgi/trace
```

原生节点应为 DMIT 出口、`warp=off`；WARP 节点应为 WARP 出口、`warp=on`。具体 IP/loc 不写死。仅测试 trace 不证明所有服务的路由。

同时做有界负向测试：

- HY2 使用错误证书 pin：连接必须失败；不要临时关闭证书验证。
- 使用新生成但不在服务端的 HY2 密码/UUID：认证必须失败。防止认证字段被忽略或用户列表未加载。
- 经授权暂时断开本次专用 WARP，原生节点应仍通，WARP 节点应失败而非原生回落；恢复并重测。未获授权时检查配置并将失效测试标为未运行。

## 活动 profile 验证

优先用 FlClash 连接列表观察新请求的 host、命中规则、代理链。若使用 Mihomo 控制接口，只连已确认的本机地址，按需要认证；不要公开 secret 或把控制端口开放给 LAN。

预期案例：`arxiv.org:443 → DomainSuffix(arxiv.org) → DIRECT`；AI 域名 → `AI → PROXY → 当前原生节点`，或用户明确选择的 WARP 节点。DoH 的出站也需核对；看到网页打开不说明 DNS 一定按策略执行。

不要因源配置写了 `DIRECT` 就宣布直连；已有长连接、另一份 profile、应用不遵循系统代理，都可能造成不同结果。

## IPv4 / IPv6 出口解释

分开检查两段：终端 → DMIT 的节点地址，以及 DMIT → 目标站的出站。节点填写 IPv4、客户端 `ipv6: false`、DNS 禁止 AAAA，仍可能由服务器为域名请求选择 IPv6。面板检测服务只是一次请求。

经本机代理比较 IPv4-only 与双栈检测端点：

```bash
curl --proxy http://127.0.0.1:7890 https://api4.ipify.org
curl --proxy http://127.0.0.1:7890 https://api64.ipify.org
curl --proxy http://127.0.0.1:7890 https://www.cloudflare.com/cdn-cgi/trace
```

在已核对的 VPS 上查看 `ip -o -6 addr show scope global` 和 direct 出站解析策略。需要比较某服务的 IPv4/IPv6 时，在 VPS 直接强制测试：

```bash
curl -4 --noproxy '*' --connect-timeout 5 --max-time 15 -o /dev/null \
  -w 'HTTP %{http_code}, target %{remote_ip}, total %{time_total}\n' https://api.anthropic.com/v1/models
curl -6 --noproxy '*' --connect-timeout 5 --max-time 15 -o /dev/null \
  -w 'HTTP %{http_code}, target %{remote_ip}, total %{time_total}\n' https://api.anthropic.com/v1/models
```

**注意**：本地 `curl -4 --proxy 127.0.0.1` 主要限制到代理这一段，不能据此声称强制了 VPS 的出站 IPv4。强制 IPv6 解析失败也不能等同于正常双栈访问失败。

没有服务级错误证据时，不为面板显示 IPv6 而禁用 VPS IPv6。若需要变更，核对所装版本的 Freedom/sockopt 字段、备份、仅改相关出站并验证。[Xray Freedom 文档](https://xtls.github.io/config/outbounds/freedom.html)。

## arXiv 直连 vs 代理速度

用同一个 PDF/静态资源、完整 GET、至少三次交替测试，两条路径下载量一致且返回成功。记录状态码、字节数、TTFB、总时间和吞吐；不用 HEAD/ping 代替下载。

```bash
curl --noproxy '*' --connect-timeout 8 --max-time 60 -L -o /dev/null \
  -w 'HTTP %{http_code}, bytes %{size_download}, TTFB %{time_starttransfer}, total %{time_total}, B/s %{speed_download}\n' 'SAME_PDF_URL'
curl --proxy http://127.0.0.1:17891 --connect-timeout 8 --max-time 60 -L -o /dev/null \
  -w 'HTTP %{http_code}, bytes %{size_download}, TTFB %{time_starttransfer}, total %{time_total}, B/s %{speed_download}\n' 'SAME_PDF_URL'
```

代理必须用单独的强制代理 runtime；正式 profile 的 arXiv `DIRECT` 规则可能让 `--proxy :7890` 实际仍直连。终端直连测量前检查系统 TUN/VPN/路由：`--noproxy` 仅绕过 curl 代理设置，不绕过系统隧道。

排除错误页面、内容变化、单次抖动、带宽并发影响；差距小于噪声时报告基本相当，不给“代理永远快/慢”结论。说明本次下载对 VPS 计费流量的影响。结束后只清理自己的临时 runtime。

## Claude / Antigravity / ChatGPT

首页 `200` 只证明该请求成功；无密钥 API `401` 说明接到了认证响应；命令行网页 `403` 可能是 Cloudflare challenge，检查响应头 `cf-mitigated: challenge` 和错误体后再判断。不要归因为地区封禁，也不要承诺账号/模型可用。

在用户使用的真实浏览器/应用中单独验证登录和一次真实模型请求；遇到 Google 登录按钮无反应，查看重定向/弹窗/回调是否发生，不能因客户区首页出现就认为目标产品会话恢复。

账号年龄/关联地区、服务支持地区、IP 位置、验证机制和应用是否经过代理，比“IPv4 还是 IPv6”这一标签更有诊断价值。WARP 不是资格绕过手段，不保证解除限制。

官方材料：[Antigravity FAQ](https://antigravity.google/docs/faq)、[Claude API IP 地址](https://platform.claude.com/docs/en/api/ip-addresses)、[Claude 位置使用说明](https://privacy.claude.com/en/articles/11186740-does-claude-use-my-location)。执行时重查这些动态要求，不把本次观察复制成服务承诺。

## DMIT 流量与运维

用户需要计费周期/剩余额度时，优先让用户在 [DMIT 客户区](https://www.dmit.io/clientarea.php) 登录，进入对应产品详情并读取流量/重置日期。产品、面板和条款不同，不能假定通用 UI 或将客户区根页面当作目标服务页面。

官方面板的计费口径优先；`ip -s link` 是接口累计计数，重启、接口重建和计费方向都会造成差别。可选 `vnstat` 只用于趋势监控，新安装不能回补过去记录；安装/启用监控或定时任务需要单独授权，不因用户问流量就部署它。

故障先查 `systemctl status`、`journalctl -u dmit-edge`、实际监听、时钟和防火墙，再看客户端凭据/证书/SNI/规则。怀疑资源问题时读取内核 OOM 记录及进程 RSS，不未经证据停用别的服务。日志分享时只截取必要诊断信息并脱敏。
