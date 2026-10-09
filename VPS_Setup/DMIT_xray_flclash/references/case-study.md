# 本次过程记录（脱敏）

范围：2026-09-27 部署及随后国内/arXiv 分流、出口诊断；整理日期 2026-10-09。真实主机、用户路径、凭据、证书及订阅不公开。以下是历史记录，不是对未来环境的保证。

## 部署选择与落地

- 保留旧 VPS/CPA，单独配置 DMIT 主机。
- Xray `26.3.27` 同时提供 HY2/UDP 443 和 VLESS + Vision + REALITY/TCP 443；专用 `dmit-edge` 服务账户、服务名和配置目录。
- WARP `2026.7.1377` 仅作为 `127.0.0.1:40000` SOCKS 出口，不接管 VPS 默认路由；不是终端设备安装全局 WARP。
- 新代理只额外开放需要的 TCP/UDP 端口，保留 SSH。没有重启整台 VPS。
- 四台设备 × 原生/WARP 各自独立凭据；HY2 使用私有证书 pin，sing-box 显式信任证书，没有关闭校验。
- HY2 示例 `users` 与所装 Xray 版本不一致，改为该版本支持的 `clients`；REALITY 目标排查后同步服务端和客户端参数。

## 多平台配置与后续更新

- Mac/Windows/Android 交付 FlClash/Mihomo；iPhone 交付 sing-box JSON，另外保留 Clash 格式备用文件。
- 原生 HY2 和 REALITY 放在 `PROXY`，`AI` 默认跟随 PROXY，WARP 仅手动备用。
- 后续加入国内域名规则库和抖音/豆包等关键域名补丁；同时调整直连 DoH、外网代理 DoH 与 Tailscale 系统解析。
- 2026-10-08 为四份 Clash profile 和两份 sing-box 正式/测试配置加 arXiv 直连与相应 DNS，先备份，再应用 Mac 当前 profile。
- Mac 活动连接观察到 arXiv 命中 `DomainSuffix(arxiv.org) → DIRECT`；AI 连接仍走 AI/PROXY。源文件改动、运行内核应用和连接验收分开记录。

## 实测范围与结论

- Mihomo `1.19.31` 独立测试过 HY2/REALITY × 原生/WARP 四组合；trace 显示原生 `warp=off`、WARP `warp=on`。
- 错误 HY2 证书 pin 被拒绝。此记录不声称已做过全部错误密码/断开 WARP 负向测试；新部署应补充。
- iPhone 配置在本机 sing-box `1.14.2` 内核检查和连接测试，不能据此宣称真实 iPhone 设备或 VPN 权限已经验收。
- 同一个约 2.22 MB arXiv PDF，每条路径三次：直连均值约 `1.9405 s`，强制代理约 `1.9470 s`，差距不足 1%。保留 DIRECT 是本次环境选择，不是普遍性能结论。
- IPv4-only 检测返回 DMIT IPv4，双栈检测返回 DMIT 网卡的原生 IPv6，trace `warp=off`。客户端禁止 IPv6 与服务器出站 IPv6 可以并存，不是 WARP 或本机 IPv6 泄漏的证据。
- 后续 Claude API 在 VPS IPv4/IPv6 均返回无密钥 `401`，网页均为 `403` challenge；Antigravity 官网 IPv4 `200`，该次强制 IPv6未能解析。没有证据仅因面板显示 IPv6 就需要改配置。

## 未据此宣称完成

Windows/手机文件传输不等于 active profile 生效；所有设备的 TUN/VPN、所有 DNS 路径、账号登录和真实模型请求需要各自验收。CLI 首页响应不证明 ChatGPT、Claude 或 Antigravity 账号可用，WARP 不保证地区解锁。

历史二进制版本、内存样本和下载速度不是当前推荐或持续运行指标。复用时重新检查官方版本、目标主机和真实客户端网络。
