---
name: dmit-vps-proxy
description: Configure or maintain a DMIT VPS proxy using Xray Hysteria2 and VLESS/REALITY, optional server-side WARP SOCKS, and FlClash/Mihomo or sing-box clients. Use for multi-device profile generation, China/arXiv direct routing, active-profile verification, traffic accounting, and IPv4/IPv6 exit diagnosis in this setup.
---

# DMIT VPS：HY2 / REALITY / FlClash

把 DMIT VPS 配置成私人代理，保留原有 SSH、服务和客户端配置。主路径为设备 → DMIT → 网站；可选路径为设备 → DMIT → 本机 WARP SOCKS → 网站。面向 Mac、Windows、Android 的 FlClash/Mihomo 和 iPhone 的 sing-box。

本目录名按仓库的 `DMIT_xxx` 分类约定保留；技能名为 `dmit-vps-proxy`。安装到技能目录时可将整个目录复制为 `dmit-vps-proxy`。此技能是部署流程与复用工具，不会自动安装或修改任何服务器。

## 操作边界

- 先只读检查，再报告目标主机、改动路径、端口、影响和回滚方案；获得用户对部署/配置变更的授权后才操作。诊断问题不意味着获准实施修复。
- 区分本次 DMIT 主机与其他 VPS、旧代理、CPA 等服务；不要因旧会话的 IP/SSH 别名而改错主机。
- 保留现有配置和当前选中的客户端 profile；不要重置防火墙、覆盖其他服务或自动启用 TUN。
- 真实 IP、UUID、HY2 密码、REALITY 私钥、订阅、证书私钥和账号信息都不进入公开仓库、聊天正文或第三方订阅转换站。每台设备和原生/WARP 路径分别生成凭据。
- 公共资源只有说明、脚本和无凭据模板。生成文件放在仓库外，目录 `0700`、文件 `0600`；服务端配置/私钥按服务账户需要设置 `root:dmit-edge 0640`。
- WARP 是可选备用出口，不是默认全机 VPN，也不承诺解决地区、账号资格或风控问题。

## 按任务读取

- 新部署、升级、证书/凭据轮换、回滚：读 [服务端流程](references/server.md)。
- 生成/更新所有平台配置、国内直连、arXiv 或 AI 分流：读 [客户端流程](references/clients.md)。
- 连接失败、IPv6 出口、速度比较、流量查看、登录异常：读 [验收与排查](references/validation.md)。
- 用户问“这次具体做了什么”：读 [脱敏过程记录](references/case-study.md)，明确历史实测与当前状态的区别。

## 关键决策

1. 根据客户端实际网络选择主节点：UDP 稳定可用 HY2；UDP 不稳定时比较 REALITY/TCP。不把某一次速度结果变成通用优先级。
2. 同一个端口号可分别用于 TCP 的 REALITY 和 UDP 的 HY2，但必须分别检查两个协议的占用及防火墙。
3. 无自有域名时，HY2 可使用私有证书加客户端 pin/显式信任。不得用关闭证书验证来“修复”握手。
4. WARP SOCKS 只承载本方案的 TCP 目标请求；WARP 凭据的 UDP 在服务端显式阻止。失效时失败，不静默回落原生出口。
5. 客户端 `ipv6: false` 不限制 DMIT → 网站这一段；面板显示 IPv6 不等于本机 IPv6 泄漏，也不等于每个目的站都走 IPv6。
6. 国内域名规则、直连 DNS、IP 规则需配套，且特殊规则置于兜底规则前。arXiv 直连属于这次实测后的选择，换网络应复测。

## 工作流

### 1. 发现现状

确认 SSH 主机指纹、系统/架构、内存/磁盘/时钟、TCP/UDP 监听、SSH 实际端口、防火墙、既有 Xray/WARP。确认目标设备的客户端和内核版本，找出源配置、已导入副本及当前运行配置。

凭据、授权、端口或兼容性不明时，继续安全的只读检查；不要覆盖现有服务以求推进。端口有冲突就报告并等待端口/架构选择。

### 2. 服务端

按服务端流程安装经过摘要校验的官方发行版；使用专用账户、独立服务名和目录。生成新凭据，测试证书/REALITY 目标，验证配置后再启动。只有明确需要备用出口时才配置 WARP 本机代理。

特别注意版本差异：历史部署的 Xray `26.3.27` HY2 入站用 `settings.clients`；在线文档可能写 `users`。按目标版本的源码/文档核对，不能只因 JSON 解析通过就认定认证生效。

### 3. 客户端与同步

生成新的私有客户端资料包后，用可复用渲染器生成所有平台：

```bash
python3 scripts/render_profiles.py \
  --access /private/path/access.json \
  --output /private/path/new-client-bundle
```

输出四份 Clash-compatible `.yaml`（内容为 JSON，YAML-compatible）和一份 iPhone sing-box JSON。脚本不安装客户端、不传输文件、不修改既有配置；拒绝写入 Git 仓库及覆盖已有目录。按客户端流程校验、备份、导入和应用。

更新现有配置时，不把历史凭据文件、临时测试配置或已备份副本当作当前配置。跨平台修改必须逐个平台核对路由和 DNS 语义，而不是机械替换字段。

### 4. 分层验收

分别报告以下证据：

- 配置语法/目标内核检查通过。
- 服务账户可读配置，服务 active/enabled，TCP/UDP 端口和 WARP 本机监听符合设计。
- 从真实客户端网络通过正确凭据连接；错误密码/证书 pin 被拒绝。
- 原生/WARP 出口分别验证，WARP 失效不会静默走原生。
- 选中的 active profile 已应用，实际连接命中预期规则。
- 浏览器登录和真实模型调用仅在实际观察后才算通过；首页 `200`、无密钥 API `401`、Cloudflare 挑战 `403` 均不能替代它们。

临时客户端使用独立数据目录、空闲端口、仅 loopback 的控制接口；记录自己启动的 PID，只结束该测试进程，不停止正式客户端。

### 5. 交付

提供私有配置的本地位置、已改路径、备份、内核版本、节点选择、测试结果和未验证项。说明文件传输不等于导入成功，本机内核检查不等于 iPhone/Windows 设备验收。无需为此任务重启整台 VPS。

本技能的官方参考集中在各流程末尾；执行时重新核对发行版、字段和服务支持地区。不要把历史版本号当成“当前最新”。

## 维护本技能时验证

```bash
python3 scripts/test_render_profiles.py -v
```

测试只生成临时合成凭据，需本机 OpenSSL，不连接真实 VPS。设置 `DMIT_TEST_MIHOMO=/path/to/mihomo` 和 `DMIT_TEST_SINGBOX=/path/to/sing-box` 可额外检查实际内核；Mihomo 可能下载规则/GeoIP 数据，输出仍在隔离临时目录。未设置时这两项明确跳过，不能声称通过内核验收。修改技能后还应运行可用的 skill-creator `quick_validate.py` 并检查公开差异没有真实凭据。
