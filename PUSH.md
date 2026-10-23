# 发布到 GitHub

本目录已经是一个可直接推送的仓库（不含模型权重、不含任何整合包运行时）。

## 0. 先确认里面没有敏感/超大文件

本仓库只有脚本和文档，`checkpoints/`、`python_embeded/`、`outputs/` 都不在这里。
推送前可以再自查一次：

```powershell
Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum |
  ForEach-Object { "{0} 个文件, 共 {1:N1} MB" -f $_.Count, ($_.Sum/1MB) }
```

应当只有几十 KB。

## 1. 装 git（这台机器目前没有）

```powershell
winget install --id Git.Git -e --source winget
```

装完**新开一个终端**（PATH 才会刷新），用 `git --version` 确认。

## 2. 在 GitHub 上建一个空仓库

浏览器打开 <https://github.com/new>：

- Repository name：例如 `indextts25-lowvram`
- 可见性随意（想让人用就是 Public）
- **不要**勾选 "Add a README file"、".gitignore"、"license" —— 本地已经有了，勾了会冲突

建好后复制仓库地址，形如 `https://github.com/<你的用户名>/indextts25-lowvram.git`。

## 3. 推送

```powershell
cd "D:\2Z\IndexTTS25-LowVRAM"

git init
git add .
git commit -m "IndexTTS-2.5 low-VRAM patch: fp16 + stage swapping for 6GB GPUs"

git branch -M main
git remote add origin https://github.com/<你的用户名>/indextts25-lowvram.git
git push -u origin main
```

首次推送会弹浏览器让你登录 GitHub（Git Credential Manager），按提示授权即可；
或者在 GitHub 上生成 Personal Access Token 当密码用。

## 4. 推完建议

- 仓库描述写一句：`6GB 显卡跑 IndexTTS-2.5：按算力选精度 + 分阶段换驻，RTX 2060 实测 3.5x`
- 在 About 里加上 Topics：`indextts` `tts` `low-vram` `fp16` `pytorch`
- 确认 `LICENSE`、`LICENSE_ZH.txt`、`NOTICE` 三个文件都在（许可协议 3.4(b) 要求）

## 合规提醒

- 本仓库是**衍生品补丁**，不是模型分发。不要往里放 `checkpoints/`（有 6.7 GB 权重，且 GitHub 单文件上限 100 MB）。
- `NOTICE` 里已按许可协议 4.1(a) 写好免责声明，**不要删**。
- 不要用 bilibili / IndexTTS 的图标或名称做仓库头像、Logo（许可协议 5.2 限制商标使用）。
