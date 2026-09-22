# 上传指南（路径 A：GitHub + Colab 徽章）

整个包 ≈ 25 MB（数据 12 MB + 执行过的 notebooks），全部可进一个 GitHub 仓库，
学生一键 "Open in Colab"，无需任何额外下载。

## 步骤（约 10 分钟，只需做一次）

1. **建仓库**：github.com → New repository → 名字必须是 `IUM_teaching_colab`
   （notebooks 的 clone 命令写死了这个名字）→ Public → 不勾选任何初始化文件。
2. **上传**：在本文件夹打开终端执行：
   ```bash
   cd C:\Users\wangy\Desktop\IUM_teaching_colab
   git init
   git add .
   git commit -m "IUM teaching series v1"
   git branch -M main
   git remote add origin https://github.com/wangyiquan1010-coder/IUM_teaching_colab.git
   git push -u origin main
   ```
   （或者用 GitHub Desktop 拖进去，一样的。）
3. ~~替换用户名占位符~~ **已完成**：README 徽章和 4 个 notebook 的 clone 行
   已全部指向 `wangyiquan1010-coder/IUM_teaching_colab`。
4. **验证**：打开 README 里的 Notebook 1 徽章 → Colab 打开 → Runtime → Run all
   → 第一个 cell 会自动 clone 仓库并切换目录，全程应无报错。
   Notebook 3 建议先 Runtime → Change runtime type → **T4 GPU**（训练 ~4 分钟；
   纯 CPU 约 20–30 分钟，也能跑）。Notebook 1、2、4 都在一分钟以内。

## 课堂使用建议

- 学生不需要 GitHub 账号：点徽章 → File → Save a copy in Drive 即可保存自己的副本。
- 改动 notebooks 后重新 push，学生下次打开自动是新版（Colab 每次从 GitHub 取最新）。
- 若以后想收作业：让学生提交 "Share" 链接或下载的 .ipynb。

## 注意

- 本文件夹是独立副本，原始代码与数据（`sharefolder\closeloopcontrol\IUM feature
  extract\`、`Desktop\IUM\IUM_50samples\`、`IUM数据与代码\`）未做任何改动。
- `tools/` 只是维护脚本（重新提取数据 / 重新生成 notebooks 用），学生用不到；
  不想公开可在 push 前删掉，不影响任何 notebook。
