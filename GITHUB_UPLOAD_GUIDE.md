# GitHub上传指南

## 方式1: 使用自动化脚本（推荐）

直接双击运行：
```
upload_to_github.bat
```

脚本会自动完成以下步骤：
1. 配置Git用户信息
2. 添加所有文件
3. 创建初始提交
4. 检查GitHub CLI
5. 创建GitHub仓库并推送

---

## 方式2: 手动上传（如果脚本失败）

### 前置条件

1. **安装GitHub CLI**（如果没有）
   - 访问：https://cli.github.com/
   - 下载并安装
   - 运行登录：`gh auth login`

### 步骤1: 配置Git

```bash
# 设置用户名和邮箱（替换为您的信息）
git config user.name "你的GitHub用户名"
git config user.email "你的邮箱@example.com"
```

### 步骤2: 添加文件并提交

```bash
# 查看将要提交的文件
git status

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial commit: Rock discontinuity analysis system with HDBSCAN"
```

### 步骤3: 使用GitHub CLI创建仓库并推送

```bash
# 创建公开仓库并推送
gh repo create rock-discontinuity-analysis --public --source=. --remote=origin --push

# 或者创建私有仓库
gh repo create rock-discontinuity-analysis --private --source=. --remote=origin --push
```

---

## 方式3: 通过GitHub网页创建仓库（不使用CLI）

### 步骤1: 在GitHub网页创建仓库

1. 访问：https://github.com/new
2. 仓库名称：`rock-discontinuity-analysis`
3. 描述：`Rock discontinuity analysis system with advanced algorithms`
4. 选择 Public 或 Private
5. **不要**勾选"Initialize with README"
6. 点击"Create repository"

### 步骤2: 配置Git并推送

```bash
# 配置Git用户信息
git config user.name "你的GitHub用户名"
git config user.email "你的邮箱@example.com"

# 添加所有文件
git add .

# 创建提交
git commit -m "Initial commit: Rock discontinuity analysis system with HDBSCAN"

# 添加远程仓库（替换YOUR_USERNAME为您的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/rock-discontinuity-analysis.git

# 重命名分支为main
git branch -M main

# 推送到GitHub
git push -u origin main
```

---

## 后续更新代码

完成首次上传后，后续更新代码只需：

```bash
# 添加更改
git add .

# 提交更改
git commit -m "更新说明"

# 推送到GitHub
git push
```

---

## 常见问题

### Q: GitHub CLI未安装怎么办？
A: 使用"方式3"通过网页创建仓库，或者访问 https://cli.github.com/ 安装GitHub CLI。

### Q: 推送时提示认证失败？
A: 运行 `gh auth login` 重新登录GitHub。

### Q: 文件太大无法推送？
A: 检查 `.gitignore` 文件，确保大型数据文件被排除。常见的大文件：
- `data-files/` 目录下的点云文件
- `output*/` 目录
- `.ply`, `.pcd` 等点云数据文件

### Q: 如何检查哪些文件会被上传？
A: 运行 `git status` 查看将要提交的文件。

---

## 检查清单

上传前确保：
- ✅ `.gitignore` 文件已创建（排除大文件和敏感信息）
- ✅ `README.md` 文件已创建
- ✅ Git用户信息已配置
- ✅ 大型数据文件不会被上传（检查 `data-files/`, `output*/` 是否在 `.gitignore` 中）
- ✅ 敏感信息已移除（API密钥、密码等）

---

## 推荐的仓库设置

- **仓库名称**: `rock-discontinuity-analysis`
- **描述**: `Rock discontinuity analysis system with HDBSCAN, MAGSAC++, and Mean-Shift algorithms`
- **可见性**: Public（如果要分享）或 Private（私人项目）
- **主题标签**: `point-cloud`, `hdbscan`, `rock-mechanics`, `discontinuity-detection`

---

需要帮助？查看 [GitHub文档](https://docs.github.com/) 或运行脚本时的提示信息。
