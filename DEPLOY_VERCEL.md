# stock326.online 部署（同 huazhangbusiness.com：GitHub + Vercel）

## 1) 初始化并推送 GitHub
```bash
cd /Users/echo00726/.openclaw/workspace/a-share-stock-site

git init
git add .
git commit -m "feat: A-share site with volume profile page"
git branch -M main
# 这里替换成你的仓库地址
git remote add origin git@github.com:<your-account>/stock326-site.git
git push -u origin main
```

## 2) Vercel 导入
- 打开 https://vercel.com/new
- 选择 `stock326-site` 仓库
- Framework Preset: Other
- Deploy

## 3) 绑定域名
在 Vercel 项目 Settings -> Domains 添加：
- stock326.online
- www.stock326.online (可选)

按页面提示去域名 DNS 平台添加记录。

## 4) 验证
- https://stock326.online/
- https://stock326.online/volume-profile

## 5) 说明
- 该站依赖行情接口，云端偶发超时属于数据源网络问题。
- `data/*.json` 在无状态平台不是永久存储；需要长期持久化时改数据库。
