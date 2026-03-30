请执行 `git diff --cached` 命令对比我暂存的文件，有些文件我故意不暂存的，不用担心，也不要加入对比，根据差异：

1. checkout -b 到一个新的分支，分支名的格式为 `hrx/<type>/<short description>`，一定要注意不要忘了这一步。
2. 生成一个详略得当的优雅的英文的 commit message。
3. 执行 `git commit -m "<commit message>"` 命令，提交代码。
4. 执行推送命令，使用以下格式（使用 `<br>` 标签保留换行显示）：
   ```bash
   TITLE=$(git log -1 --format=%s) && BODY=$(git log -1 --format=%b | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/<br>/g') && git push -u origin HEAD -o merge_request.create -o "merge_request.title=$TITLE" -o "merge_request.description=Closes SF-<issue-number><br><br>$BODY"
   ```
   注意：
   - 默认会关闭 issue（包含 `Closes SF-<issue-number>`）
   - 如果用户特别强调不要关闭 issue，则移除 "Closes SF-<issue-number><br><br>" 部分
   - sed 命令将换行符替换为 `<br>` 标签，在 GitLab MR 描述中正确显示

# commit message 模板

```
[SF-<issue-number>] <type>(short description): <long description>

- description 1
- description 2
- ...
```

# 示例

```
[SF-123] feat(unitree_demo): add unitree demo
```