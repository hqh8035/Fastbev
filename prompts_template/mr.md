请执行 `git diff --cached` 命令，并根据差异

1. checkout -b 到一个新的分支，分支名的格式为 `hqh/<type>/<short description>`，一定要注意不要忘了这一步。
2. 生成一个详略得当的优雅的英文的 commit message。
3. 执行 `git commit -m "<commit message>"` 命令，提交代码。
4. 执行 `git push -u origin hqh/<type>/<short description>` 命令，推送代码。

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