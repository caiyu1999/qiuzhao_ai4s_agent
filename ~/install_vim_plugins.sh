#!/bin/bash

echo "正在安装Vim插件..."

# 打开vim并安装插件
vim -c 'PlugInstall' -c 'qa'

echo "插件安装完成！"

# 安装COC的Python语言服务器
echo "正在安装COC Python语言服务器..."
vim -c 'CocInstall coc-python coc-json coc-yaml' -c 'qa'

echo "所有插件安装完成！"
echo ""
echo "==== Vim配置说明 ===="
echo "已配置的插件和功能："
echo "1. 语法高亮和缩进 - python-syntax, vim-python-pep8-indent"
echo "2. 代码补全 - COC.nvim"
echo "3. 语法检查 - ALE (支持flake8, pylint, mypy)"
echo "4. 文件浏览 - NERDTree (Ctrl+n打开/关闭)"
echo "5. 模糊查找 - FZF (Ctrl+p查找文件, Ctrl+f搜索内容)"
echo "6. Git集成 - vim-fugitive, vim-gitgutter"
echo "7. 状态栏 - vim-airline"
echo "8. 主题 - gruvbox"
echo "9. 文档字符串 - vim-pydocstring (Ctrl+d)"
echo "10. 代码注释 - nerdcommenter"
echo "11. 自动括号配对 - auto-pairs"
echo "12. 缩进线显示 - indentLine"
echo ""
echo "==== 常用快捷键 ===="
echo "领导键: ,"
echo ",w - 保存文件"
echo ",q - 退出"
echo ",wq - 保存并退出"
echo ",v - 垂直分屏"
echo ",h - 水平分屏"
echo ",r - 运行当前Python文件"
echo "Ctrl+n - 打开/关闭文件树"
echo "Ctrl+p - 文件搜索"
echo "Ctrl+f - 内容搜索"
echo "Ctrl+d - 生成Python文档字符串"
echo "Tab - 代码补全"
echo "gd - 跳转到定义"
echo "gr - 查找引用"
echo "Space - 折叠/展开代码"
echo ""
echo "现在可以使用 vim 打开Python文件进行编辑了！" 