# Vim Python开发环境配置指南

## 配置概述

已为您的vim配置了完整的Python开发环境，包括以下功能：

### 已安装的插件

1. **vim-plug** - 插件管理器
2. **python-syntax** - Python语法高亮增强
3. **vim-python-pep8-indent** - PEP8标准缩进
4. **COC.nvim** - 智能代码补全
5. **ALE** - 异步语法检查和代码格式化
6. **NERDTree** - 文件浏览器
7. **FZF** - 模糊文件查找
8. **vim-fugitive** - Git集成
9. **vim-gitgutter** - Git状态显示
10. **vim-airline** - 状态栏美化
11. **gruvbox** - 颜色主题
12. **vim-pydocstring** - Python文档字符串生成
13. **nerdcommenter** - 代码注释
14. **auto-pairs** - 自动括号配对
15. **indentLine** - 缩进线显示
16. **vim-virtualenv** - Python虚拟环境支持

### 已安装的工具

- **ripgrep** - 快速文本搜索
- **fzf** - 模糊查找工具
- **flake8** - Python代码检查
- **black** - Python代码格式化
- **autopep8** - PEP8代码格式化
- **isort** - import排序
- **pylint** - Python静态分析
- **mypy** - Python类型检查

## 使用方法

### 安装插件

如果插件没有自动安装，可以手动安装：

1. 打开vim
2. 执行 `:PlugInstall`
3. 等待安装完成
4. 重启vim

或者运行安装脚本：
```bash
~/install_vim_plugins.sh
```

### 常用快捷键

#### 基本操作
- **领导键**: `,`（逗号）
- `,w` - 保存文件
- `,q` - 退出
- `,wq` - 保存并退出

#### 分屏操作
- `,v` - 垂直分屏
- `,h` - 水平分屏
- `Ctrl+h/j/k/l` - 在窗口间切换

#### 文件操作
- `Ctrl+n` - 打开/关闭文件树（NERDTree）
- `Ctrl+p` - 文件搜索（FZF）
- `Ctrl+f` - 内容搜索（FZF + ripgrep）

#### Python开发
- `,r` - 运行当前Python文件（python）
- `,R` - 运行当前Python文件（python3）
- `Ctrl+d` - 生成Python文档字符串
- `Tab` - 代码补全
- `gd` - 跳转到定义
- `gr` - 查找引用
- `gi` - 跳转到实现
- `gy` - 跳转到类型定义

#### 代码编辑
- `Space` - 折叠/展开代码
- `gcc` - 注释/取消注释当前行
- `gc` + 移动命令 - 注释选中区域

### 配置文件位置

- **主配置**: `~/.vimrc`
- **COC配置**: `~/.vim/coc-settings.json`
- **简化配置备份**: `~/.vimrc_simple`

### 语法检查和格式化

ALE插件会自动进行：
- **保存时自动格式化**（使用black和isort）
- **实时语法检查**（使用flake8、pylint、mypy）
- **错误高亮显示**

### 代码补全

COC.nvim提供：
- **智能代码补全**
- **函数签名提示**
- **错误诊断**
- **代码跳转**

### 主题和外观

- **颜色主题**: gruvbox（深色）
- **状态栏**: vim-airline
- **缩进线**: 可视化显示
- **行号**: 显示行号和相对行号

## 故障排除

### 如果插件无法安装

1. 检查网络连接
2. 尝试手动克隆插件：
   ```bash
   git clone https://github.com/preservim/nerdtree.git ~/.vim/plugged/nerdtree
   ```

### 如果COC不工作

1. 确保Node.js已安装：`node --version`
2. 手动安装语言服务器：
   ```vim
   :CocInstall coc-python coc-json coc-yaml
   ```

### 如果语法检查不工作

1. 确保Python工具已安装：
   ```bash
   pip install flake8 black autopep8 isort pylint mypy
   ```

### 如果插件配置有问题

可以使用简化版配置：
```bash
cp ~/.vimrc_simple ~/.vimrc
```

## 测试配置

使用提供的测试文件验证配置：
```bash
vim test_vim_config.py
```

在vim中测试：
1. 语法高亮是否正常
2. 按`Ctrl+n`是否打开文件树
3. 按`Tab`是否有代码补全
4. 按`,r`是否能运行Python文件

## 进一步定制

您可以编辑`~/.vimrc`文件来：
- 修改快捷键
- 添加新插件
- 调整主题颜色
- 修改代码格式化规则

## 学习资源

- [Vim基础教程](https://vimschool.netlify.app/)
- [Python-vim配置指南](https://github.com/python-mode/python-mode)
- [COC.nvim文档](https://github.com/neoclide/coc.nvim)

祝您Python开发愉快！🐍✨ 