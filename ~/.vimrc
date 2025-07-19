" =============================================================================
" Vim配置文件 - Python开发环境
" =============================================================================

" 基本设置
set number                    " 显示行号
set relativenumber           " 显示相对行号
set tabstop=4                " tab键显示为4个空格
set shiftwidth=4             " 自动缩进时使用4个空格
set expandtab                " 将tab转换为空格
set autoindent               " 自动缩进
set smartindent              " 智能缩进
set hlsearch                 " 高亮搜索结果
set incsearch                " 增量搜索
set ignorecase               " 搜索时忽略大小写
set smartcase                " 智能大小写搜索
set showmatch                " 显示匹配的括号
set encoding=utf-8           " 设置编码为UTF-8
set fileencoding=utf-8       " 文件编码
set backspace=indent,eol,start " 设置退格键行为
set ruler                    " 显示光标位置
set showcmd                  " 显示命令
set wildmenu                 " 命令行补全
set laststatus=2             " 总是显示状态栏
set mouse=a                  " 启用鼠标支持
syntax on                    " 语法高亮

" 插件管理 - 使用vim-plug
call plug#begin('~/.vim/plugged')

" Python语法和缩进
Plug 'vim-python/python-syntax'
Plug 'Vimjas/vim-python-pep8-indent'

" 代码补全
Plug 'neoclide/coc.nvim', {'branch': 'release'}

" 语法检查和linting
Plug 'dense-analysis/ale'

" 文件浏览器
Plug 'preservim/nerdtree'
Plug 'Xuyuanp/nerdtree-git-plugin'

" 模糊查找
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

" Git集成
Plug 'tpope/vim-fugitive'
Plug 'airblade/vim-gitgutter'

" 状态栏
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'

" 颜色主题
Plug 'morhetz/gruvbox'
Plug 'dracula/vim', { 'as': 'dracula' }

" Python文档字符串
Plug 'heavenshell/vim-pydocstring', { 'do': 'make install', 'for': 'python' }

" 代码注释
Plug 'preservim/nerdcommenter'

" 自动配对括号
Plug 'jiangmiao/auto-pairs'

" 缩进线显示
Plug 'Yggdroot/indentLine'

" Python虚拟环境
Plug 'jmcantrell/vim-virtualenv'

call plug#end()

" =============================================================================
" 插件配置
" =============================================================================

" Gruvbox主题配置
colorscheme gruvbox
set background=dark

" Python语法高亮增强
let g:python_highlight_all = 1

" NERDTree配置
map <C-n> :NERDTreeToggle<CR>
let NERDTreeShowHidden=1
let NERDTreeIgnore=['\.pyc$', '\~$', '\.git$', '__pycache__']
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif

" ALE语法检查配置
let g:ale_linters = {
\   'python': ['flake8', 'pylint', 'mypy'],
\}
let g:ale_fixers = {
\   'python': ['autopep8', 'black', 'isort'],
\}
let g:ale_fix_on_save = 1
let g:ale_python_flake8_options = '--max-line-length=88'
let g:ale_python_black_options = '--line-length=88'

" FZF配置
map <C-p> :Files<CR>
map <C-f> :Rg<CR>

" COC配置
" 使用Tab键进行补全
inoremap <silent><expr> <TAB>
      \ coc#pum#visible() ? coc#pum#next(1) :
      \ CheckBackspace() ? "\<Tab>" :
      \ coc#refresh()
inoremap <expr><S-TAB> coc#pum#visible() ? coc#pum#prev(1) : "\<C-h>"

function! CheckBackspace() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction

" 使用回车键确认补全
inoremap <silent><expr> <CR> coc#pum#visible() ? coc#pum#confirm()
                              \: "\<C-g>u\<CR>\<c-r>=coc#on_enter()\<CR>"

" GoTo代码跳转
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)

" Airline配置
let g:airline_theme='gruvbox'
let g:airline#extensions#tabline#enabled = 1
let g:airline_powerline_fonts = 1

" IndentLine配置
let g:indentLine_char = '│'
let g:indentLine_color_term = 239

" Pydocstring配置
let g:pydocstring_formatter = 'google'
nmap <silent> <C-d> <Plug>(pydocstring)

" NERDCommenter配置
let g:NERDSpaceDelims = 1
let g:NERDCompactSexyComs = 1

" =============================================================================
" 自定义快捷键
" =============================================================================

" 领导键设置
let mapleader = ","

" 保存和退出
nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>wq :wq<CR>

" 分屏操作
nnoremap <leader>v :vsplit<CR>
nnoremap <leader>h :split<CR>

" 窗口切换
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" 缓冲区切换
nnoremap <leader>n :bnext<CR>
nnoremap <leader>p :bprev<CR>

" 快速编辑vimrc
nnoremap <leader>ev :vsplit $MYVIMRC<CR>
nnoremap <leader>sv :source $MYVIMRC<CR>

" Python特定设置
autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab
autocmd FileType python setlocal foldmethod=indent
autocmd FileType python setlocal foldlevel=99

" =============================================================================
" 其他有用的设置
" =============================================================================

" 记住上次编辑位置
autocmd BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif

" 高亮当前行
set cursorline

" 设置折叠
set foldmethod=indent
set foldlevel=99
nnoremap <space> za

" 删除行尾空格
autocmd BufWritePre *.py :%s/\s\+$//e

" Python运行快捷键
autocmd FileType python nnoremap <leader>r :!python %<CR>
autocmd FileType python nnoremap <leader>R :!python3 %<CR> 