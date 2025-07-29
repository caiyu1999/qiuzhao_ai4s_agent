"""
使用Rich库的Layout和Live类演示一个完整的终端"应用程序"。
这个例子展示了如何创建一个复杂的终端界面，包含多个面板、进度条和实时更新。
"""

from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# 创建控制台对象，用于输出到终端
console = Console()


def make_layout() -> Layout:
    """
    定义整个应用程序的布局结构。
    Layout是Rich中用于分割屏幕空间的组件，类似于网页中的div布局。
    
    返回:
        Layout: 配置好的布局对象
    """
    # 创建根布局
    layout = Layout(name="root")

    # 将根布局垂直分割为三部分：
    # - header: 顶部区域，固定高度3行
    # - main: 主体区域，占据剩余空间的比例为1
    # - footer: 底部区域，固定高度7行
    layout.split(
        Layout(name="header", size=3),      # 顶部：标题栏
        Layout(name="main", ratio=1),       # 中间：主要内容区
        Layout(name="footer", size=7),      # 底部：进度条区域
    )
    
    # 将主体区域水平分割为两部分：
    # - side: 左侧边栏
    # - body: 右侧主体，占据2倍空间，最小宽度60字符
    layout["main"].split_row(
        Layout(name="side"),                # 左侧边栏
        Layout(name="body", ratio=2, minimum_size=60),  # 右侧主体
    )
    
    # 将左侧边栏垂直分割为两个盒子
    layout["side"].split(
        Layout(name="box1"),    # 上方盒子：显示布局树
        Layout(name="box2")     # 下方盒子：显示代码语法高亮
    )
    
    return layout


def make_sponsor_message() -> Panel:
    """
    创建一个包含赞助商信息的面板。
    这个函数演示如何使用Table.grid创建无边框表格，以及如何添加链接。
    
    返回:
        Panel: 包含赞助商信息的面板
    """
    # 创建一个网格表格（无边框），用于显示联系信息
    sponsor_message = Table.grid(padding=1)  # padding=1 表示每个单元格内边距为1
    sponsor_message.add_column(style="green", justify="right")  # 第一列：绿色，右对齐
    sponsor_message.add_column(no_wrap=True)  # 第二列：不换行
    
    # 添加行数据，第二列包含可点击的链接
    sponsor_message.add_row(
        "Twitter",
        "[u blue link=https://twitter.com/textualize]https://twitter.com/textualize",
    )
    sponsor_message.add_row(
        "CEO",
        "[u blue link=https://twitter.com/willmcgugan]https://twitter.com/willmcgugan",
    )
    sponsor_message.add_row(
        "Textualize", "[u blue link=https://www.textualize.io]https://www.textualize.io"
    )

    # 创建另一个网格表格作为容器
    message = Table.grid(padding=1)
    message.add_column()
    message.add_column(no_wrap=True)
    message.add_row(sponsor_message)

    # 将表格包装在一个面板中，并设置样式
    message_panel = Panel(
        Align.center(
            Group("\n", Align.center(sponsor_message)),  # Group用于组合多个元素
            vertical="middle",  # 垂直居中对齐
        ),
        box=box.ROUNDED,        # 使用圆角边框
        padding=(1, 2),         # 内边距：垂直1，水平2
        title="[b red]Thanks for trying out Rich!",  # 面板标题，粗体红色
        border_style="bright_blue",  # 边框颜色
    )
    return message_panel


class Header:
    """
    显示带有时钟的头部组件。
    这个类实现了__rich__方法，使其可以直接被Rich渲染。
    """

    def __rich__(self) -> Panel:
        """
        Rich渲染方法。当这个对象被Rich渲染时，会调用这个方法。
        
        返回:
            Panel: 包含标题和时间的面板
        """
        # 创建一个可扩展的网格表格
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)  # 第一列：居中，占据剩余空间
        grid.add_column(justify="right")            # 第二列：右对齐
        
        # 添加一行数据：应用标题和当前时间
        grid.add_row(
            "[b]Rich[/b] Layout application",  # 粗体标题
            datetime.now().ctime().replace(":", "[blink]:[/]"),  # 时间，冒号闪烁效果
        )
        
        # 返回一个蓝底白字的面板
        return Panel(grid, style="white on blue")


def make_syntax() -> Syntax:
    """
    创建一个代码语法高亮显示组件。
    Syntax组件可以对各种编程语言的代码进行语法高亮。
    
    返回:
        Syntax: 语法高亮的代码块
    """
    # 示例Python代码
    code = """\
def ratio_resolve(total: int, edges: List[Edge]) -> List[int]:
    sizes = [(edge.size or None) for edge in edges]

    # While any edges haven't been calculated
    while any(size is None for size in sizes):
        # Get flexible edges and index to map these back on to sizes list
        flexible_edges = [
            (index, edge)
            for index, (size, edge) in enumerate(zip(sizes, edges))
            if size is None
        ]
        # Remaining space in total
        remaining = total - sum(size or 0 for size in sizes)
        if remaining <= 0:
            # No room for flexible edges
            sizes[:] = [(size or 0) for size in sizes]
            break
        # Calculate number of characters in a ratio portion
        portion = remaining / sum((edge.ratio or 1) for _, edge in flexible_edges)

        # If any edges will be less than their minimum, replace size with the minimum
        for index, edge in flexible_edges:
            if portion * edge.ratio <= edge.minimum_size:
                sizes[index] = edge.minimum_size
                break
        else:
            # Distribute flexible space and compensate for rounding error
            # Since edge sizes can only be integers we need to add the remainder
            # to the following line
            _modf = modf
            remainder = 0.0
            for index, edge in flexible_edges:
                remainder, size = _modf(portion * edge.ratio + remainder)
                sizes[index] = int(size)
            break
    # Sizes now contains integers only
    return cast(List[int], sizes)
    """
    # 创建语法高亮对象：Python语言，显示行号
    syntax = Syntax(code, "python", line_numbers=True)
    return syntax


# ==================== 进度条设置 ====================

# 创建任务进度条，包含多个列：
# - 任务描述
# - 旋转器（显示任务正在进行）
# - 进度条
# - 百分比文本
job_progress = Progress(
    "{task.description}",           # 显示任务描述
    SpinnerColumn(),               # 旋转指示器
    BarColumn(),                   # 进度条
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),  # 百分比，右对齐，0位小数
)

# 添加三个示例任务
job_progress.add_task("[green]Cooking")        # 绿色任务，无总数（无限进度）
job_progress.add_task("[magenta]Baking", total=200)   # 紫色任务，总数200
job_progress.add_task("[cyan]Mixing", total=400)      # 青色任务，总数400

# 计算所有任务的总工作量
total = sum(task.total for task in job_progress.tasks)

# 创建总体进度条
overall_progress = Progress()
overall_task = overall_progress.add_task("All Jobs", total=int(total))

# 创建进度条表格，将总体进度和任务进度并排显示
progress_table = Table.grid(expand=True)
progress_table.add_row(
    # 总体进度面板
    Panel(
        overall_progress,
        title="Overall Progress",       # 面板标题
        border_style="green",          # 绿色边框
        padding=(2, 2),               # 内边距
    ),
    # 任务进度面板
    Panel(
        job_progress, 
        title="[b]Jobs",              # 粗体标题
        border_style="red",           # 红色边框
        padding=(1, 2)               # 内边距
    ),
)

# ==================== 布局组装 ====================

# 创建布局并填充各个区域
layout = make_layout()
layout["header"].update(Header())                    # 头部：时钟和标题
layout["body"].update(make_sponsor_message())        # 主体：赞助商信息
layout["box2"].update(Panel(make_syntax(), border_style="green"))  # 左下：代码高亮
layout["box1"].update(Panel(layout.tree, border_style="red"))      # 左上：布局树
layout["footer"].update(progress_table)              # 底部：进度条

# ==================== 实时显示 ====================

from time import sleep
from rich.live import Live

# 使用Live类创建实时更新的显示
# - refresh_per_second=10: 每秒刷新10次
# - screen=True: 使用全屏模式（清除屏幕内容）
with Live(layout, refresh_per_second=10, screen=True):
    # 主循环：持续更新进度直到所有任务完成
    while not overall_progress.finished:
        sleep(0.1)  # 每100毫秒更新一次
        
        # 更新每个任务的进度
        for job in job_progress.tasks:
            if not job.finished:
                job_progress.advance(job.id)  # 推进任务进度
        
        # 计算总完成量并更新总体进度
        completed = sum(task.completed for task in job_progress.tasks)
        overall_progress.update(overall_task, completed=completed)