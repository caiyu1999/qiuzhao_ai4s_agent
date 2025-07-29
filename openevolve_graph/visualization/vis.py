from rich.layout import Layout 
from rich import print 
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from datetime import datetime
from openevolve_graph.Config.config import Config 
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from openevolve_graph.visualization.vis_class import IslandData_vis, best_program_vis, overall_information_vis, visualize_data
from rich.syntax import Syntax
from openevolve_graph.visualization.socket_sc import SimpleServer
from typing import Optional, List, cast
import threading
import time
from math import modf

def make_layout(config: Config) -> Layout:
    """
    定义整个应用程序的布局结构。
    Layout是Rich中用于分割屏幕空间的组件，类似于网页中的div布局。
    
    返回:
        Layout: 配置好的布局对象
    """
    # 创建根布局
    layout = Layout(name="root")
    # 进行布局分割 
    
    layout.split(
        Layout(name="region_1", size=6),      # 顶部：标题栏
        Layout(name="region_2"), #下方为主要信息区域
    )
    
    layout["region_1"].split_column(
        Layout(name="header",ratio=1),
        Layout(name="iterations_all",ratio=1), # 进度条 = 当前iteration/max_iterations 并预估完成时间
    )
    

    layout["region_2"].split_row(
        Layout(name="left_region",ratio= 3), 
        Layout(name="best program",minimum_size=60), 
        
    )
    
    # 左侧区域分为岛屿区域和总体信息区域
    layout["left_region"].split_column(
        Layout(name="islands_region", ratio=8),
        Layout(name="all_information", ratio=2)
    )
    
    # 根据岛屿数量动态创建岛屿布局
    num_islands = config.island.num_islands
    if num_islands > 0:
        # 创建岛屿布局 - 根据岛屿数量进行垂直分割
        layout["islands_region"].split_column(
            *[Layout(name=f"island_{i}", ratio=1) for i in range(num_islands)]
        )
    
    layout["best program"].split_column(
        Layout(name="code",ratio= 8), #这里展示bestprogram的代码 
        Layout(name="information",ratio=2), # 这里展示bestprogram的信息
    )
    
    return layout




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
        
        # 添加一行数据：应用标题和当前时间（红色字体）
        grid.add_row(
            "[bold italic red]OpenEvolve[/bold italic red] v0.1.0",  # 粗体斜体红色标题
            f"[italic red]{datetime.now().ctime().replace(':', '[blink]:[/]')}[/italic red]",  # 斜体红色时间，冒号闪烁
        )
        # 返回一个红色边框、黑底红字的面板
        return Panel(grid, border_style="red", style="red on black")

def make_syntax(code:str, language:str="python") -> Syntax:
    
    """
    创建一个代码语法高亮显示组件。
    Syntax组件可以对各种编程语言的代码进行语法高亮。
    
    返回:
        Syntax: 语法高亮的代码块
    """
    # 创建语法高亮对象：Python语言，显示行号
    syntax = Syntax(code, language, line_numbers=False)
    return syntax


def create_island_panel(island_data: IslandData_vis) -> Panel:
    """为单个岛屿创建信息面板"""
    table = Table(show_header=False, box=None)
    table.add_column("属性", style="cyan", width=15)
    table.add_column("值", style="white", width=30)
    
    # 添加岛屿基本信息
    table.add_row("状态", island_data.status if isinstance(island_data.status,str) else str(island_data.status))
    table.add_row("迭代次数", str(island_data.iterations))
    table.add_row("程序数量", str(island_data.num_programs))
    
    
    # 添加会议信息
    meeting_progress = f"{island_data.now_meeting}/{island_data.next_meeting}"
    table.add_row("会议进度", meeting_progress)
    
    table.add_row("最新程序ID", island_data.latest_program_id[:15] + "..." if len(island_data.latest_program_id) > 15 else island_data.latest_program_id)
    table.add_row("父代程序ID", island_data.sample_program_id[:15] + "..." if len(island_data.sample_program_id) > 15 else island_data.sample_program_id)
    table.add_row("最佳程序ID", island_data.best_program_id[:15] + "..." if len(island_data.best_program_id) > 15 else island_data.best_program_id)
    table.add_row("最佳指标", str(island_data.best_program_metrics))
    table.add_row("提示词", island_data.prompt[:40] + "..." if len(island_data.prompt) > 40 else island_data.prompt)
    
    return Panel(table, title=f"岛屿 {island_data.id}", border_style="blue")


def create_island_table(islands_data: dict[str, IslandData_vis]) -> Table:
    """创建岛屿信息表格（保留用于兼容性）"""
    table = Table(show_header=True, header_style="bold magenta")
    
    table.add_column("岛屿", style="cyan", width=8)
    table.add_column("状态", style="green", width=12)
    table.add_column("迭代", style="yellow", width=8)
    table.add_column("程序数", style="blue", width=8)
    table.add_column("最新程序", style="white", width=15)
    table.add_column("最佳指标", style="red", width=15)
    
    for island_id, island_data in islands_data.items():
        # 格式化最佳指标
        metrics_str = ""
        if island_data.best_program_metrics:
            first_metric = list(island_data.best_program_metrics.values())[0]
            metrics_str = f"{first_metric:.3f}"
        
        table.add_row(
            f"Island_{island_id}",
            island_data.status,
            str(island_data.iterations),
            str(island_data.num_programs),
            island_data.latest_program_id[:12] + "..." if len(island_data.latest_program_id) > 12 else island_data.latest_program_id,
            metrics_str
        )
    
    return table


def create_best_program_info(best_program: best_program_vis) -> Table:
    """创建最佳程序信息表格"""
    table = Table(show_header=False, box=None)
    table.add_column("属性", style="cyan")
    table.add_column("值", style="white")
    
    table.add_row("程序ID", best_program.id[:20] + "..." if len(best_program.id) > 20 else best_program.id)
    if best_program.sample_program_id:
        parent_id = best_program.sample_program_id[:20] + "..." if len(best_program.sample_program_id) > 20 else best_program.sample_program_id
        table.add_row("父代程序ID", parent_id)
    table.add_row("来源岛屿", best_program.from_island)
    table.add_row("发现轮次", str(best_program.iteration_found))
    table.add_row("复杂度", f"{best_program.complexity:.3f}")
    table.add_row("多样性", f"{best_program.diversity:.3f}")
    
    # 添加指标
    for metric_name, metric_value in best_program.metrics.items():
        table.add_row(metric_name, f"{metric_value:.3f}")
    
    return table


class VisualizationApp:
    """实时可视化应用程序"""
    
    def __init__(self, config: Config, server: Optional[SimpleServer]):
        self.config = config
        self.server = server
        self.layout = make_layout(config)
        self.running = False
        
        # 初始化组件
        self.header = Header()
        self.overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        self.overall_task = self.overall_progress.add_task("Evolution Progress", total=config.max_iterations)
        
        # 设置静态组件
        self.layout["header"].update(self.header)
        self.layout["iterations_all"].update(self.overall_progress)
        
    def update_display(self):
        """更新显示内容"""
        try:
            # 获取最新的可视化数据
            if self.server is None:
                return
            vis_data = self.server.get_vis_data()
            
            # 更新各个岛屿的信息面板
            for i in range(self.config.island.num_islands):
                island_id = str(i)
                if island_id in vis_data.islands_data:
                    island_data = vis_data.islands_data[island_id]
                    island_panel = create_island_panel(island_data)
                    self.layout[f"island_{i}"].update(island_panel)

            # 更新最佳程序代码
            if vis_data.best_program and vis_data.best_program.code:
                code_syntax = make_syntax(vis_data.best_program.code, "python")
                self.layout["code"].update(Panel(code_syntax, title="最佳程序代码", border_style="green"))
            
            # 更新最佳程序信息
            if vis_data.best_program and vis_data.best_program.id:
                best_info_table = create_best_program_info(vis_data.best_program)
                self.layout["information"].update(Panel(best_info_table, title="最佳程序信息", border_style="green"))
            
            # 更新总体信息
            overall_table = Table(show_header=False, box=None)
            overall_table.add_column("属性", style="cyan")
            overall_table.add_column("值", style="white")
            overall_table.add_row("总程序数", str(vis_data.overall_information.num_programs))
            overall_table.add_row("会议次数", str(vis_data.overall_information.num_meetings))
            self.layout["all_information"].update(Panel(overall_table, title="总体信息", border_style="yellow"))
            
            # 更新进度条
            if vis_data.islands_data:
                current_iteration = max(island.iterations for island in vis_data.islands_data.values()) if vis_data.islands_data else 0
                self.overall_progress.update(self.overall_task, completed=current_iteration)
            
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def run(self):
        """运行可视化应用"""
        self.running = True
        
        with Live(self.layout, refresh_per_second=4, screen=True, transient=True) as live:
            while self.running:
                self.update_display()
                time.sleep(0.25)
    
    def stop(self):
        """停止可视化应用"""
        self.running = False


def start_visualization(config: Config, server: SimpleServer):
    """启动可视化界面"""
    app = VisualizationApp(config, server)
    return app

if __name__ == "__main__":
    # This part is for testing the layout and components directly
    # It won't use the server and will not reflect real-time updates
    try:
        config = Config.from_yaml("/Users/caiyu/Desktop/langchain/openevolve_graph/openevolve_graph/test/test_config.yaml")
        app = VisualizationApp(config, None) # Pass None for server as we are not using it
        app.run()
    except KeyboardInterrupt:
        print("Visualization stopped by user.")