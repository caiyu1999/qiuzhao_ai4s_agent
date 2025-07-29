"""
OpenEvolve项目的Rich监控界面示例

这个示例展示如何使用Rich库为OpenEvolve项目创建一个
美观的实时监控界面，包括多岛屿进度、最佳程序信息等。
"""

import time
import random
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich import box
from rich.align import Align
from rich.text import Text

# ==================== 数据模型 ====================

@dataclass
class ProgramInfo:
    """程序信息数据类"""
    id: str
    generation: int
    iteration: int
    metrics: float
    language: str = "python"

@dataclass
class IslandStatus:
    """岛屿状态数据类"""
    island_id: str
    current_iteration: int
    max_iterations: int
    best_program: ProgramInfo
    status: str  # "running", "completed", "waiting"
    last_update: datetime

# ==================== Rich监控界面类 ====================

class OpenEvolveMonitor:
    """OpenEvolve进化过程监控界面"""
    
    def __init__(self, num_islands: int = 4, max_iterations: int = 100):
        self.console = Console()
        self.num_islands = num_islands
        self.max_iterations = max_iterations
        
        # 初始化岛屿状态
        self.islands: Dict[str, IslandStatus] = {}
        for i in range(num_islands):
            self.islands[str(i)] = IslandStatus(
                island_id=str(i),
                current_iteration=0,
                max_iterations=max_iterations,
                best_program=ProgramInfo(
                    id=f"prog_{i}_0",
                    generation=0,
                    iteration=0,
                    metrics=random.uniform(0.1, 0.3)
                ),
                status="waiting",
                last_update=datetime.now()
            )
        
        # 全局最佳程序
        self.global_best = self.islands["0"].best_program
        
    def create_layout(self) -> Layout:
        """创建监控界面布局"""
        layout = Layout(name="root")
        
        # 主要布局：头部、主体、底部
        layout.split_column(
            Layout(name="header", size=3),      # 头部：标题和时间
            Layout(name="main", ratio=1),       # 主体：主要监控内容
            Layout(name="footer", size=8),      # 底部：进度条
        )
        
        # 主体分为左右两部分
        layout["main"].split_row(
            Layout(name="left", ratio=2),       # 左侧：岛屿状态表格
            Layout(name="right", ratio=1),      # 右侧：最佳程序信息
        )
        
        return layout
    
    def create_header(self) -> Panel:
        """创建头部面板"""
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right")
        
        grid.add_row(
            "[bold cyan]🧬 OpenEvolve 进化监控系统[/bold cyan]",
            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
        )
        
        return Panel(grid, style="white on blue", box=box.ROUNDED)
    
    def create_islands_table(self) -> Table:
        """创建岛屿状态表格"""
        table = Table(title="🏝️ 岛屿状态监控", box=box.ROUNDED)
        
        # 添加列
        table.add_column("岛屿ID", style="cyan", width=8)
        table.add_column("状态", style="green", width=10)
        table.add_column("迭代进度", style="yellow", width=12)
        table.add_column("最佳程序ID", style="magenta", width=15)
        table.add_column("精度", style="red", width=10)
        table.add_column("最后更新", style="dim", width=12)
        
        # 添加每个岛屿的数据
        for island in self.islands.values():
            # 状态显示
            status_map = {
                "running": "🟢 运行中",
                "completed": "✅ 完成",
                "waiting": "⏸️ 等待中"
            }
            status_display = status_map.get(island.status, "❓ 未知")
            
            # 进度显示
            progress_text = f"{island.current_iteration}/{island.max_iterations}"
            
            # 精度显示（保留4位小数）
            metrics_text = f"{island.best_program.metrics:.4f}"
            
            # 时间显示
            time_text = island.last_update.strftime("%H:%M:%S")
            
            table.add_row(
                island.island_id,
                status_display,
                progress_text,
                island.best_program.id,
                metrics_text,
                time_text
            )
        
        return table
    
    def create_best_program_panel(self) -> Panel:
        """创建最佳程序信息面板"""
        content = []
        
        content.append("[bold green]🏆 全局最佳程序[/bold green]\n")
        content.append(f"[cyan]程序ID:[/cyan] {self.global_best.id}")
        content.append(f"[cyan]代数:[/cyan] {self.global_best.generation}")
        content.append(f"[cyan]发现迭代:[/cyan] {self.global_best.iteration}")
        content.append(f"[cyan]精度:[/cyan] [bold red]{self.global_best.metrics:.6f}[/bold red]")
        content.append(f"[cyan]语言:[/cyan] {self.global_best.language}")
        
        # 添加一些统计信息
        content.append("\n[bold yellow]📊 统计信息[/bold yellow]")
        
        running_count = sum(1 for island in self.islands.values() if island.status == "running")
        completed_count = sum(1 for island in self.islands.values() if island.status == "completed")
        
        content.append(f"[dim]运行中岛屿:[/dim] {running_count}")
        content.append(f"[dim]完成岛屿:[/dim] {completed_count}")
        content.append(f"[dim]总岛屿数:[/dim] {self.num_islands}")
        
        # 计算平均精度
        avg_metrics = sum(island.best_program.metrics for island in self.islands.values()) / len(self.islands)
        content.append(f"[dim]平均精度:[/dim] {avg_metrics:.4f}")
        
        return Panel(
            "\n".join(content),
            title="[bold]最佳程序信息[/bold]",
            border_style="green",
            box=box.ROUNDED
        )
    
    def create_progress_panel(self) -> Panel:
        """创建进度条面板"""
        # 创建进度条
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TextColumn("[progress.completed]{task.completed}/{task.total}"),
            expand=True
        )
        
        # 为每个岛屿添加进度条
        for island in self.islands.values():
            color = "green" if island.status == "completed" else "cyan" if island.status == "running" else "dim"
            task_id = progress.add_task(
                f"[{color}]岛屿 {island.island_id}[/{color}]",
                total=island.max_iterations,
                completed=island.current_iteration
            )
        
        return Panel(
            progress,
            title="[bold]🔄 进化进度[/bold]",
            border_style="blue",
            box=box.ROUNDED
        )
    
    def update_island_status(self, island_id: str, iteration: int, new_program: Optional[ProgramInfo] = None, status: str = "running"):
        """更新岛屿状态"""
        if island_id in self.islands:
            island = self.islands[island_id]
            island.current_iteration = iteration
            island.status = status
            island.last_update = datetime.now()
            
            if new_program is not None:
                island.best_program = new_program
                
                # 检查是否是全局最佳
                if new_program.metrics > self.global_best.metrics:
                    self.global_best = new_program
    
    def simulate_evolution(self):
        """模拟进化过程"""
        layout = self.create_layout()
        
        with Live(layout, refresh_per_second=4, screen=True) as live:
            # 运行模拟
            for iteration in range(self.max_iterations):
                # 更新每个岛屿的状态
                for island_id in self.islands.keys():
                    if random.random() > 0.1:  # 90%概率继续运行
                        current_iter = min(iteration + random.randint(0, 2), self.max_iterations)
                        
                        # 偶尔生成新的更好的程序
                        if random.random() > 0.8:  # 20%概率找到更好的程序
                            new_program = ProgramInfo(
                                id=f"prog_{island_id}_{current_iter}",
                                generation=current_iter // 10,
                                iteration=current_iter,
                                metrics=self.islands[island_id].best_program.metrics + random.uniform(0.001, 0.05)
                            )
                            status = "completed" if current_iter >= self.max_iterations else "running"
                            self.update_island_status(island_id, current_iter, new_program, status)
                        else:
                            status = "completed" if current_iter >= self.max_iterations else "running"
                            self.update_island_status(island_id, current_iter, status=status)
                
                # 更新布局内容
                layout["header"].update(self.create_header())
                layout["left"].update(self.create_islands_table())
                layout["right"].update(self.create_best_program_panel())
                layout["footer"].update(self.create_progress_panel())
                
                time.sleep(0.5)  # 控制更新速度
                
                # 检查是否所有岛屿都完成了
                if all(island.status == "completed" for island in self.islands.values()):
                    break
        
        # 显示最终结果
        self.console.print(Panel.fit(
            f"[bold green]🎉 进化完成！[/bold green]\n\n"
            f"[cyan]全局最佳程序:[/cyan] {self.global_best.id}\n"
            f"[cyan]最终精度:[/cyan] [bold red]{self.global_best.metrics:.6f}[/bold red]\n"
            f"[cyan]发现代数:[/cyan] {self.global_best.generation}\n"
            f"[cyan]发现迭代:[/cyan] {self.global_best.iteration}",
            title="进化结果",
            border_style="green"
        ))

# ==================== 主函数 ====================

def main():
    """主函数 - 运行OpenEvolve监控示例"""
    console = Console()
    
    # 显示欢迎信息
    console.print(Panel.fit(
        "[bold cyan]🧬 OpenEvolve 监控系统演示[/bold cyan]\n\n"
        "这个演示展示了如何使用Rich库为OpenEvolve项目\n"
        "创建一个美观的实时监控界面。\n\n"
        "[dim]按 Ctrl+C 可以随时退出[/dim]",
        title="欢迎",
        border_style="blue"
    ))
    
    time.sleep(2)
    
    try:
        # 创建并运行监控器
        monitor = OpenEvolveMonitor(num_islands=4, max_iterations=50)
        monitor.simulate_evolution()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ 用户中断了演示[/yellow]")
    
    console.print("\n[green]感谢使用OpenEvolve监控系统！[/green]")

if __name__ == "__main__":
    main() 