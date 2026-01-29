import json
import logging
import time
import numpy as np
from typing import Optional, Callable, Any
import rich.box                
import rich                
from rich.columns import Columns as RichColumns                
from rich.console import Console as RichConsole                
from rich.table import Table as RichTable                

logger = logging.getLogger(__name__)


class DatabaseDisplay:
    """Handles rich console display and formatting for database summaries."""

    def __init__(
        self,
        cursor,
        conn,
        config,
        island_manager,
        count_programs_func: Callable[[], int],
        get_best_program_func: Callable[[], Optional[Any]],
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.island_manager = island_manager
        self.count_programs_func = count_programs_func
        self.get_best_program_func = get_best_program_func

    def print_program_summary(self, program, console: Optional[RichConsole] = None):
        """Print a rich summary of a newly added program in two rows."""
        _console = console or RichConsole()

                                      
        best_program_overall = self.get_best_program_func()
        best_score_str = "[dim]N/A[/dim]"
        if best_program_overall and best_program_overall.combined_score is not None:
            best_score_val = best_program_overall.combined_score
            if best_score_val is not None:
                best_score_str = f"[bold yellow]{best_score_val:.3f}[/bold yellow]"

                                                                   
        table = RichTable(
            title=(
                f"[bold green]Program Evaluation Summary - "
                f"Gen {program.generation}[/bold green]"
            ),
            border_style="green",
            box=rich.box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            width=120,                                 
        )

                                                   
        table.add_column(
            "GenID: " + str(program.generation),
            style="cyan",
            justify="center",
            width=12,
        )
        table.add_column("Island", style="magenta", justify="center", width=8)
        table.add_column("Status", style="white", justify="center", width=14)
        table.add_column("Score", style="white", justify="right", width=8)
        table.add_column(
            "Patch Name",
            style="yellow",
            justify="left",
            width=32,
            overflow="ellipsis",
        )
        table.add_column(
            "Type",
            style="yellow",
            justify="left",
            width=6,
            overflow="ellipsis",
        )
        table.add_column("Complex", style="yellow", justify="right", width=7)
        table.add_column("Cost", style="green", justify="right", width=7)
        table.add_column("Time", style="blue", justify="right", width=5)

                                       
        if program.correct:
            status_display = "[bold green]✓ Correct[/bold green]"
        else:
            status_display = "[bold red]✗ Incorrect[/bold red]"

                                              
        score_display = "[dim]N/A[/dim]"
        if program.combined_score is not None:
            score = program.combined_score
            if score > 0.8:
                color = "bold green"
            elif score > 0.5:
                color = "green"
            else:
                color = "yellow"
            score_display = f"[{color}]{score:.3f}[/{color}]"

                               
        cost_display = "[dim]N/A[/dim]"
        if program.metadata:
            api_cost = float(program.metadata.get("api_costs", 0))
            embed_cost = float(program.metadata.get("embed_cost", 0))
            novelty_cost = float(program.metadata.get("novelty_cost", 0))
            meta_cost = float(program.metadata.get("meta_cost", 0))
            total_cost = api_cost + embed_cost + novelty_cost + meta_cost
            cost_display = f"${total_cost:.3f}"

                     
        time_display = "[dim]N/A[/dim]"
        if program.metadata and "compute_time" in program.metadata:
            time_val = program.metadata["compute_time"]
            if time_val > 60:
                time_display = f"{time_val / 60:.1f}m"
            else:
                time_display = f"{time_val:.1f}s"

                                                    
        metadata = program.metadata or {}
        patch_name_raw = metadata.get("patch_name", "[dim]N/A[/dim]")
        if patch_name_raw is None:
            patch_name_raw = "[dim]N/A[/dim]"
        patch_name = str(patch_name_raw)[:30]

        patch_type_raw = metadata.get("patch_type", "[dim]N/A[/dim]")
        if patch_type_raw is None:
            patch_type_raw = "[dim]N/A[/dim]"
        patch_type = str(patch_type_raw)

                          
        island_display = (
            f"I-{program.island_idx}" if program.island_idx is not None else "N/A"
        )
        table.add_row(
            f"Best: {best_score_str}",
            island_display,
            status_display,
            score_display,
            patch_name,
            patch_type,
            f"{program.complexity:.1f}",
            cost_display,
            time_display,
        )
        _console.print(table)

    def print_summary(self, console: Optional[RichConsole] = None) -> None:
        """Print a summary of the database contents to the terminal."""
        if not self.cursor or not self.conn:
            logger.error("Database not connected. Cannot print summary.")
            return

        _console = console or RichConsole()

                                                                          
        total_api_cost = 0
        total_embed_cost = 0
        total_novelty_cost = 0
        total_meta_cost = 0
        total_compute_time = 0
        avg_score = 0.0
        best_score = 0.0                         
        num_with_scores = 0
        all_scores = []
        if self.cursor:                             
            query = "SELECT metadata, combined_score FROM programs"
            self.cursor.execute(query)
            for row in self.cursor.fetchall():
                if row["metadata"]:
                    metadata = json.loads(row["metadata"] or "{}")
                    if "api_costs" in metadata:
                        total_api_cost += float(metadata["api_costs"])
                    if "embed_cost" in metadata:
                        total_embed_cost += float(metadata["embed_cost"])
                    if "novelty_cost" in metadata:
                        total_novelty_cost += float(metadata["novelty_cost"])
                    if "meta_cost" in metadata:
                        total_meta_cost += float(metadata["meta_cost"])
                    if "compute_time" in metadata:
                        total_compute_time += float(metadata["compute_time"])

                if row["combined_score"] is not None:
                    score = float(row["combined_score"])
                    avg_score += score
                    if score > best_score:                                     
                        best_score = score
                    num_with_scores += 1
                    all_scores.append(score)
        median_score = np.median(all_scores)

                                
        summary_table = RichTable(
            title="[bold cyan]Program Database Summary[/bold cyan]",
            border_style="cyan",
            box=rich.box.ROUNDED,
            width=40,                                   
        )
        summary_table.add_column("Metric", style="cyan bold", no_wrap=True)
        summary_table.add_column("Value", style="magenta")

                                                        
        summary_table.add_row(
            "Overall Best Score",
            f"[bold cyan]{best_score:.2f}[/bold cyan]"
            if num_with_scores > 0
            else "[dim]N/A[/dim]",
        )

                                 
        total_programs = self.count_programs_func()
        summary_table.add_row("Total Programs", f"[bold]{total_programs}[/bold]")

                          
        self.cursor.execute("SELECT COUNT(*) FROM programs WHERE correct = 1")
        correct_programs = (self.cursor.fetchone() or [0])[0]
        correct_percentage = (
            (correct_programs / total_programs * 100) if total_programs > 0 else 0
        )
        summary_table.add_row(
            "Correct Programs",
            f"[bold]{correct_programs}[/bold] / {total_programs} "
            f"({correct_percentage:.0f}%)",
        )

                      
        self.cursor.execute("SELECT COUNT(*) FROM archive")
        archive_count = (self.cursor.fetchone() or [0])[0]
        archive_percentage = (
            (archive_count / self.config.archive_size * 100)
            if self.config.archive_size > 0
            else 0
        )
        summary_table.add_row(
            "Archived Programs",
            f"[bold]{archive_count}[/bold] / {self.config.archive_size} "
            f"({archive_percentage:.0f}%)",
        )

                            
        if hasattr(self.config, "num_islands") and self.config.num_islands > 0:
            island_str = self.island_manager.format_island_display()
            summary_table.add_row("Island Populations", island_str)

                                    
            migration_info = self.island_manager.get_migration_info()
            if migration_info:
                summary_table.add_row("Migration Policy", migration_info)

                                       
        best_program_table_renderable = None
        best_program = self.get_best_program_func()
        if best_program:
            best_program_table = RichTable(
                title="[bold yellow]Best Program[/bold yellow]",
                border_style="yellow",
                box=rich.box.ROUNDED,
                width=40,                                   
            )
            best_program_table.add_column("Attribute", style="green bold")
            best_program_table.add_column("Value", style="yellow")

                                      
            short_id = (
                best_program.id[:8] + "..."
                if len(best_program.id) > 8
                else best_program.id
            )
            best_program_table.add_row("ID", f"[dim]{short_id}[/dim]")
            best_program_table.add_row("Generation", str(best_program.generation))
            complexity_val = f"{best_program.complexity:.2f}"
            best_program_table.add_row("Complexity", complexity_val)
            if best_program.embedding:
                embedding_val = f"{best_program.embedding[0]:.2f}"
            else:
                embedding_val = "N/A"
            best_program_table.add_row("Embedding[0]", embedding_val)

            if best_program.combined_score is not None:
                score_display = (
                    f"[bold green]{best_program.combined_score:.2f}[/bold green]"
                )
                best_program_table.add_row("Metric: Score", score_display)
            else:
                best_program_table.add_row("Metrics", "[dim]N/A[/dim]")

                                     
            timestamp_display = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(best_program.timestamp)
            )
            best_program_table.add_row("Timestamp", timestamp_display)
            best_program_table_renderable = best_program_table

                                       
        cost_table = RichTable(
            title="[bold magenta]Cost & Stats Summary[/bold magenta]",
            border_style="magenta",
            box=rich.box.ROUNDED,
            width=40,                                   
        )
        cost_table.add_column("Metric", style="magenta bold")
        cost_table.add_column("Value", style="green")

                              
        total_cost = (
            total_api_cost + total_embed_cost + total_novelty_cost + total_meta_cost
        )
        if total_cost > 0:
            cost_table.add_row("Total API Cost", f"[bold]${total_api_cost:.2f}[/bold]")
            cost_table.add_row(
                "Total Embedding Cost", f"[bold]${total_embed_cost:.2f}[/bold]"
            )
            cost_table.add_row(
                "Total Novelty Cost", f"[bold]${total_novelty_cost:.2f}[/bold]"
            )
            cost_table.add_row(
                "Total Meta Cost", f"[bold]${total_meta_cost:.2f}[/bold]"
            )
            cost_table.add_row("Total Combined Cost", f"[bold]${total_cost:.2f}[/bold]")
            if total_programs > 0:
                avg_cost = total_cost / total_programs
                cost_table.add_row("Avg $/Program", f"${avg_cost:.2f}")

                                       
        if total_compute_time > 0:
            hours = int(total_compute_time // 3600)
            minutes = int((total_compute_time % 3600) // 60)
            seconds = int(total_compute_time % 60)
            time_str = f"{hours}h {minutes}m {seconds}s"
            cost_table.add_row("Total Compute", time_str)

                                      
        if num_with_scores > 0:
            avg_score_display = f"{avg_score / num_with_scores:.2f}"
            cost_table.add_row("Avg Score", avg_score_display)
            cost_table.add_row("Best Score", f"[bold]{best_score:.2f}[/bold]")
            cost_table.add_row("Median Score", f"{median_score:.2f}")

                                                                     
        tables_to_display = [summary_table]
        if best_program_table_renderable:
            tables_to_display.append(best_program_table_renderable)
        tables_to_display.append(cost_table)

        _console.print(RichColumns(tables_to_display))

                                                                             
        highlight_table = RichTable(
            title="[bold green]Top 10 Best Performing Programs[/bold green]",
            border_style="green",
            box=rich.box.ROUNDED,
            show_lines=True,
            width=120,                                     
        )
                                                  
        highlight_table.add_column(
            "Rank", style="magenta bold", justify="center", width=6
        )
        highlight_table.add_column("Gen", style="cyan bold", justify="center", width=6)
        highlight_table.add_column("✓/✗", style="red bold", justify="center", width=4)
        highlight_table.add_column(
            "Score", style="green bold", justify="right", width=8
        )
        highlight_table.add_column(
            "Complexity", style="yellow", justify="right", width=10
        )
        highlight_table.add_column(
            "Patch Name",
            style="blue",
            justify="left",
            width=32,
            overflow="ellipsis",
        )
        highlight_table.add_column("Type", style="cyan", justify="left", width=8)
        highlight_table.add_column("Island", style="magenta", justify="center", width=8)
        highlight_table.add_column("Children", style="blue", justify="right", width=8)
        highlight_table.add_column("Timestamp", style="dim", width=19)

                                                                 
        query = (
            "SELECT id, code, language, parent_id, generation, timestamp, "
            "combined_score, public_metrics, private_metrics, "
            "complexity, embedding, metadata, correct, island_idx, "
            "children_count "
            "FROM programs WHERE combined_score IS NOT NULL AND correct = 1 "
            "ORDER BY combined_score DESC LIMIT 10"
        )
        self.cursor.execute(query)
        top_program_rows = self.cursor.fetchall()

        if not top_program_rows:
            msg = "[yellow]No programs with scores in the database to display.[/yellow]"
            _console.print(msg)
            return

                                         
        for rank, row_data in enumerate(top_program_rows, 1):
            p_dict = dict(row_data)
            p_dict["public_metrics"] = (
                json.loads(p_dict["public_metrics"])
                if p_dict.get("public_metrics")
                else {}
            )
            p_dict["private_metrics"] = (
                json.loads(p_dict["private_metrics"])
                if p_dict.get("private_metrics")
                else {}
            )
            metadata_str = p_dict.get("metadata")
            p_dict["metadata"] = json.loads(metadata_str) if metadata_str else {}
                                                                  
            if "correct" in p_dict:
                p_dict["correct"] = bool(p_dict["correct"])

                                                                 
            from .dbase import Program

            prog = Program.from_dict(p_dict)

                           
            combined_score_val = prog.combined_score
            combined_score_str = (
                f"{combined_score_val:.3f}" if combined_score_val is not None else "N/A"
            )

                                   
            correct_str = (
                "[bold green]✓[/bold green]"
                if prog.correct
                else "[bold red]✗[/bold red]"
            )

                            
            island_display = (
                f"I{prog.island_idx}" if prog.island_idx is not None else "N/A"
            )

                            
            children_count = prog.children_count or 0

            timestamp = time.localtime(prog.timestamp)
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", timestamp)

                                                     
            if rank == 1:
                rank_str = "[bold gold1]#1[/bold gold1]"
                score_str = f"[bold gold1]{combined_score_str}[/bold gold1]"
            elif rank == 2:
                rank_str = "[bold bright_white]#2[/bold bright_white]"
                score_str = (
                    f"[bold bright_white]{combined_score_str}[/bold bright_white]"
                )
            elif rank == 3:
                rank_str = "[bold orange1]#3[/bold orange1]"
                score_str = f"[bold orange1]{combined_score_str}[/bold orange1]"
            else:
                rank_str = f"#{rank}"
                score_str = combined_score_str

            highlight_table.add_row(
                rank_str,
                str(prog.generation),
                correct_str,
                score_str,
                f"{prog.complexity:.1f}",
                prog.metadata.get("patch_name", "N/A")[:30],
                prog.metadata.get("patch_type", "N/A")[:6],
                island_display,
                str(children_count),
                ts_str,
            )

        _console.print(highlight_table)

    def set_last_iteration(self, last_iteration: int):
        """Set the last iteration for display purposes."""
        self.last_iteration = last_iteration

    def print_sampling_summary(
        self,
        parent,
        archive_inspirations,
        top_k_inspirations,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
    ):
        """Print a summary of the sampled parent and inspirations."""
        console = RichConsole()

                                                                                      
        if target_generation is not None:
            gen_display = target_generation
        else:
                                                                     
            gen_display = parent.generation + 1

                                                              
        title_parts = [
            f"[bold red]Parent & Context Sampling Summary - Gen {gen_display}"
        ]

                                             
        if (
            novelty_attempt is not None
            and max_novelty_attempts is not None
            and resample_attempt is not None
            and max_resample_attempts is not None
        ):
            title_parts.append(
                f" (Novelty: {novelty_attempt}/{max_novelty_attempts}, "
                f"Resample: {resample_attempt}/{max_resample_attempts})"
            )

        title_parts.append("[/bold red]")

                                                                               
                                                                          
        table = RichTable(
            title="".join(title_parts),
            border_style="red",
            show_header=True,
            header_style="bold cyan",
            width=120,                                     
        )

                                                        
        table.add_column("Role", style="cyan bold", width=12)
        table.add_column("Gen", style="magenta", justify="center", width=5)
        table.add_column("Island", style="red", justify="center", width=8)
        table.add_column("✓/✗", style="white", justify="center", width=6)
        table.add_column("Score", style="green", justify="right", width=8)
        table.add_column(
            "Patch Name",
            style="yellow",
            justify="left",
            width=32,
            overflow="ellipsis",
        )
        table.add_column(
            "Type",
            style="yellow",
            justify="left",
            width=6,
            overflow="ellipsis",
        )
        table.add_column("Complex", style="blue", justify="right", width=7)
        table.add_column("Cost", style="green", justify="right", width=7)
        table.add_column("Time", style="cyan", justify="right", width=5)

        def format_program_row(prog, role_name):
            """Format a program row with all details."""
                                     
            if prog.combined_score is not None:
                score = prog.combined_score
                if score > 0.8:
                    score_display = f"[bold green]{score:.3f}[/bold green]"
                elif score > 0.5:
                    score_display = f"[green]{score:.3f}[/green]"
                else:
                    score_display = f"[yellow]{score:.3f}[/yellow]"
            else:
                score_display = "[dim]N/A[/dim]"

                    
            island = f"I-{prog.island_idx}" if prog.island_idx is not None else "N/A"

                         
            correct = (
                "[bold green]✓[/bold green]"
                if prog.correct
                else "[bold red]✗[/bold red]"
            )

                            
            cost_display = "[dim]N/A[/dim]"
            if prog.metadata:
                api_cost = float(prog.metadata.get("api_costs", 0))
                embed_cost = float(prog.metadata.get("embed_cost", 0))
                novelty_cost = float(prog.metadata.get("novelty_cost", 0))
                meta_cost = float(prog.metadata.get("meta_cost", 0))
                total_cost = api_cost + embed_cost + novelty_cost + meta_cost
                cost_display = f"${total_cost:.3f}"

                  
            time_display = "[dim]N/A[/dim]"
            if prog.metadata and "compute_time" in prog.metadata:
                time_val = prog.metadata["compute_time"]
                if time_val > 60:
                    time_display = f"{time_val / 60:.1f}m"
                else:
                    time_display = f"{time_val:.1f}s"

                                 
            patch_name = prog.metadata.get("patch_name", "[dim]N/A[/dim]")[:30]
            patch_type = prog.metadata.get("patch_type", "[dim]N/A[/dim]")

            return [
                role_name,
                str(prog.generation),
                island,
                correct,
                score_display,
                patch_name,
                patch_type,
                f"{prog.complexity:.1f}",
                cost_display,
                time_display,
            ]

                        
        table.add_row(*format_program_row(parent, "[bold]PARENT[/bold]"))

                                  
        for i, prog in enumerate(archive_inspirations):
            table.add_row(*format_program_row(prog, f"Archive-{i + 1}"))

                                
        for i, prog in enumerate(top_k_inspirations):
            table.add_row(*format_program_row(prog, f"TopK-{i + 1}"))

        console.print(table)
