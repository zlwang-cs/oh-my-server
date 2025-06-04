#!/usr/bin/env python3
"""
Interactive Megatron Parallel Parameters Calculator
An interactive command-line tool for calculating Megatron-LM parallel configuration parameters
"""

import math
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.align import Align
from rich.prompt import Prompt, IntPrompt, Confirm
from typing import Optional, Dict, Tuple, Any
from sympy import symbols, solve, Eq

console = Console()

class InteractiveMegatronCalculator:
    """Interactive Megatron Parallel Parameters Calculator"""
    
    def __init__(self):
        self.model_size_presets = {
            '7B': 7, '13B': 13, '30B': 30, '65B': 65, '175B': 175
        }
        
        self.parameters = {
            'total_gpus': {
                'name': 'Total GPU Count',
                'short': 'GPU',
                'description': 'Total number of GPUs available (enter ? to calculate)',
                'type': 'int',
                'default': 8,
                'value': None,
                'order': 1
            },
            'micro_batch_size': {
                'name': 'Micro Batch Size',
                'short': 'MBS',
                'description': 'Micro batch size per GPU (enter ? to calculate)',
                'type': 'int',
                'default': 4,
                'value': None,
                'order': 2
            },
            'global_batch_size': {
                'name': 'Global Batch Size',
                'short': 'GBS',
                'description': 'Total batch size across all GPUs (enter ? to calculate)',
                'type': 'int',
                'default': 8,
                'value': None,
                'order': 3
            },
            'gradient_accumulation_steps': {
                'name': 'Gradient Accumulation Steps',
                'short': 'GAS',
                'description': 'Number of gradient accumulation steps (enter ? to calculate)',
                'type': 'int',
                'default': 1,
                'value': None,
                'order': 4
            },
            'tensor_parallel': {
                'name': 'Tensor Parallel Size',
                'short': 'TP',
                'description': 'Tensor model parallel size (enter ? to calculate)',
                'type': 'int',
                'default': 2,
                'value': None,
                'order': 5
            },
            'pipeline_parallel': {
                'name': 'Pipeline Parallel Size',
                'short': 'PP',
                'description': 'Pipeline model parallel size (enter ? to calculate)',
                'type': 'int',
                'default': 2,
                'value': None,
                'order': 6
            },
            'data_parallel': {
                'name': 'Data Parallel Size',
                'short': 'DP',
                'description': 'Data parallel size (enter ? to calculate)',
                'type': 'int',
                'default': 2,
                'value': None,
                'order': 7
            },
            'model_size': {
                'name': 'Model Size',
                'short': 'MODEL',
                'description': 'Model parameter size (7B, 13B, 30B, 65B, 175B, or custom)',
                'type': 'str',
                'default': '7B',
                'value': None,
                'order': 8
            },
            'sequence_length': {
                'name': 'Sequence Length',
                'short': 'SEQ',
                'description': 'Maximum sequence length',
                'type': 'int',
                'default': 2048,
                'value': None,
                'order': 9
            }
        }
    
    def run(self):
        """Run the interactive calculator"""
        self._display_header()
        self._display_parameters_overview()
        self._collect_user_inputs()
        
        param_source = self._set_parameter_source()
        
        # Calculate configuration
        config = self._calculate_configuration()
        
        # Display results
        self._display_results(config, param_source)
        
        return config
    
    def _display_header(self):
        """Display application header"""
        title = Text("🚀 INTERACTIVE MEGATRON PARALLEL CALCULATOR", style="bold magenta")
        subtitle = Text("Enter your parameters step by step", style="italic cyan")
        
        header_panel = Panel(
            Align.center(f"{title}\n{subtitle}"),
            box=box.DOUBLE_EDGE,
            border_style="bright_blue",
            width=128
            
        )
        console.print(header_panel)
        console.print()
    
    def _display_parameters_overview(self):
        """Display all available parameters"""
        console.print("📋 [bold yellow]Available Parameters:[/bold yellow]\n")
        
        # All parameters table (sorted by order)
        param_table = Table(title="🎯 Configuration Parameters", box=box.ROUNDED, border_style="blue")
        param_table.add_column("Short", style="cyan", width=8)
        param_table.add_column("Parameter Name", style="yellow", width=30)
        param_table.add_column("Default", style="green", width=10)
        param_table.add_column("Description", style="white", width=67)
        
        # Sort parameters by order
        sorted_params = sorted(self.parameters.items(), key=lambda x: x[1]['order'])
        
        for key, param in sorted_params:
            default_str = str(param['default'])
            param_table.add_row(param['short'], param['name'], default_str, param['description'])
        
        console.print(param_table)
        console.print()
        
        console.print("💡 [italic]Press Enter to use default values, or enter '?' to calculate from other parameters[/italic]\n")
        console.print("🔍 [italic]Key relationship: GBS = MBS × DP × GAS; Total GPUs = TP × PP × DP[/italic]\n")
    
    def _collect_user_inputs(self):
        """Collect user inputs interactively"""
        console.print("🎯 [bold green]Please enter your parameters:[/bold green]\n")
        
        # Collect all parameters in sorted order
        sorted_params = sorted(self.parameters.items(), key=lambda x: x[1]['order'])
        for key, param in sorted_params:
            self._prompt_parameter(key, param)
        
        console.print()
    
    def _prompt_parameter(self, key: str, param: Dict[str, Any]):
        """Prompt user for a specific parameter"""
        prompt_text = f"[cyan]{param['short']}[/cyan] - {param['name']}"
        default_value = param['default']
        
        if param['type'] == 'int':
            value_str = Prompt.ask(prompt_text, default=str(default_value))
            if value_str.strip() == "?":
                param['value'] = "?"
                console.print(f"   [yellow]Will calculate {param['short']} from other parameters[/yellow]")
            else:
                try:
                    param['value'] = int(value_str)
                except ValueError:
                    console.print(f"[red]Invalid input. Using default value {default_value}.[/red]")
                    param['value'] = default_value
        else:  # string type
            value = Prompt.ask(prompt_text, default=str(default_value))
            param['value'] = value
    
    def _set_parameter_source(self) -> Dict[str, str]:
        """Label the source of each parameter based on user input
        
        Returns:
            Dict[str, str]: Dictionary mapping parameter names to their sources
            ('default', 'user specified', or 'calculated')
        """
        param_sources = {}
        
        for key, param in self.parameters.items():
            if param['value'] == "?":
                param_sources[key] = "calculated"
            elif param['value'] == param['default']:
                param_sources[key] = "default"
            else:
                param_sources[key] = "user specified"
        
        return param_sources
    
    def _calculate_configuration(self) -> Dict:
        """Calculate the parallel configuration"""
        console.print("🔄 [bold yellow]Calculating optimal configuration...[/bold yellow]\n")
        
        model_size = self.parameters['model_size']['value']
        model_params = self._parse_model_size(model_size)
        sequence_length = self.parameters['sequence_length']['value']
        
        total_gpus = self.parameters['total_gpus']['value']
        micro_batch_size = self.parameters['micro_batch_size']['value']
        global_batch_size = self.parameters['global_batch_size']['value']
        gradient_accumulation_steps = self.parameters['gradient_accumulation_steps']['value']
        
        tp = self.parameters['tensor_parallel']['value']
        pp = self.parameters['pipeline_parallel']['value']
        dp = self.parameters['data_parallel']['value']
        
        # Calculate missing parameters
        config = self._calculate_all_parameters(
            total_gpus, model_params, sequence_length,
            micro_batch_size, global_batch_size, gradient_accumulation_steps,
            tp, pp, dp
        )
        
        return config
    
    def _parse_model_size(self, model_size: str) -> int:
        """Parse model size string to parameter count"""
        if model_size in self.model_size_presets:
            return self.model_size_presets[model_size]
        elif model_size.endswith('B'):
            try:
                return int(model_size[:-1])
            except ValueError:
                raise ValueError(f"Cannot parse model size: {model_size}")
        else:
            try:
                return int(model_size)
            except ValueError:
                raise ValueError(f"Cannot parse model size: {model_size}")
    
    def _calculate_all_parameters(self, total_gpus, model_params, sequence_length,
                                  micro_batch_size, global_batch_size, gradient_accumulation_steps,
                                  tp, pp, dp) -> Dict:
        """Calculate all missing parameters using system of linear equations"""

        # Define symbols for all parameters
        tp_sym, pp_sym, dp_sym = symbols('tp pp dp')
        total_gpus_sym = symbols('total_gpus') 
        mbs_sym = symbols('micro_batch_size')
        gbs_sym = symbols('global_batch_size')
        gas_sym = symbols('gradient_accumulation_steps')

        # Build system of equations
        equations = []

        # Core equation: total_gpus = tp * pp * dp
        equations.append(Eq(total_gpus_sym, tp_sym * pp_sym * dp_sym))

        # Batch size equation: global_batch_size = micro_batch_size * dp * gradient_accumulation_steps
        equations.append(Eq(gbs_sym, mbs_sym * dp_sym * gas_sym))

        # Add known values as equations
        if total_gpus != "?":
            equations.append(Eq(total_gpus_sym, total_gpus))
        if tp != "?":
            equations.append(Eq(tp_sym, tp))
        if pp != "?":
            equations.append(Eq(pp_sym, pp))
        if dp != "?":
            equations.append(Eq(dp_sym, dp))
        if micro_batch_size != "?":
            equations.append(Eq(mbs_sym, micro_batch_size))
        if global_batch_size != "?":
            equations.append(Eq(gbs_sym, global_batch_size))
        if gradient_accumulation_steps != "?":
            equations.append(Eq(gas_sym, gradient_accumulation_steps))

        # Solve the system of equations
        try:
            solution = solve(equations)
        except Exception as e:    
            raise ValueError(f"❌ Cannot solve the equation system: {e}")

        if not solution:
            # Debug: Check the equation values when no solution is found
            console.print("\n❌ [red bold]No solution found. Debugging equation values:[/red bold]")
            
            # Check total_gpus = tp * pp * dp equation
            tp_val = tp
            pp_val = pp
            dp_val = dp
            total_gpus_val = total_gpus
            
            console.print(f"  [cyan]Equation 1:[/cyan] [white]total_gpus = tp × pp × dp[/white]")
            if tp_val != "?" and pp_val != "?" and dp_val != "?" and total_gpus_val != "?":
                calculated_total = tp_val * pp_val * dp_val
                if calculated_total != total_gpus_val:
                    console.print(f"    [red]❌ Mismatch:[/red] total_gpus([yellow]{total_gpus_val}[/yellow]) ≠ calculated_total_gpus([yellow]{calculated_total} = {tp_val}[/yellow] × [yellow]{pp_val}[/yellow] × [yellow]{dp_val}[/yellow])")
                else:
                    console.print(f"    [green]✓ Matches:[/green] total_gpus([yellow]{total_gpus_val}[/yellow]) = calculated_total_gpus([yellow]{calculated_total} = {tp_val}[/yellow] × [yellow]{pp_val}[/yellow] × [yellow]{dp_val}[/yellow])")
            else:
                console.print(f"    [dim]Current:[/dim] total_gpus([yellow]{total_gpus_val}[/yellow]) = [yellow]{tp_val}[/yellow] × [yellow]{pp_val}[/yellow] × [yellow]{dp_val}[/yellow]")
            
            # Check global_batch_size = micro_batch_size * dp * gradient_accumulation_steps equation
            gbs_val = global_batch_size
            mbs_val = micro_batch_size
            gas_val = gradient_accumulation_steps
            
            console.print(f"  [cyan]Equation 2:[/cyan] [white]global_batch_size = micro_batch_size × dp × gradient_accumulation_steps[/white]")
            if mbs_val != "?" and dp_val != "?" and gas_val != "?" and gbs_val != "?":
                calculated_gbs = mbs_val * dp_val * gas_val
                if calculated_gbs != gbs_val:
                    console.print(f"    [red]❌ Mismatch:[/red] global_batch_size([yellow]{gbs_val}[/yellow]) ≠ calculated_global_batch_size([yellow]{calculated_gbs} = {mbs_val}[/yellow] × [yellow]{dp_val}[/yellow] × [yellow]{gas_val}[/yellow])")
                else:
                    console.print(f"    [green]✓ Matches:[/green] global_batch_size([yellow]{gbs_val}[/yellow]) = calculated_global_batch_size([yellow]{calculated_gbs} = {mbs_val}[/yellow] × [yellow]{dp_val}[/yellow] × [yellow]{gas_val}[/yellow])")
            else:
                console.print(f"    [dim]Current:[/dim] global_batch_size([yellow]{gbs_val}[/yellow]) = [yellow]{mbs_val}[/yellow] × [yellow]{dp_val}[/yellow] × [yellow]{gas_val}[/yellow]")
            
            raise ValueError("❌ No solution found for the given parameters.")
        
        # Multiple solutions means the parameters are under-constrained
        if isinstance(solution, list) and len(solution) > 1:
            raise ValueError("❌ Multiple solutions found. Please provide more constraints or specific values.")
        if isinstance(solution, list) and len(solution) == 1:
            solution = solution[0]
        
        for key, value in solution.items():
            try:
                solution[key] = int(value)
            except (ValueError, TypeError):
                raise ValueError(f"❌ Invalid solution value for {key}: {value}. Expected an integer.")

        try:
            # Extract values from solution
            if total_gpus == "?":
                total_gpus = int(solution[total_gpus_sym])
            
            if tp == "?":
                tp = int(solution[tp_sym])
            
            if pp == "?":
                pp = int(solution[pp_sym])
            
            if dp == "?":
                dp = int(solution[dp_sym])
            
            if micro_batch_size == "?":
                micro_batch_size = int(solution[mbs_sym])
            
            if global_batch_size == "?":
                global_batch_size = int(solution[gbs_sym])
            
            if gradient_accumulation_steps == "?":
                gradient_accumulation_steps = int(solution[gas_sym])
        except (TypeError, KeyError) as e:
            raise ValueError(f"Failed to extract gradient_accumulation_steps from solution: {str(e)}")

        effective_batch_size_per_gpu = micro_batch_size * gradient_accumulation_steps
        total_effective_batch_size = global_batch_size

        # Estimate memory and communication
        estimated_memory_per_gpu_gb = self._estimate_memory_usage(
            tp, pp, model_params, sequence_length, micro_batch_size
        )
        communication_overhead = self._estimate_communication_overhead(tp, pp)

        return {
            'total_gpus': total_gpus,
            'model_params_billions': model_params,
            'sequence_length': sequence_length,
            'micro_batch_size': micro_batch_size,
            'global_batch_size': global_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'tensor_parallel': tp,
            'pipeline_parallel': pp,
            'data_parallel': dp,
            'effective_batch_size_per_gpu': effective_batch_size_per_gpu,
            'total_effective_batch_size': total_effective_batch_size,
            'estimated_memory_per_gpu_gb': estimated_memory_per_gpu_gb,
            'communication_overhead': communication_overhead,
        }
    
    def _provide_fallback_solution(self, total_gpus, tp, pp, dp, micro_batch_size, global_batch_size, gradient_accumulation_steps):
        """Provide reasonable fallback values when equation system cannot be solved"""
        fallback = {}
        
        # Use provided values or reasonable defaults
        total_gpus = total_gpus or 8
        tp = tp or 1
        pp = pp or 1
        dp = dp or (total_gpus // (tp * pp))
        micro_batch_size = micro_batch_size or 1
        global_batch_size = global_batch_size or 128
        gradient_accumulation_steps = gradient_accumulation_steps or max(1, global_batch_size // (micro_batch_size * dp))
        
        return {
            symbols('total_gpus'): total_gpus,
            symbols('tp'): tp,
            symbols('pp'): pp,
            symbols('dp'): dp,
            symbols('micro_batch_size'): micro_batch_size,
            symbols('global_batch_size'): global_batch_size,
            symbols('gradient_accumulation_steps'): gradient_accumulation_steps
        }
    
    def _estimate_memory_usage(self, tp: int, pp: int, model_params: int, 
                              seq_len: int, micro_batch: int) -> float:
        """Estimate memory usage per GPU (GB)"""
        model_memory = (model_params * 1e9 * 2) / (tp * 1e9)  # FP16
        activation_memory = (micro_batch * seq_len * model_params * 0.008) / tp
        optimizer_memory = model_memory * 1.5 if pp == 1 else model_memory * 0.5
        total_memory = model_memory + activation_memory + optimizer_memory + 2
        return round(total_memory, 1)
    
    def _estimate_communication_overhead(self, tp: int, pp: int) -> str:
        """Estimate communication overhead level"""
        if tp > 4 and pp > 2:
            return "Very High"
        elif tp > 2 and pp > 1:
            return "High" 
        elif tp > 1 or pp > 1:
            return "Medium"
        else:
            return "Low"
    
    def _display_results(self, config: Dict, param_sources: Dict[str, str]):
        """Display the final configuration results"""
        console.print("✅ [bold green]Configuration Complete![/bold green]\n")
        
        # Batch Size Parameters Table
        batch_table = Table(title="📥 Batch Size Parameters", box=box.ROUNDED, border_style="cyan")
        batch_table.add_column("Parameter", style="cyan", width=30)
        batch_table.add_column("Value", style="yellow", width=20)
        batch_table.add_column("Source", style="green", width=10)
        batch_table.add_column("Note", style="italic", width=55)
        
        batch_table.add_row("Micro Batch Size", str(config['micro_batch_size']), param_sources['micro_batch_size'], "Number of samples processed per GPU per forward pass")
        batch_table.add_row("Gradient Accumulation Steps", str(config['gradient_accumulation_steps']), param_sources['gradient_accumulation_steps'], "Number of forward/backward passes before update")
        batch_table.add_row("Global Batch Size", str(config['global_batch_size']), param_sources['global_batch_size'], "Total samples processed across all GPUs per update")
        batch_table.add_row("[italic]Effective Batch per GPU[/italic]", str(config['effective_batch_size_per_gpu']), "-", "Actual batch size processed by each GPU")
        batch_table.add_row("[italic]Total Effective Batch Size[/italic]", str(config['total_effective_batch_size']), "-", "Total batch size across entire system")
        
        console.print(batch_table)
        console.print()
        
        # Parallel Configuration Table
        parallel_table = Table(title="⚙️ Parallel Configuration", box=box.ROUNDED, border_style="green")
        parallel_table.add_column("Parameter", style="cyan", width=30)
        parallel_table.add_column("Value", style="yellow", width=20)
        parallel_table.add_column("Source", style="green", width=10)
        parallel_table.add_column("Note", style="italic", width=55)
        
        parallel_table.add_row("Total GPUs", str(config['total_gpus']), param_sources['total_gpus'], "Total number of GPUs used for training")
        parallel_table.add_row("Tensor Parallel (TP)", str(config['tensor_parallel']), param_sources['tensor_parallel'], "Number of GPUs for layer parallelism")
        parallel_table.add_row("Pipeline Parallel (PP)", str(config['pipeline_parallel']), param_sources['pipeline_parallel'], "Number of GPUs for model stage parallelism")
        parallel_table.add_row("Data Parallel (DP)", str(config['data_parallel']), param_sources['data_parallel'], "Number of GPUs for data parallelism")
        
        console.print(parallel_table)
        console.print()
        
        # Performance Metrics Table
        perf_table = Table(title="📊 Performance Metrics", box=box.ROUNDED, border_style="blue")
        perf_table.add_column("Parameter", style="cyan", width=30)
        perf_table.add_column("Value", style="yellow", width=20)
        perf_table.add_column("Source", style="green", width=10)
        perf_table.add_column("Note", style="italic", width=55)
        
        perf_table.add_row("Model Size", f"{self.parameters['model_size']['value']} ({config['model_params_billions']}B)", param_sources['model_size'], "Size of model in billions of parameters")
        perf_table.add_row("Sequence Length", str(config['sequence_length']), param_sources['sequence_length'], "Maximum length of input sequences")
        perf_table.add_row("[italic]Estimated Memory per GPU[/italic]", f"{config['estimated_memory_per_gpu_gb']} GB", "-", "Approximate memory required per GPU")
        perf_table.add_row("[italic]Communication Overhead[/italic]", config['communication_overhead'], "-", "Level of inter-GPU communication required")
        
        console.print(perf_table)
        console.print()
        
        # Formula verification
        formula_panel = Panel(
            f"[bold green]✅ Formula Verification:[/bold green]\n"
            f"[cyan]GBS = MBS × DP × GAS[/cyan]\n"
            f"{config['global_batch_size']} = {config['micro_batch_size']} × {config['data_parallel']} × {config['gradient_accumulation_steps']}\n"
            f"[cyan]Total GPUs = TP × PP × DP[/cyan]\n"
            f"{config['total_gpus']} = {config['tensor_parallel']} × {config['pipeline_parallel']} × {config['data_parallel']}",
            border_style="green",
            title="🔍 Validation",
            width=128
        )
        console.print(formula_panel)

def main():
    """Main function"""
    try:
        # Create calculator once and reuse it
        calculator = InteractiveMegatronCalculator()
        first_run = True
        
        while True:
            try:
                if not first_run:
                    # Add visual separator between runs
                    console.clear()
                    console.print("🔄 [bold cyan]Starting new calculation with previous values as defaults[/bold cyan]\n")
                else:
                    first_run = False
                    
                # Run calculator and get configuration
                config = calculator.run()
                
                # Create mapping between config keys and parameter keys
                config_and_param_keys = [
                    'total_gpus',
                    'micro_batch_size',
                    'global_batch_size',
                    'gradient_accumulation_steps',
                    'tensor_parallel',
                    'pipeline_parallel',
                    'data_parallel',
                    'sequence_length'
                ]
                
                # Update default values for next run
                for key in config_and_param_keys:
                    if key in config and key in calculator.parameters:
                        calculator.parameters[key]['default'] = config[key]
                
                # Special case for model size
                if 'model_params_billions' in config:
                    model_size_str = f"{config['model_params_billions']}B"
                    calculator.parameters['model_size']['default'] = model_size_str
                
                # Reset all parameter values to None for next run
                for param in calculator.parameters.values():
                    param['value'] = None
                
                # Ask if user wants to calculate again
                if not Confirm.ask("\n🔄 Would you like to calculate another configuration?", default=True):
                    console.print("\n👋 [yellow]Goodbye![/yellow]")
                    break
                    
            except KeyboardInterrupt:
                console.print("\n👋 [yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n❌ [red]Error: {e}[/red]")
                if not Confirm.ask("🔄 Would you like to continue and try again?", default=True):
                    console.print("\n👋 [yellow]Goodbye![/yellow]")
                    break
                # Reset all parameter values to None for retry
                for param in calculator.parameters.values():
                    param['value'] = None
                continue
                
    except KeyboardInterrupt:
        console.print("\n👋 [yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n❌ [red]Fatal Error: {e}[/red]")

if __name__ == "__main__":
    main()
