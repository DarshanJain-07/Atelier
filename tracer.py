import asyncio
import os
import sys
import threading

import pandas as pd
import torch
from fastapi import BackgroundTasks

from main import RunProfile, SimulationRequest, run_simulation

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

class LinearTracer:
    def __init__(self):
        # We use thread-local storage to keep track of call depth per thread
        self.local = threading.local()

    def get_state(self):
        if not hasattr(self.local, "depth"):
            self.local.depth = 0
            self.local.call_stack = []
        return self.local

    def format_value(self, v):
        if isinstance(v, torch.Tensor):
            if v.is_sparse:
                return f"SparseTensor{list(v.shape)}[{v.dtype}]"
            return f"Tensor{list(v.shape)}[{v.dtype}]"
        if isinstance(v, pd.DataFrame):
            return f"DataFrame{v.shape}"
        if isinstance(v, dict):
            return f"dict(len={len(v)})"
        if isinstance(v, (list, tuple)):
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                return f"{type(v).__name__} of Tensors(len={len(v)})"
            return f"{type(v).__name__}(len={len(v)})"
        if type(v).__module__ == "builtins":
            if isinstance(v, (int, float, str, bool)):
                rep = repr(v)
                return rep if len(rep) < 20 else rep[:17] + "..."
            return type(v).__name__
        return v.__class__.__name__

    def trace(self, frame, event, arg):
        if event not in ("call", "return"):
            return self.trace

        co = frame.f_code
        filename = co.co_filename
        func_name = co.co_name
        base_filename = os.path.basename(filename)

        # 1. Broadly target everything in the project root, ignoring venv and site-packages
        if PROJECT_ROOT not in filename or "site-packages" in filename or "venv" in filename or ".venv" in filename:
            return self.trace

        # 2. Ignore the tracer script itself
        if base_filename == "trace_real_simulation.py":
            return self.trace

        # 3. Keep single underscore methods (e.g., _neighbor_average, _calculate_threshold),
        # but filter out double-underscore python internal methods (dunders) and list comprehensions
        if func_name.startswith("<") or func_name.startswith("__"):
            return self.trace

        state = self.get_state()

        if event == "call":
            arg_count = co.co_argcount
            arg_names = co.co_varnames[:arg_count]
            locals_dict = frame.f_locals

            args_str = []
            for name in arg_names:
                if name in locals_dict and name != "self" and name != "cls":
                    val = locals_dict[name]
                    args_str.append(f"{name}={self.format_value(val)}")

            indent = "  " * state.depth
            module_name = base_filename.replace(".py", "")

            t_name = threading.current_thread().name
            t_prefix = f"[{t_name}]" if t_name != "MainThread" else ""

            print(f"{t_prefix}{indent}▶ [{module_name}] {func_name}({', '.join(args_str)})")
            state.depth += 1
            state.call_stack.append(func_name)

        elif event == "return":
            if state.call_stack and state.call_stack[-1] == func_name:
                state.call_stack.pop()
                state.depth -= 1
                indent = "  " * state.depth
                ret_str = self.format_value(arg)

                t_name = threading.current_thread().name
                t_prefix = f"[{t_name}]" if t_name != "MainThread" else ""

                print(f"{t_prefix}{indent}◀ Return: {ret_str}")

        return self.trace

async def run():
    print("Setting up FULL deep simulation request...")
    req = SimulationRequest(
        news_text="A massive data breach at a major global bank has leaked the life savings details of millions of people, sparking outrage and panic.",
        runs=[
            RunProfile(
                agent_count=50,
                seed=42,
                use_agent_memory=True,
                use_network_topology=True,
                use_algorithmic_amplification=True,
                enable_evolution=True,
            ),
        ],
    )
    bg_tasks = BackgroundTasks()

    tracer = LinearTracer()
    print("\n" + "="*120)
    print("STARTING DEEP LINEAR DATA FLOW TRACE (ALL PROJECT FUNCTIONS)")
    print("="*120)

    sys.setprofile(tracer.trace)
    threading.setprofile(tracer.trace)
    try:
        await run_simulation(req, bg_tasks)
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        sys.setprofile(None)
        threading.setprofile(None)

    print("="*120)
    print("TRACE COMPLETE")
    print("="*120)

if __name__ == "__main__":
    asyncio.run(run())
