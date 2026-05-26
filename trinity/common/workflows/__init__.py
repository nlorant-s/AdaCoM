# -*- coding: utf-8 -*-
"""Workflow module"""
from trinity.common.workflows.envs.browse_comp_plus.bcp_simple_react_workflow import (
    BCPSimpleToolReActWorkflow,
)
from trinity.common.workflows.mcp_bench_workflow import (
    MCPBenchWorkflow,
)
from trinity.common.workflows.eval_workflow import (
    AsyncMathEvalWorkflow,
    MathEvalWorkflow,
)
from trinity.common.workflows.workflow import (
    WORKFLOWS,
    AsyncMathWorkflow,
    AsyncSimpleWorkflow,
    MathWorkflow,
    SimpleWorkflow,
    Task,
    Workflow,
)

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
    "AsyncSimpleWorkflow",
    "SimpleWorkflow",
    "AsyncMathWorkflow",
    "MathWorkflow",
    "AsyncMathEvalWorkflow",
    "MathEvalWorkflow",
    "BCPSimpleToolReActWorkflow",
    "MCPBenchWorkflow",
]
