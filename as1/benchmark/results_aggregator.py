# -*- coding: utf-8 -*-
"""Results aggregation for MCP-Bench.

Ported from mcp-bench project. Calculates summary statistics across tasks.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregates benchmark results across multiple tasks."""

    @staticmethod
    def safe_avg(values: List[Any]) -> float:
        """Average numeric values, filtering out None."""
        filtered = [v for v in values if v is not None and isinstance(v, (int, float))]
        return sum(filtered) / len(filtered) if filtered else 0.0

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate evaluation results from multiple tasks.

        Args:
            results: List of per-task result dicts, each with an 'evaluation' key
                containing the 6-dimension scores and metrics.

        Returns:
            Dict with aggregated statistics across all dimensions.
        """
        completed = [r for r in results if r.get("status") == "completed" and r.get("evaluation")]
        total = len(results)

        if not completed:
            return self._empty_summary(total)

        # Collect scores
        tf, gr, ta, pa, da, pe = [], [], [], [], [], []
        exec_times, rounds_list, tool_counts = [], [], []
        server_counts = []
        cross_server = 0

        for r in completed:
            ev = r["evaluation"]
            tf.append(ev.get("task_fulfillment", 0))
            gr.append(ev.get("grounding", 0))
            ta.append(ev.get("tool_appropriateness", 0))
            pa.append(ev.get("parameter_accuracy", 0))
            da.append(ev.get("dependency_awareness", 0))
            pe.append(ev.get("parallelism_and_efficiency", 0))

            exec_times.append(r.get("execution_time", 0))
            rounds_list.append(r.get("total_rounds", 0))
            tool_counts.append(r.get("total_tool_calls", 0))

            sm = ev.get("server_utilization_metrics", {})
            server_counts.append(sm.get("server_count", 0))
            if sm.get("cross_server_coordination"):
                cross_server += 1

        # Combined scores
        combined = []
        for i in range(len(tf)):
            combined.append((tf[i] + gr[i] + ta[i] + pa[i] + da[i] + pe[i]) / 6)

        return {
            "task_statistics": {
                "total_tasks": total,
                "completed_tasks": len(completed),
                "failed_tasks": total - len(completed),
                "completion_rate": len(completed) / total if total > 0 else 0.0,
            },
            "llm_judge_metrics": {
                "avg_task_fulfillment": self.safe_avg(tf),
                "avg_grounding": self.safe_avg(gr),
                "avg_tool_appropriateness": self.safe_avg(ta),
                "avg_parameter_accuracy": self.safe_avg(pa),
                "avg_dependency_awareness": self.safe_avg(da),
                "avg_parallelism_efficiency": self.safe_avg(pe),
                "avg_combined": self.safe_avg(combined),
            },
            "performance_metrics": {
                "avg_execution_time": self.safe_avg(exec_times),
                "avg_total_rounds": self.safe_avg(rounds_list),
                "avg_tool_calls_per_task": self.safe_avg(tool_counts),
            },
            "server_utilization_metrics": {
                "avg_servers_used": self.safe_avg(server_counts),
                "cross_server_rate": cross_server / len(completed) if completed else 0.0,
            },
        }

    def format_summary(self, agg: Dict[str, Any]) -> str:
        """Format aggregated results as a readable summary string."""
        jm = agg.get("llm_judge_metrics", {})
        ts = agg.get("task_statistics", {})
        pm = agg.get("performance_metrics", {})

        lines = [
            "=" * 60,
            "MCP-BENCH EVALUATION SUMMARY",
            "=" * 60,
            f"Tasks: {ts.get('completed_tasks', 0)}/{ts.get('total_tasks', 0)} completed "
            f"({ts.get('completion_rate', 0):.1%})",
            "",
            "LLM Judge Scores (1-10):",
            f"  Task Fulfillment:     {jm.get('avg_task_fulfillment', 0):.2f}",
            f"  Grounding:            {jm.get('avg_grounding', 0):.2f}",
            f"  Tool Appropriateness: {jm.get('avg_tool_appropriateness', 0):.2f}",
            f"  Parameter Accuracy:   {jm.get('avg_parameter_accuracy', 0):.2f}",
            f"  Dependency Awareness: {jm.get('avg_dependency_awareness', 0):.2f}",
            f"  Parallelism:          {jm.get('avg_parallelism_efficiency', 0):.2f}",
            f"  Combined Average:     {jm.get('avg_combined', 0):.2f}",
            "",
            f"Avg Execution Time: {pm.get('avg_execution_time', 0):.1f}s",
            f"Avg Rounds/Task:    {pm.get('avg_total_rounds', 0):.1f}",
            f"Avg Tool Calls:     {pm.get('avg_tool_calls_per_task', 0):.1f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    @staticmethod
    def _empty_summary(total: int) -> Dict[str, Any]:
        return {
            "task_statistics": {
                "total_tasks": total,
                "completed_tasks": 0,
                "failed_tasks": total,
                "completion_rate": 0.0,
            },
            "llm_judge_metrics": {
                "avg_task_fulfillment": 0.0,
                "avg_grounding": 0.0,
                "avg_tool_appropriateness": 0.0,
                "avg_parameter_accuracy": 0.0,
                "avg_dependency_awareness": 0.0,
                "avg_parallelism_efficiency": 0.0,
                "avg_combined": 0.0,
            },
            "performance_metrics": {
                "avg_execution_time": 0.0,
                "avg_total_rounds": 0.0,
                "avg_tool_calls_per_task": 0.0,
            },
            "server_utilization_metrics": {
                "avg_servers_used": 0.0,
                "cross_server_rate": 0.0,
            },
        }
