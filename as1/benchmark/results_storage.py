# -*- coding: utf-8 -*-
"""Structured file storage for benchmark results.

Ported from mcp-bench project.

Directory structure:
  benchmark_results/
    {model_name}_{timestamp}/
      evaluation_result.json    # aggregated metrics
      evaluation_meta.json      # benchmark configuration
      {task_id}/
        solution.json           # task execution result
        evaluation/
          judge_score.json      # LLM judge evaluation
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class BenchmarkResultsStorage:
    """File system based storage for benchmark results."""

    SOLUTION_FILE_NAME = "solution.json"
    EVALUATION_DIR_NAME = "evaluation"
    EVALUATION_RESULT_FILE = "evaluation_result.json"
    EVALUATION_META_FILE = "evaluation_meta.json"

    def __init__(
        self,
        base_dir: str = "benchmark_results",
        model_name: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.base_dir = base_dir
        self.model_name = model_name or "unknown_model"
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
        self.run_dir = os.path.join(base_dir, f"{safe_model_name}_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

    def _get_task_path(self, task_id: str, *args: str) -> str:
        return os.path.join(self.run_dir, task_id, *args)

    def save_task_result(self, task_id: str, result: Dict[str, Any]) -> str:
        """Save task execution result (solution)."""
        path = self._get_task_path(task_id, self.SOLUTION_FILE_NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return path

    def save_task_evaluation(
        self, task_id: str, evaluation: Dict[str, Any], metric_name: str = "judge_score",
    ) -> str:
        """Save task evaluation result."""
        path = self._get_task_path(task_id, self.EVALUATION_DIR_NAME, f"{metric_name}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        return path

    def save_aggregated_results(self, results: Dict[str, Any]) -> str:
        """Save aggregated evaluation results."""
        path = os.path.join(self.run_dir, self.EVALUATION_RESULT_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return path

    def save_meta_info(self, meta_info: Dict[str, Any]) -> str:
        """Save benchmark configuration and meta information."""
        path = os.path.join(self.run_dir, self.EVALUATION_META_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)
        return path

    def task_result_exists(self, task_id: str) -> bool:
        """Check if a task result already exists."""
        path = self._get_task_path(task_id, self.SOLUTION_FILE_NAME)
        return os.path.exists(path) and os.path.getsize(path) > 0

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load a task result from storage."""
        path = self._get_task_path(task_id, self.SOLUTION_FILE_NAME)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_run_directory(self) -> str:
        return self.run_dir
