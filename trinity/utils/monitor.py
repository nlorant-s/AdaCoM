"""Monitor"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
except ImportError:
    mlflow = None
from torch.utils.tensorboard import SummaryWriter

from trinity.common.config import Config
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

MONITOR = Registry("monitor")


def gather_metrics(
    metric_list: List[Dict], prefix: str, output_stats: List[str] = ["mean", "max", "min"]
) -> Dict:
    if not metric_list:
        return {}
    try:
        df = pd.DataFrame(metric_list)
        numeric_df = df.select_dtypes(include=[np.number])
        # Skip columns that are already std@k — aggregating std further is meaningless
        primary_cols = [c for c in numeric_df.columns if "/std@" not in c and not c.startswith("std@")]
        stats_df = numeric_df[primary_cols].agg(output_stats)
        metric = {}
        for col in stats_df.columns:
            for stats in output_stats:
                metric[f"{prefix}/{col}/{stats}"] = stats_df.loc[stats, col].item()
        return metric
    except Exception as e:
        raise ValueError(f"Failed to gather metrics: {e}") from e


class Monitor(ABC):
    """Monitor"""

    def __init__(
        self,
        project: str,
        name: str,
        role: str,
        config: Config = None,  # pass the global Config for recording
    ) -> None:
        self.project = project
        self.name = name
        self.role = role
        self.config = config

    @abstractmethod
    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        """Log a table"""

    @abstractmethod
    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""

    @abstractmethod
    def close(self) -> None:
        """Close the monitor"""

    def __del__(self) -> None:
        self.close()

    def calculate_metrics(
        self, data: dict[str, Union[List[float], float]], prefix: Optional[str] = None
    ) -> dict[str, float]:
        metrics = {}
        for key, val in data.items():
            if prefix is not None:
                key = f"{prefix}/{key}"

            if isinstance(val, List):
                if len(val) > 1:
                    metrics[f"{key}/mean"] = np.mean(val)
                    metrics[f"{key}/max"] = np.amax(val)
                    metrics[f"{key}/min"] = np.amin(val)
                elif len(val) == 1:
                    metrics[key] = val[0]
            else:
                metrics[key] = val
        return metrics

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {}


@MONITOR.register_module("tensorboard")
class TensorboardMonitor(Monitor):
    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        self.tensorboard_dir = os.path.join(config.monitor.cache_dir, "tensorboard", role)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.logger = SummaryWriter(self.tensorboard_dir)
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        pass

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        for key in data:
            self.logger.add_scalar(key, data[key], step)
        self.console_logger.info(f"Step {step}: {data}")

    def close(self) -> None:
        self.logger.close()


@MONITOR.register_module("wandb")
class WandbMonitor(Monitor):
    """Monitor with Weights & Biases.

    Args:
        base_url (`Optional[str]`): The base URL of the W&B server. If not provided, use the environment variable `WANDB_BASE_URL`.
        api_key (`Optional[str]`): The API key for W&B. If not provided, use the environment variable `WANDB_API_KEY`.
    """

    # Disable wandb after this many consecutive failures, to avoid per-step network
    # stalls killing throughput when the wandb backend is unreachable for an extended period.
    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        assert wandb is not None, "wandb is not installed. Please install it to use WandbMonitor."
        if not group:
            group = name
        monitor_args = config.monitor.monitor_args or {}
        if base_url := monitor_args.get("base_url"):
            os.environ["WANDB_BASE_URL"] = base_url
        if api_key := monitor_args.get("api_key"):
            os.environ["WANDB_API_KEY"] = api_key
        self.console_logger = get_logger(__name__, in_ray_actor=True)
        self._disabled = False
        self._consecutive_failures = 0
        self.logger = None
        init_timeouts = [120, 240, 480]
        for attempt, timeout in enumerate(init_timeouts, start=1):
            try:
                self.logger = wandb.init(
                    project=project,
                    group=group,
                    name=f"{name}_{role}",
                    tags=[role],
                    config=config,
                    save_code=False,
                    settings=wandb.Settings(init_timeout=timeout),
                )
                if attempt > 1:
                    self.console_logger.info(
                        f"wandb.init succeeded on attempt {attempt} (timeout={timeout}s)."
                    )
                break
            except Exception as e:
                if attempt < len(init_timeouts):
                    self.console_logger.warning(
                        f"wandb.init attempt {attempt} failed (timeout={timeout}s, retrying): {e}"
                    )
                else:
                    self._disabled = True
                    self.console_logger.warning(
                        f"wandb.init failed after {attempt} attempts, disabling wandb for this run "
                        f"(training continues): {e}"
                    )

    def _note_failure(self, op: str, err: Exception) -> None:
        self._consecutive_failures += 1
        self.console_logger.warning(
            f"wandb {op} failed (#{self._consecutive_failures}, non-fatal): {err}"
        )
        if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
            self._disabled = True
            self.console_logger.warning(
                f"wandb disabled after {self._consecutive_failures} consecutive failures; "
                "subsequent logs will be skipped for the rest of this run."
            )

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        if self._disabled or self.logger is None:
            return
        try:
            table = wandb.Table(dataframe=experiences_table)
            self.logger.log({table_name: table}, step=step, commit=False)
            self._consecutive_failures = 0
        except Exception as e:
            self._note_failure("log_table", e)

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        # Write console log first so metrics are preserved even if wandb fails.
        self.console_logger.info(f"Step {step}: {data}")
        if self._disabled or self.logger is None:
            return
        try:
            self.logger.log(data, step=step, commit=commit)
            self._consecutive_failures = 0
        except Exception as e:
            self._note_failure("log", e)

    def close(self) -> None:
        if self.logger is None:
            return
        try:
            self.logger.finish()
        except Exception as e:
            self.console_logger.warning(f"wandb.finish failed (non-fatal): {e}")
        finally:
            self.logger = None
            self._disabled = True

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {
            "base_url": None,
            "api_key": None,
        }


@MONITOR.register_module("mlflow")
class MlflowMonitor(Monitor):
    """Monitor with MLflow.

    Args:
        uri (`Optional[str]`): The tracking server URI. If not provided, the default is `http://localhost:5000`.
        username (`Optional[str]`): The username to login. If not provided, the default is `None`.
        password (`Optional[str]`): The password to login. If not provided, the default is `None`.
    """

    def __init__(
        self, project: str, group: str, name: str, role: str, config: Config = None
    ) -> None:
        assert (
            mlflow is not None
        ), "mlflow is not installed. Please install it to use MlflowMonitor."
        monitor_args = config.monitor.monitor_args or {}
        if username := monitor_args.get("username"):
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
        if password := monitor_args.get("password"):
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        mlflow.set_tracking_uri(config.monitor.monitor_args.get("uri", "http://localhost:5000"))
        mlflow.set_experiment(project)
        mlflow.enable_system_metrics_logging()
        mlflow.start_run(
            run_name=f"{name}_{role}",
            tags={
                "group": group,
                "role": role,
            },
        )
        mlflow.log_params(config.flatten())
        self.console_logger = get_logger(__name__, in_ray_actor=True)

    def log_table(self, table_name: str, experiences_table: pd.DataFrame, step: int):
        experiences_table["step"] = step
        mlflow.log_table(data=experiences_table, artifact_file=f"{table_name}.json")

    def log(self, data: dict, step: int, commit: bool = False) -> None:
        """Log metrics."""
        self.console_logger.info(f"Step {step}: {data}")
        # Replace all '@' in keys with '_at_', as MLflow does not support '@' in metric names
        data = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=data, step=step)

    def close(self) -> None:
        mlflow.end_run()

    @classmethod
    def default_args(cls) -> Dict:
        """Return default arguments for the monitor."""
        return {
            "uri": "http://localhost:5000",
            "username": None,
            "password": None,
        }
