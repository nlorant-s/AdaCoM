import os
import json
import logging
import hashlib
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import random
import string


class ExperimentLogger:
    """Logger system for managing experiment results with LLM invocation tracking."""
    
    def __init__(self, base_dir: str = "./benchmark_results", 
                 test_mem_config: str = None, 
                 dataset: str = None):
        """
        Initialize the ExperimentLogger.
        
        Args:
            base_dir: Base directory for storing results (default: ./benchmark_results)
            test_mem_config: Test memory configuration name (e.g., "mem_config_v1")
            dataset: Dataset name (e.g., "Browsecamp", "GAIA_web")
        """
        self.base_dir = Path(base_dir)
        self.test_mem_config = test_mem_config
        self.dataset = dataset
        
        # Build experiment directory path
        if test_mem_config and dataset:
            self.experiment_dir = self.base_dir / test_mem_config / dataset
        elif test_mem_config:
            self.experiment_dir = self.base_dir / test_mem_config
        else:
            # Fallback to timestamp-based directory if no config provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = self.base_dir / f"experiment_{timestamp}"
        
        self.current_run_dir = None
        self.run_number = None
        self.invoke_dir = None
        self.logger = None
        self.file_handler = None
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def start_run(self, run_id: Union[int, str], clear_existing: bool = True) -> Path:
        """
        Start a new run and create its directory structure.

        Args:
            run_id: Run ID (corresponds to example/test case ID, can be int or query_id string)
            clear_existing: If True, clear existing directory before starting (default: True)

        Returns:
            Path to the run directory
        """
        # Use the provided run_id directly (convert to string for path compatibility)
        self.run_number = str(run_id)
        
        # Create run directory
        self.current_run_dir = self.experiment_dir / str(self.run_number)
        
        # Clear existing directory if requested
        if clear_existing and self.current_run_dir.exists():
            import shutil
            import time
            # Robust directory removal with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(self.current_run_dir)
                    break
                except OSError as e:
                    if attempt < max_retries - 1:
                        # Wait a bit and retry
                        time.sleep(0.5)
                    else:
                        # On final attempt, log warning and try to continue
                        import warnings
                        warnings.warn(f"Failed to remove directory {self.current_run_dir}: {e}. Will try to overwrite files.")
                        # Don't raise - let mkdir handle it
                        break
            
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create invoke directory
        self.invoke_dir = self.current_run_dir / "invoke"
        self.invoke_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Started run {self.run_number} in {self.current_run_dir}")
        self.logger.info(f"Config: {self.test_mem_config}, Dataset: {self.dataset}")
        
        return self.current_run_dir
    
    def _setup_logging(self):
        """Setup logging configuration for the current run."""
        if self.logger:
            # Remove old handlers
            if self.file_handler:
                self.logger.removeHandler(self.file_handler)
                self.file_handler.close()
        
        # Create logger
        self.logger = logging.getLogger(f"experiment_run_{self.run_number}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create file handler
        log_file = self.current_run_dir / "logging.log"
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(self.file_handler)
        
        # Also add console handler
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(formatter)
        # self.logger.addHandler(console_handler)
    
    def log_llm_invocation(self, 
                          model_class: str,
                          model_name: str,
                          messages: List[Dict],
                          response: Any = None,
                          **kwargs) -> str:
        """
        Log an LLM invocation to a JSON file.
        
        Args:
            model_class: Class name of the model wrapper
            model_name: Name of the model
            messages: List of message dictionaries
            response: Response from the model
            **kwargs: Additional arguments to log
            
        Returns:
            Path to the saved invocation file
        """
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Generate timestamp and random suffix
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        
        # Create filename
        filename = f"model_{model_class}_{timestamp}_{suffix}.json"
        filepath = self.invoke_dir / filename
        
        # Prepare invocation data
        invocation_data = {
            "model_class": model_class,
            "timestamp": timestamp,
            "arguments": {
                "model": model_name,
                "messages": messages,
                **kwargs
            }
        }
        
        if response is not None:
            invocation_data["response"] = self._serialize_response(response)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(invocation_data, f, indent=4, ensure_ascii=False)
        
        # self.logger.info(f"Logged LLM invocation to {filename}")
        
        return str(filepath)
    
    def _serialize_response(self, response: Any) -> Any:
        """
        Serialize response object to JSON-compatible format.
        
        Args:
            response: Response object from LLM
            
        Returns:
            JSON-serializable representation
        """
        try:
            if response is None:
                return None
            elif isinstance(response, (str, int, float, bool)):
                return response
            elif isinstance(response, dict):
                # This handles ChatResponse (which inherits from dict) and regular dicts
                result = {}
                for k, v in response.items():
                    try:
                        result[k] = self._serialize_response(v)
                    except Exception:
                        result[k] = str(v)
                return result
            elif isinstance(response, (list, tuple)):
                return [self._serialize_response(item) for item in response]
            elif hasattr(response, '__dict__'):
                # Convert object to dictionary, including all attributes
                result = {}
                for k, v in response.__dict__.items():
                    if not k.startswith('_'):
                        try:
                            result[k] = self._serialize_response(v)
                        except Exception:
                            # If serialization fails for an attribute, convert to string
                            result[k] = str(v)
                # Also try to get common attributes directly
                for attr in ['content', 'message', 'text', 'choices', 'usage', 'metadata']:
                    if hasattr(response, attr) and attr not in result:
                        try:
                            result[attr] = self._serialize_response(getattr(response, attr))
                        except Exception:
                            result[attr] = str(getattr(response, attr))
                return result
            else:
                return str(response)
        except Exception as e:
            # If all else fails, return a string representation with debug info
            return {
                "serialization_error": str(e),
                "response_type": str(type(response)),
                "response_str": str(response)
            }
    
    def log_info(self, message: str):
        """Log info message."""
        if self.logger:
            self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        if self.logger:
            self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message."""
        if self.logger:
            self.logger.error(message)
    
    def log_debug(self, message: str):
        """Log debug message."""
        if self.logger:
            self.logger.debug(message)
    
    def save_memory(self, memory_data: Dict[str, Any]):
        """
        Save memory data to memory.json file.
        
        Args:
            memory_data: Dictionary containing memory information
        """
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call start_run() first.")
        
        memory_file = self.current_run_dir / "memory.json"
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Saved memory to {memory_file}")
    
    def save_report(self, report_data: Dict[str, Any], filename: str = "detailed_report.json"):
        """
        Save report data to JSON file.
        
        Args:
            report_data: Dictionary containing report information
            filename: Name of the report file
        """
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call start_run() first.")
        
        report_file = self.current_run_dir / filename
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Saved report to {report_file}")
    
    def end_run(self):
        """End the current run and cleanup resources."""
        if self.logger:
            self.logger.info(f"Ended run {self.run_number}")
            
            # Close file handler
            if self.file_handler:
                self.logger.removeHandler(self.file_handler)
                self.file_handler.close()
                self.file_handler = None
        
        self.current_run_dir = None
        self.run_number = None
        self.invoke_dir = None
        self.logger = None
    
    def get_all_runs(self) -> List[int]:
        """
        Get list of all run IDs in the experiment.
        
        Returns:
            List of run IDs
        """
        runs = []
        if self.experiment_dir.exists():
            for d in self.experiment_dir.iterdir():
                if d.is_dir() and d.name.isdigit():
                    runs.append(int(d.name))
        return sorted(runs)
    
    def get_run_dir(self, run_id: int) -> Path:
        """
        Get the directory path for a specific run.
        
        Args:
            run_id: The run ID
            
        Returns:
            Path to the run directory
        """
        return self.experiment_dir / str(run_id)
    
    def load_run_invocations(self, run_id: int) -> List[Dict]:
        """
        Load all LLM invocations from a specific run.
        
        Args:
            run_id: Run ID to load invocations from
            
        Returns:
            List of invocation dictionaries
        """
        run_dir = self.experiment_dir / str(run_id)
        invoke_dir = run_dir / "invoke"
        
        if not invoke_dir.exists():
            return []
        
        invocations = []
        for json_file in sorted(invoke_dir.glob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                invocations.append(json.load(f))
        
        return invocations