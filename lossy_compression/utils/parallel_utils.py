"""
Utilities for parallel execution of LLM/SLM calls.
"""

import concurrent.futures
import asyncio
from typing import List, Tuple, Dict, Callable, Any
import time
from functools import wraps

# Configuration
MAX_WORKERS = 3  # Limit concurrent API calls to avoid rate limits
RATE_LIMIT_DELAY = 0.1  # Small delay between calls to avoid rate limits


def rate_limited(func):
    """Add rate limiting to function calls."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        time.sleep(RATE_LIMIT_DELAY)
        return func(*args, **kwargs)

    return wrapper


def parallel_execute(tasks: List[Tuple[Callable, tuple, dict]],
                     max_workers: int = MAX_WORKERS) -> List[Any]:
    """
    Execute multiple functions in parallel.
    
    Args:
        tasks: List of (function, args, kwargs) tuples
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of results in the same order as tasks
    """
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        # Submit all tasks
        futures = []
        for func, args, kwargs in tasks:
            future = executor.submit(func, *args, **kwargs)
            futures.append(future)

        # Collect results in order
        results = []
        for future in futures:
            results.append(future.result())

        return results


def parallel_map(func: Callable,
                 items: List[Any],
                 max_workers: int = MAX_WORKERS) -> List[Any]:
    """
    Apply a function to multiple items in parallel.
    
    Args:
        func: Function to apply
        items: List of items to process
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of results
    """
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    return results


class BatchProcessor:
    """Process multiple prompts in parallel batches."""

    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers

    def process_batch(self, prompts: List[str], process_func: Callable,
                      **kwargs) -> List[Any]:
        """
        Process a batch of prompts in parallel.
        
        Args:
            prompts: List of prompts to process
            process_func: Function to process each prompt
            **kwargs: Additional arguments for process_func
        
        Returns:
            List of results
        """
        tasks = [(process_func, (prompt, ), kwargs) for prompt in prompts]
        return parallel_execute(tasks, self.max_workers)
