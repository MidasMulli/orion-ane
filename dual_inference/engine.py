"""
Dual-Path Inference Engine
==========================

Smart routing layer that uses both ANE and GPU on Apple Silicon.
Zero-cost routing via static rules. ANE only fires in parallel with GPU work.

"Your Mac has two AI processors. Most agent frameworks only use one. This uses both."
"""

import os
import sys
import time
import gc
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# Add parent dir for real_draft
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'speculative'))


# ── Data Types ───────────────────────────────────────────────────────

class ComputePath(Enum):
    GPU = "gpu"
    ANE = "ane"
    AUTO = "auto"


@dataclass
class Task:
    prompt: str
    max_tokens: int = 50
    task_type: str = "unknown"          # classify, extract, analyze, generate, etc.
    force_path: Optional[ComputePath] = None
    expected_output_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    text: str
    elapsed_ms: float
    path: ComputePath
    task: Task
    tokens_generated: int = 0


@dataclass
class ExecutionPlan:
    gpu_tasks: List[Task]
    ane_tasks: List[Task]
    parallel: bool = False


# ── Layer 1: Task Classifier (zero cost) ─────────────────────────────

class TaskClassifier:
    """Classifies tasks by complexity using zero-cost heuristics."""

    SIMPLE_TYPES = {
        "classify", "extract", "extract_field", "validate",
        "validate_format", "yes_no", "keyword_match",
        "route_intent", "label", "tag", "check",
        "summarize_short", "count", "list_fields",
    }

    COMPLEX_TYPES = {
        "analyze", "generate", "generate_long", "compare",
        "explain", "code", "negotiate", "draft", "reason",
        "synthesize", "critique", "recommend",
    }

    # Keywords suggesting simple tasks
    SIMPLE_KEYWORDS = [
        "classify", "is this", "yes or no", "true or false",
        "which category", "what type", "standard or non-standard",
        "extract the", "what is the", "list the",
        "one word", "one phrase", "briefly",
    ]

    # Keywords suggesting complex tasks
    COMPLEX_KEYWORDS = [
        "analyze", "explain why", "compare", "implications",
        "draft a memo", "provide detailed", "risk assessment",
        "step by step", "evaluate whether", "non-standard",
        "recommendations", "in detail", "comprehensive",
    ]

    def classify(self, task: Task) -> ComputePath:
        """Classify task → GPU or ANE candidate. Zero inference cost."""

        # Explicit override
        if task.force_path:
            return task.force_path

        # Task type hint
        task_type = task.task_type.lower()
        if task_type in self.SIMPLE_TYPES:
            return ComputePath.ANE
        if task_type in self.COMPLEX_TYPES:
            return ComputePath.GPU

        # Expected output length
        expected = task.expected_output_tokens or task.max_tokens
        if expected <= 20:
            return ComputePath.ANE
        if expected > 100:
            return ComputePath.GPU

        # Keyword matching on prompt
        prompt_lower = task.prompt.lower()
        simple_score = sum(1 for kw in self.SIMPLE_KEYWORDS if kw in prompt_lower)
        complex_score = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in prompt_lower)

        if simple_score > complex_score and simple_score > 0:
            return ComputePath.ANE
        if complex_score > 0:
            return ComputePath.GPU

        # Prompt length heuristic
        if len(task.prompt) < 100 and expected < 50:
            return ComputePath.ANE

        # Default: GPU (conservative — wrong GPU wastes nothing, wrong ANE = garbage)
        return ComputePath.GPU


# ── Layer 2: Parallel Scheduler ──────────────────────────────────────

class ParallelScheduler:
    """Identifies parallel execution opportunities."""

    def __init__(self):
        self.classifier = TaskClassifier()

    def schedule(self, tasks: List[Task]) -> ExecutionPlan:
        gpu_tasks = []
        ane_tasks = []

        for task in tasks:
            path = self.classifier.classify(task)
            if path == ComputePath.ANE:
                ane_tasks.append(task)
            else:
                gpu_tasks.append(task)

        # CRITICAL RULE: ANE never runs alone.
        # If no GPU work exists, everything goes to GPU.
        if not gpu_tasks:
            gpu_tasks = ane_tasks
            ane_tasks = []

        return ExecutionPlan(
            gpu_tasks=gpu_tasks,
            ane_tasks=ane_tasks,
            parallel=len(ane_tasks) > 0 and len(gpu_tasks) > 0
        )


# ── Layer 3: Task Decomposer ────────────────────────────────────────

class TaskDecomposer:
    """Breaks compound tasks into parallel-executable subtasks."""

    @staticmethod
    def decompose_batch(items: List[dict], simple_key: str = "type",
                        simple_values: set = None,
                        simple_prompt_template: str = "",
                        complex_prompt_template: str = "",
                        simple_max_tokens: int = 30,
                        complex_max_tokens: int = 80) -> List[Task]:
        """Split a batch of items into tasks by complexity."""
        if simple_values is None:
            simple_values = set()

        tasks = []
        for item in items:
            is_simple = item.get(simple_key, "") in simple_values
            if is_simple:
                tasks.append(Task(
                    prompt=simple_prompt_template.format(**item),
                    max_tokens=simple_max_tokens,
                    task_type="classify",
                    metadata=item,
                ))
            else:
                tasks.append(Task(
                    prompt=complex_prompt_template.format(**item),
                    max_tokens=complex_max_tokens,
                    task_type="analyze",
                    metadata=item,
                ))
        return tasks

    @staticmethod
    def decompose_extract_and_analyze(document: str,
                                      extract_prompt: str,
                                      analyze_prompt: str) -> List[Task]:
        """Split into extraction (simple) + analysis (complex)."""
        return [
            Task(prompt=extract_prompt, max_tokens=30,
                 task_type="extract", metadata={"role": "extract"}),
            Task(prompt=analyze_prompt, max_tokens=120,
                 task_type="analyze", metadata={"role": "analyze"}),
        ]


# ── Layer 4: Execution Engine ────────────────────────────────────────

class DualPathEngine:
    """Manages concurrent ANE + GPU inference."""

    def __init__(self, gpu_model_name="mlx-community/Qwen3-8B-4bit",
                 ane_model_name="Qwen/Qwen3-0.6B",
                 verbose=True):
        self.gpu_model_name = gpu_model_name
        self.ane_model_name = ane_model_name
        self.verbose = verbose
        self.scheduler = ParallelScheduler()
        self.decomposer = TaskDecomposer()

        self._gpu_model = None
        self._gpu_tok = None
        self._ane_model = None
        self._loaded = False

    def load(self):
        """Load both models."""
        import mlx.core as mx
        from mlx_lm import load as mlx_load

        if self.verbose:
            print("Loading GPU model...")
        t0 = time.time()
        self._gpu_model, self._gpu_tok = mlx_load(self.gpu_model_name)
        if self.verbose:
            print(f"  GPU ready in {time.time()-t0:.1f}s")

        if self.verbose:
            print("Loading ANE model...")
        from real_draft import RealDraftModel
        self._ane_model = RealDraftModel(model_name=self.ane_model_name)
        t0 = time.time()
        self._ane_model.load_and_compile(
            status_fn=lambda msg: print(f"  {msg}") if self.verbose else None,
            fused=True
        )
        if self.verbose:
            print(f"  ANE ready in {time.time()-t0:.1f}s")
        gc.collect()
        self._loaded = True

    def _gpu_inference(self, prompt: str, max_tokens: int) -> tuple:
        """Run inference on GPU via MLX."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        t0 = time.time()
        ids = self._gpu_tok.encode(prompt)
        cache = make_prompt_cache(self._gpu_model)
        x = mx.array([ids])
        logits = self._gpu_model(x, cache=cache)
        mx.eval(logits)

        tokens = []
        for _ in range(max_tokens):
            tok = mx.argmax(logits[0, -1, :]).item()
            tokens.append(tok)
            if tok == self._gpu_tok.eos_token_id:
                break
            x = mx.array([[tok]])
            logits = self._gpu_model(x, cache=cache)
            mx.eval(logits)

        text = self._gpu_tok.decode(tokens)
        elapsed_ms = (time.time() - t0) * 1000
        return text, elapsed_ms, len(tokens)

    def _ane_inference(self, prompt: str, max_tokens: int) -> tuple:
        """Run inference on ANE."""
        t0 = time.time()
        ids = self._ane_model.encode(prompt)
        self._ane_model.reset_cache()
        logits = None
        for i, tid in enumerate(ids):
            logits = self._ane_model.forward_token(tid, i)
        pos = len(ids)

        tokens = []
        for _ in range(max_tokens):
            tok = int(np.argmax(logits))
            tokens.append(tok)
            logits = self._ane_model.forward_token(tok, pos)
            pos += 1

        text = self._ane_model.decode(tokens)
        elapsed_ms = (time.time() - t0) * 1000
        return text, elapsed_ms, len(tokens)

    def execute(self, tasks: List[Task], mode: str = "auto") -> List[TaskResult]:
        """
        Execute tasks with routing.

        mode:
          "auto"     — smart routing (ANE parallel + GPU)
          "gpu_only" — everything on GPU (baseline comparison)
          "parallel_gpu" — two GPU models, no ANE (comparison)
        """
        assert self._loaded, "Call .load() first"

        if mode == "gpu_only":
            return self._execute_gpu_only(tasks)
        elif mode == "auto":
            return self._execute_dual_path(tasks)
        else:
            return self._execute_gpu_only(tasks)

    def _execute_gpu_only(self, tasks: List[Task]) -> List[TaskResult]:
        """Baseline: GPU handles everything sequentially."""
        results = []
        for task in tasks:
            text, ms, n_tok = self._gpu_inference(task.prompt, task.max_tokens)
            results.append(TaskResult(
                text=text, elapsed_ms=ms, path=ComputePath.GPU,
                task=task, tokens_generated=n_tok
            ))
        return results

    def _execute_dual_path(self, tasks: List[Task]) -> List[TaskResult]:
        """Smart routing: ANE parallel + GPU."""
        plan = self.scheduler.schedule(tasks)

        if not plan.parallel:
            # No parallel opportunity — everything on GPU
            return self._execute_gpu_only(plan.gpu_tasks)

        # Run both paths concurrently
        gpu_results = []
        ane_results = []

        def gpu_worker():
            for task in plan.gpu_tasks:
                text, ms, n_tok = self._gpu_inference(task.prompt, task.max_tokens)
                gpu_results.append(TaskResult(
                    text=text, elapsed_ms=ms, path=ComputePath.GPU,
                    task=task, tokens_generated=n_tok
                ))

        def ane_worker():
            for task in plan.ane_tasks:
                text, ms, n_tok = self._ane_inference(task.prompt, task.max_tokens)
                ane_results.append(TaskResult(
                    text=text, elapsed_ms=ms, path=ComputePath.ANE,
                    task=task, tokens_generated=n_tok
                ))

        t_gpu = threading.Thread(target=gpu_worker)
        t_ane = threading.Thread(target=ane_worker)
        t_gpu.start()
        t_ane.start()
        t_gpu.join()
        t_ane.join()

        # Merge results in original task order
        all_results = gpu_results + ane_results
        return all_results

    def execute_compound(self, tasks: List[Task]) -> Dict[str, List[TaskResult]]:
        """
        Execute with task decomposition.
        Returns dict with 'gpu_only' and 'dual_path' results for comparison.
        """
        # GPU-only baseline
        t0 = time.time()
        gpu_results = self.execute(tasks, mode="gpu_only")
        gpu_wall_ms = (time.time() - t0) * 1000

        # Dual-path
        t0 = time.time()
        dual_results = self.execute(tasks, mode="auto")
        dual_wall_ms = (time.time() - t0) * 1000

        return {
            "gpu_only": {"results": gpu_results, "wall_ms": gpu_wall_ms},
            "dual_path": {"results": dual_results, "wall_ms": dual_wall_ms},
        }
