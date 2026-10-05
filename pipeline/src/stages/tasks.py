"""The task graph (protocol Section 9.2, D13, D15, D17, D18, D20).

- Tuning: one study per global model x frequency x target (9 x 2 x 5 = 90).
- Evaluation, global models: cohort scope, one task per feature sample x frequency x
  strategy x model (6 x 2 x 3 x 9 = 324); tercile scope, one per populated bucket of each
  feature and frequency (at most 972; 918 when nonlinearity has an empty Medium bucket in
  both frequencies); seed check, evolving seasonality x frequency x strategy x the seven
  stochastic models x the extra seeds (84).
- Evaluation, statistical models: pool-independent (D18), so computed once per series of
  the union of a frequency's feature samples, per model and strategy, in chunks of series;
  they serve the 108 statistical cohort tasks of the grid.
Every task covers all windows. Shards are family-frequency pairs (statistical-monthly,
ml-quarterly, ...); priority 0 = cohort, 1 = tercile, 2 = seed check (Section 9.4).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.data.schemas import FEATURE_NAMES
from src.forecast.registry import FAMILY, GLOBAL_MODELS, STATISTICAL_MODELS, STRATEGIES, TARGETS

FREQUENCIES = ["monthly", "quarterly"]
STOCHASTIC_MODELS = ["RandomForest", "XGBoost", "NLinear", "NHITS", "LSTM", "TFT", "PatchTST"]  # D17
SEED_CHECK_FEATURE = "feature_evolving_seasonality"                                              # D17, A15
BUCKET_SCOPE = {"Low": "low", "Medium": "medium", "High": "high"}
STATISTICAL_CHUNK = 250


@dataclass(frozen=True)
class Task:
    kind: str                  # "global" | "statistical" | "tune"
    frequency: str
    model: str
    strategy: str | None       # None for tuning
    target: str | None         # tuning only
    feature_name: str | None   # None for statistical and tuning
    scope: str | None          # cohort | low | medium | high (evaluation)
    seed: int | None
    priority: int
    chunk: int | None = None   # statistical only: index of the chunk of series

    @property
    def family(self) -> str:
        return FAMILY[self.model]

    @property
    def shard(self) -> str:
        return f"{self.family}-{self.frequency}"

    @property
    def task_id(self) -> str:
        if self.kind == "tune":
            return f"tune|{self.model}|{self.frequency}|{self.target}"
        if self.kind == "statistical":
            return f"stat|{self.model}|{self.frequency}|{self.strategy}|chunk{self.chunk:03d}"
        return f"eval|{self.feature_name}|{self.frequency}|{self.strategy}|{self.model}|{self.scope}|seed{self.seed}"

    def record(self) -> dict:
        return {**asdict(self), "family": self.family, "shard": self.shard, "task_id": self.task_id}


def tuning_tasks() -> list[Task]:
    return [Task("tune", f, model, None, target, None, None, None, 0)
            for model in GLOBAL_MODELS for f in FREQUENCIES for target in TARGETS]


def evaluation_tasks(bucket_summary: pd.DataFrame, samples: pd.DataFrame, main_seed: int,
                     extra_seeds: list[int]) -> list[Task]:
    """All evaluation tasks from the prepare bundle's bucket summary and samples."""
    tasks: list[Task] = []
    for f in FREQUENCIES:
        for feature in FEATURE_NAMES:
            summary = bucket_summary[(bucket_summary["frequency"] == f) & (bucket_summary["feature_name"] == feature)]
            if len(summary) != 1:
                raise ValueError(f"bucket summary must hold one row for {feature} {f}")
            populated = summary["populated_buckets"].iloc[0].split(",")
            for strategy in STRATEGIES:
                for model in GLOBAL_MODELS:
                    tasks.append(Task("global", f, model, strategy, None, feature, "cohort", main_seed, 0))
                    for bucket in populated:
                        tasks.append(Task("global", f, model, strategy, None, feature, BUCKET_SCOPE[bucket], main_seed, 1))
        union = sorted(samples.loc[samples["frequency"] == f, "unique_id"].unique())
        n_chunks = -(-len(union) // STATISTICAL_CHUNK)
        for model in STATISTICAL_MODELS:
            for strategy in STRATEGIES:
                for chunk in range(n_chunks):
                    tasks.append(Task("statistical", f, model, strategy, None, None, "cohort", None, 0, chunk))
        for strategy in STRATEGIES:
            for model in STOCHASTIC_MODELS:
                for seed in extra_seeds:
                    tasks.append(Task("global", f, model, strategy, None, SEED_CHECK_FEATURE, "cohort", seed, 2))
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicated task ids")
    return sorted(tasks, key=lambda t: (t.priority, t.task_id))


def statistical_chunk_members(samples: pd.DataFrame, frequency: str, chunk: int) -> list[str]:
    union = sorted(samples.loc[samples["frequency"] == frequency, "unique_id"].unique())
    members = union[chunk * STATISTICAL_CHUNK: (chunk + 1) * STATISTICAL_CHUNK]
    if not members:
        raise ValueError(f"no series in statistical chunk {chunk} of {frequency}")
    return members


def global_task_members(task: Task, samples: pd.DataFrame, buckets: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """The training pool of a global task and every member's bucket for the task's feature."""
    sample = buckets[(buckets["frequency"] == task.frequency) & (buckets["feature_name"] == task.feature_name)]
    bucket_of = dict(zip(sample["unique_id"], sample["feature_tercile_bucket"]))
    if task.scope == "cohort":
        members = sorted(bucket_of)
    else:
        wanted = {v: k for k, v in BUCKET_SCOPE.items()}[task.scope]
        members = sorted(uid for uid, b in bucket_of.items() if b == wanted)
    expected = set(samples[(samples["frequency"] == task.frequency) & (samples["feature_name"] == task.feature_name)]["unique_id"])
    if set(bucket_of) != expected:
        raise ValueError(f"{task.task_id}: buckets and sample disagree")
    if not members:
        raise ValueError(f"{task.task_id}: empty training pool")
    return members, {uid: bucket_of[uid] for uid in members}


def grid_counts(tasks: list[Task]) -> dict:
    """Counts in the protocol's terms (Section 9.2)."""
    g = [t for t in tasks if t.kind == "global"]
    stat_chunks = [t for t in tasks if t.kind == "statistical"]
    return {
        "cohort_global": sum(t.priority == 0 for t in g),
        "cohort_statistical_views": len({(f, m, s) for t in stat_chunks for f, m, s in [(t.frequency, t.model, t.strategy)]}) * len(FEATURE_NAMES),
        "tercile": sum(t.priority == 1 for t in g),
        "seed_check": sum(t.priority == 2 for t in g),
        "statistical_chunks": len(stat_chunks),
    }
