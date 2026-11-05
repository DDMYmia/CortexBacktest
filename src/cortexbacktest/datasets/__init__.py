"""Dataset assembly layer.

This package connects low-level data inputs with research and event workflows.
"""


from .availability import (
    FactorAvailability as FactorAvailability,
    align_factor_availability as align_factor_availability,
)
from .diagnostics import (
    BarDiagnostics as BarDiagnostics,
    diagnose_bars as diagnose_bars,
)
from .duck import (
    DEFAULT_EXCHANGE_TZ as DEFAULT_EXCHANGE_TZ,
    ResearchStore as ResearchStore,
)
from .loaders import (
    BacktestDataset as BacktestDataset,
    load_backtest_dataset as load_backtest_dataset,
    load_backtest_dataset_dir as load_backtest_dataset_dir,
    load_target_weights as load_target_weights,
    publish_backtest_snapshot as publish_backtest_snapshot,
    save_backtest_dataset as save_backtest_dataset,
)
from .provenance import (
    DatasetProvenance as DatasetProvenance,
    sha256_file as sha256_file,
    verify_file_hashes as verify_file_hashes,
)
from .schema import (
    BarResolution as BarResolution,
    DataQualityReport as DataQualityReport,
    bars_to_events as bars_to_events,
    normalize_bars as normalize_bars,
    validate_bars as validate_bars,
)

# Keep the public export list statically visible to type checkers.
__all__ = [
    "FactorAvailability", "align_factor_availability", "BarDiagnostics", "diagnose_bars",
    "DEFAULT_EXCHANGE_TZ", "ResearchStore", "BacktestDataset", "load_backtest_dataset",
    "load_backtest_dataset_dir", "load_target_weights", "publish_backtest_snapshot", "save_backtest_dataset",
    "DatasetProvenance", "sha256_file", "verify_file_hashes", "BarResolution", "DataQualityReport",
    "bars_to_events", "normalize_bars", "validate_bars",
]
