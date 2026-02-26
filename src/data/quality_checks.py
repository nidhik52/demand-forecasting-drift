"""
Data Quality Checks — Great Expectations Integration
=====================================================
Project : Drift-Aware Continuous Learning Framework
File    : src/data/quality_checks.py

FW5 IMPLEMENTATION — Data Quality Gates
-----------------------------------------
Validates incoming demand data before it enters the forecasting pipeline.
Prevents degraded retrains caused by upstream data issues.

Uses Great Expectations if installed; falls back to pandas-based checks
if not, so the module works standalone without the full GE stack.

CHECKS PERFORMED
----------------
1. schema_check        — Required columns (ds, y, category) present
2. no_null_check       — No nulls in ds, y, or category
3. non_negative_demand — y >= 0 (demand cannot be negative)
4. date_format_check   — ds parseable as datetime
5. category_check      — Only known categories present
6. demand_range_check  — y within plausible bounds (0 to 500k)
7. date_continuity     — No gaps larger than 7 days per category
8. duplicate_check     — No (ds, category) duplicates

USAGE
-----
    from src.data.quality_checks import DemandDataValidator
    import pandas as pd

    df = pd.read_csv('data/processed/final_demand_series.csv')
    validator = DemandDataValidator(df)
    results   = validator.run_all_checks()

    # Print summary
    validator.print_report(results)

    # Raise on failure
    validator.raise_if_failed(results)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

EXPECTED_CATEGORIES = {
    'Electronics & Tech',
    'Entertainment & Office',
    'Fashion & Accessories',
    'Health & Personal Care',
    'Home & Lifestyle',
    'Sports & Outdoors',
}

DEMAND_MIN  =       0.0   # Rs.
DEMAND_MAX  = 500_000.0   # Rs. per day per category (upper sanity bound)
MAX_GAP_DAYS  = 7         # max tolerated gap in time series


@dataclass
class CheckResult:
    check    : str
    passed   : bool
    message  : str
    n_issues : int = 0


# ─────────────────────────────────────────────────────────
# Core validator
# ─────────────────────────────────────────────────────────

class DemandDataValidator:
    """
    Validates a demand DataFrame (columns: ds, y, category).

    Parameters
    ----------
    df : pd.DataFrame  — the demand data to validate
    use_great_expectations : bool
        If True and GE is installed, also runs a GE checkpoint.
        If False or GE not installed, only pandas checks are run.
    """

    def __init__(self, df: pd.DataFrame, use_great_expectations: bool = True):
        self.df  = df.copy()
        self.use_ge = use_great_expectations
        self._ge_available = self._check_ge()

    @staticmethod
    def _check_ge() -> bool:
        try:
            import great_expectations  # type: ignore[import-untyped]  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Individual checks ────────────────────────────────

    def check_schema(self) -> CheckResult:
        required = {'ds', 'y', 'category'}
        missing  = required - set(self.df.columns)
        return CheckResult(
            check    = 'schema_check',
            passed   = len(missing) == 0,
            message  = f'Missing columns: {missing}' if missing else 'All required columns present.',
            n_issues = len(missing),
        )

    def check_no_nulls(self) -> CheckResult:
        cols   = [c for c in ['ds', 'y', 'category'] if c in self.df.columns]
        nulls  = self.df[cols].isnull().sum()
        total  = int(nulls.sum())
        return CheckResult(
            check    = 'no_null_check',
            passed   = total == 0,
            message  = f'Null counts: {nulls.to_dict()}' if total else 'No nulls found.',
            n_issues = total,
        )

    def check_non_negative_demand(self) -> CheckResult:
        if 'y' not in self.df.columns:
            return CheckResult('non_negative_demand', False, 'y column missing.', 1)
        neg = int((self.df['y'] < 0).sum())
        return CheckResult(
            check    = 'non_negative_demand',
            passed   = neg == 0,
            message  = f'{neg} rows with negative demand.' if neg else 'All demand values ≥ 0.',
            n_issues = neg,
        )

    def check_date_format(self) -> CheckResult:
        if 'ds' not in self.df.columns:
            return CheckResult('date_format_check', False, 'ds column missing.', 1)
        try:
            pd.to_datetime(self.df['ds'])
            return CheckResult('date_format_check', True, 'All dates parseable.')
        except Exception as e:
            return CheckResult('date_format_check', False, f'Date parse error: {e}', 1)

    def check_categories(self) -> CheckResult:
        if 'category' not in self.df.columns:
            return CheckResult('category_check', False, 'category column missing.', 1)
        found    = set(self.df['category'].unique())
        unknown  = found - EXPECTED_CATEGORIES
        return CheckResult(
            check    = 'category_check',
            passed   = len(unknown) == 0,
            message  = f'Unknown categories: {unknown}' if unknown else f'All {len(found)} categories recognised.',
            n_issues = len(unknown),
        )

    def check_demand_range(self) -> CheckResult:
        if 'y' not in self.df.columns:
            return CheckResult('demand_range_check', False, 'y column missing.', 1)
        out_of_range = int(((self.df['y'] < DEMAND_MIN) | (self.df['y'] > DEMAND_MAX)).sum())
        return CheckResult(
            check    = 'demand_range_check',
            passed   = out_of_range == 0,
            message  = f'{out_of_range} rows outside [{DEMAND_MIN:.0f}, {DEMAND_MAX:.0f}].'
                       if out_of_range else f'All demand within [{DEMAND_MIN:.0f}, {DEMAND_MAX:.0f}].',
            n_issues = out_of_range,
        )

    def check_date_continuity(self) -> CheckResult:
        if not {'ds', 'category'}.issubset(self.df.columns):
            return CheckResult('date_continuity', False, 'ds or category column missing.', 1)
        df = self.df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        max_gaps: list[str] = []
        for cat, cdf in df.groupby('category'):
            dates = cdf['ds'].sort_values()
            if len(dates) < 2:
                continue
            gaps = (dates.diff().dt.days.dropna())
            big_gap = int(gaps.max())
            if big_gap > MAX_GAP_DAYS:
                max_gaps.append(f'{cat}: {big_gap}d gap')
        return CheckResult(
            check    = 'date_continuity',
            passed   = len(max_gaps) == 0,
            message  = 'Gaps > 7 days: ' + '; '.join(max_gaps) if max_gaps else f'No gaps > {MAX_GAP_DAYS} days.',
            n_issues = len(max_gaps),
        )

    def check_duplicates(self) -> CheckResult:
        if not {'ds', 'category'}.issubset(self.df.columns):
            return CheckResult('duplicate_check', False, 'ds or category column missing.', 1)
        n_dup = int(self.df.duplicated(subset=['ds', 'category']).sum())
        return CheckResult(
            check    = 'duplicate_check',
            passed   = n_dup == 0,
            message  = f'{n_dup} duplicate (ds, category) rows.' if n_dup else 'No duplicates.',
            n_issues = n_dup,
        )

    # ── Great Expectations checkpoint ────────────────────

    def _run_ge_checks(self) -> Optional[CheckResult]:
        """
        Runs a Great Expectations validation suite if GE is installed.
        Returns a single summarised CheckResult.
        """
        if not self._ge_available:
            return None
        try:
            import great_expectations as ge  # type: ignore[import-untyped]
            ge_df = ge.from_pandas(self.df)

            results = [
                ge_df.expect_column_to_exist('ds'),
                ge_df.expect_column_to_exist('y'),
                ge_df.expect_column_to_exist('category'),
                ge_df.expect_column_values_to_not_be_null('ds'),
                ge_df.expect_column_values_to_not_be_null('y'),
                ge_df.expect_column_min_to_be_between('y', min_value=0),
                ge_df.expect_column_max_to_be_between('y', max_value=DEMAND_MAX),
                ge_df.expect_column_values_to_be_in_set('category', list(EXPECTED_CATEGORIES)),
            ]
            failed = [r for r in results if not r['success']]
            return CheckResult(
                check    = 'great_expectations_suite',
                passed   = len(failed) == 0,
                message  = f'GE: {len(results) - len(failed)}/{len(results)} expectations passed.'
                           + (f' Failed: {[r["expectation_config"]["expectation_type"] for r in failed]}' if failed else ''),
                n_issues = len(failed),
            )
        except Exception as e:
            return CheckResult('great_expectations_suite', False, f'GE error: {e}', 1)

    # ── Run all ──────────────────────────────────────────

    def run_all_checks(self) -> List[CheckResult]:
        """Run all checks and return list of CheckResults."""
        checks: List[CheckResult] = [
            self.check_schema(),
            self.check_no_nulls(),
            self.check_non_negative_demand(),
            self.check_date_format(),
            self.check_categories(),
            self.check_demand_range(),
            self.check_date_continuity(),
            self.check_duplicates(),
        ]
        if self.use_ge and self._ge_available:
            ge_result = self._run_ge_checks()
            if ge_result:
                checks.append(ge_result)
        return checks

    def print_report(self, results: Optional[List[CheckResult]] = None) -> None:
        """Print a readable validation report."""
        if results is None:
            results = self.run_all_checks()
        passed = sum(1 for r in results if r.passed)
        print(f"\nData Quality Report — {passed}/{len(results)} checks passed")
        print('=' * 60)
        for r in results:
            icon = '✅' if r.passed else '❌'
            print(f"  {icon}  {r.check:<30}  {r.message}")
        print('=' * 60)
        if passed < len(results):
            print(f"  ⚠️  {len(results) - passed} check(s) FAILED — review before training.\n")
        else:
            print("  All checks passed — data is ready for training.\n")

    def raise_if_failed(self, results: Optional[List[CheckResult]] = None) -> None:
        """Raise ValueError if any check failed."""
        if results is None:
            results = self.run_all_checks()
        failed = [r for r in results if not r.passed]
        if failed:
            summary = "; ".join(f"{r.check}: {r.message}" for r in failed)
            raise ValueError(f"Data quality failed ({len(failed)} checks): {summary}")


# ─────────────────────────────────────────────────────────
# Standalone runner (used by Airflow and CLI)
# ─────────────────────────────────────────────────────────

def validate_demand_data(csv_path: str, raise_on_failure: bool = True) -> List[CheckResult]:
    """
    Convenience function — load CSV, run all checks, print report.

    Parameters
    ----------
    csv_path : str — path to final_demand_series.csv
    raise_on_failure : bool — raise ValueError if any check fails

    Returns
    -------
    List[CheckResult]
    """
    df        = pd.read_csv(csv_path)
    validator = DemandDataValidator(df)
    results   = validator.run_all_checks()
    validator.print_report(results)
    if raise_on_failure:
        validator.raise_if_failed(results)
    return results


if __name__ == '__main__':
    import sys, os
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'processed', 'final_demand_series.csv'
    )
    validate_demand_data(csv_path, raise_on_failure=False)
