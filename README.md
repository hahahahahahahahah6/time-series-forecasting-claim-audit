# Documentation evidence auditor

A local, static audit tool comparing documentation language with source operations.
It uses the standard library, parses source without executing it, and performs no network access.

```sh
python3 -m time_series_forecasting_claim_audit audit .
time-series-forecasting-claim-audit audit . --json --fail-on medium
```

The root README and Markdown files beneath a docs directory are examined.
Fenced blocks are skipped; inline code counts. Source in examples, fixtures,
and docs is excluded. Tests are reported separately. Symlinks are skipped.

Use repeated `--ignore PREFIX` options to exclude relative path prefixes.
`--verbose` includes evidence paths. `--version` prints the package version.
The default threshold is high; medium also fails on test evidence alone.
Exit status is 0 for success, 2 for findings meeting the threshold, and 1 for usage or I/O errors.
JSON includes a verdict and findings with rule, severity, and message fields.

```text
documented-time-series-forecasting-claim-false
documented-time-series-forecasting-claim-test-only
documented-no-time-series-forecasting-claim-false
```

Install with `pip install .`. Run checks with `python3 -m unittest discover`.

## Scope

This tool checks one operation: a model call that produces forward
estimates from series data, paired with a construction call in the
same module. Fitting alone, backtesting alone, loading history
alone, charting alone, and error-metric evaluation alone are not
counted. Generic estimators with no series-model construction are
not counted either.

Neighbouring checkers cover other operations: the classification,
detection, and plotting checkers in the same series inspect their
own call patterns, never the paired construction-plus-estimate
pattern checked here.

## Honest limitations

- Static pattern matching, not semantic judgement: dead code counts
  as evidence, and an alias chain that the tracker cannot follow is
  missed.
- Exact phrases only: paraphrases of the checked claim are missed.
- A denial is only recognised when it uses one of the listed
  denial phrases.
- The estimate call must sit in the same module as the
  construction; a pair split across modules is not counted.
