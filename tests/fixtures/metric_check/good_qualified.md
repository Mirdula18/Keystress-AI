# Good fixture — metrics carrying their source

Every metric here states where it came from, so the checker must pass this file.

The classifier reaches 90% accuracy on synthetic data.

| Metric | Score | Data source |
|---|---|---|
| Accuracy | ~90% | synthetic (hand-authored labels) |

Measured on synthetic data only:

Precision was 0.9013 and recall was 0.89.

The UI shows a confidence of 85%, which is uncalibrated.

A reviewed exception, justified inline:

Accuracy target for the release gate is 80. metrics-ok: threshold config, not a claim

CSS-style values must not trip the checker:

    .result-confidence { font-size: 0.9rem; margin-top: 2px; }
