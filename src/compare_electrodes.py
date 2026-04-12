"""End-to-end comparison workflow for Ag/AgCl and PEDOT electrodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config
from .data_loader import load_feature_table
from .preprocessing.pipeline import run_pipeline
from .recordings import (
    build_recording_manifest,
    derive_stable_channel_subset,
    filter_summary_for_channels,
    summarize_recordings,
)
from .train import train_model


def _load_split_payload(path: Path) -> dict[str, list[str]]:
    with path.open() as fh:
        return json.load(fh)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def _recording_key_set(summary_df: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(row.word), str(row.session_token))
        for row in summary_df.itertuples()
        if bool(row.is_usable)
    }


def _recording_ids_for_keys(summary_df: pd.DataFrame, keys: set[tuple[str, str]]) -> set[str]:
    subset = summary_df[
        summary_df.apply(lambda row: (str(row["word"]), str(row["session_token"])) in keys, axis=1)
    ].copy()
    return set(subset["recording_id"].tolist())


def _summary_for_keys(summary_df: pd.DataFrame, keys: set[tuple[str, str]]) -> pd.DataFrame:
    return summary_df[
        summary_df.apply(lambda row: (str(row["word"]), str(row["session_token"])) in keys, axis=1)
    ].reset_index(drop=True)


def _matched_seed(split_name: str, word: str, session_token: str) -> int:
    return config.MATCHED_RANDOM_SEED + sum(ord(ch) for ch in f"{split_name}|{word}|{session_token}")


def build_matched_trial_subsets(
    feature_tables: dict[str, pd.DataFrame],
    split_payloads: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, list[str]]], pd.DataFrame]:
    """Downsample each electrode to the same trial budget per split/word/session."""
    selected_ids = {electrode: {"train": [], "val": [], "test": []} for electrode in feature_tables}
    summary_rows: list[dict[str, object]] = []

    for split_name in ("train", "val", "test"):
        split_tables = {
            electrode: table[table["trial_id"].isin(split_payloads[electrode].get(split_name, []))].copy()
            for electrode, table in feature_tables.items()
        }
        common_keys = set.intersection(
            *[
                {
                    (str(row.word), str(row.session_token))
                    for row in split_table[["word", "session_token"]].drop_duplicates().itertuples(index=False)
                }
                for split_table in split_tables.values()
            ]
        )

        for word, session_token in sorted(common_keys):
            per_electrode = {}
            for electrode, split_table in split_tables.items():
                mask = (split_table["word"] == word) & (split_table["session_token"] == session_token)
                per_electrode[electrode] = split_table.loc[mask].sort_values("trial_id").copy()

            budget = min(len(table) for table in per_electrode.values())
            if budget <= 0:
                continue

            summary_rows.append(
                {
                    "split": split_name,
                    "word": word,
                    "session_token": session_token,
                    "trial_budget": int(budget),
                    **{f"{electrode}_available_trials": int(len(table)) for electrode, table in per_electrode.items()},
                }
            )

            for electrode, table in per_electrode.items():
                sampled = table.sample(n=budget, random_state=_matched_seed(split_name, word, session_token))
                selected_ids[electrode][split_name].extend(sorted(sampled["trial_id"].tolist()))

    subset_tables: dict[str, pd.DataFrame] = {}
    subset_payloads: dict[str, dict[str, list[str]]] = {}
    for electrode, feature_df in feature_tables.items():
        chosen = {
            split_name: sorted(ids)
            for split_name, ids in selected_ids[electrode].items()
        }
        chosen_ids = set(sum(chosen.values(), []))
        subset_tables[electrode] = feature_df[feature_df["trial_id"].isin(chosen_ids)].copy().reset_index(drop=True)
        subset_payloads[electrode] = chosen
    return subset_tables, subset_payloads, pd.DataFrame(summary_rows)


def _predict_split(bundle: dict[str, object], split_df: pd.DataFrame) -> pd.Series:
    predictions = bundle["model"].predict(split_df[bundle["feature_columns"]])
    labels = [bundle["label_classes"][int(index)] for index in predictions]
    return pd.Series(labels, index=split_df.index)


def _per_session_metrics(
    bundle: dict[str, object],
    features_df: pd.DataFrame,
    split_payload: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, trial_ids in split_payload.items():
        split_df = features_df[features_df["trial_id"].isin(trial_ids)].copy()
        if split_df.empty:
            continue
        pred = _predict_split(bundle, split_df)
        split_df = split_df.assign(predicted_word=pred.values)
        for session_token, session_df in split_df.groupby("session_token"):
            accuracy = float((session_df["word"] == session_df["predicted_word"]).mean())
            rows.append(
                {
                    "split": split_name,
                    "session_token": session_token,
                    "n_examples": int(len(session_df)),
                    "accuracy": accuracy,
                }
            )
    return pd.DataFrame(rows)


def _write_report_artifacts(
    namespace: str,
    bundle: dict[str, object],
    features_df: pd.DataFrame,
    split_payload: dict[str, list[str]],
) -> None:
    report_dir = config.get_report_dir(namespace)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = bundle["metrics"]
    _save_json(report_dir / "metrics.json", metrics_payload)

    for split_name, metrics in metrics_payload.items():
        pd.DataFrame(metrics["classification_report"]).T.to_csv(report_dir / f"{split_name}_classification_report.csv")
        pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]).to_csv(
            report_dir / f"{split_name}_confusion_matrix.csv"
        )

    per_session_df = _per_session_metrics(bundle, features_df, split_payload)
    per_session_df.to_csv(report_dir / "per_session_metrics.csv", index=False)


def _qc_caveats(summary_df: pd.DataFrame) -> list[str]:
    total = int(len(summary_df))
    usable = int(summary_df["is_usable"].astype(bool).sum())
    caveats = [f"{usable}/{total} recordings passed base QC."]
    suspicious = int((summary_df["suspicious_channel_count"] > 0).sum())
    if suspicious:
        caveats.append(f"{suspicious} recordings include suspicious channels.")
    mismatch = int(summary_df["has_length_mismatch"].astype(bool).sum())
    if mismatch:
        caveats.append(f"{mismatch} recordings have channel-length mismatches.")
    return caveats


def _write_comparison_summary(
    report_dir: Path,
    operational_results: dict[str, dict[str, object]],
    matched_results: dict[str, dict[str, object]],
    caveats: dict[str, list[str]],
    matched_trial_budget: pd.DataFrame,
    matched_channels: tuple[str, ...],
) -> None:
    def best_metric_line(result: dict[str, object]) -> str:
        metrics = result["bundle"]["metrics"]
        for split_name in ("test", "val"):
            split_metrics = metrics.get(split_name)
            if split_metrics:
                return (
                    f"{split_name} accuracy={split_metrics['accuracy']:.4f}, "
                    f"macro_f1={split_metrics['macro_f1']:.4f}"
                )
        return "no held-out split available"

    lines = [
        "# Electrode Comparison Summary",
        "",
        "## Operational",
    ]
    for electrode, result in operational_results.items():
        lines.append(f"- {electrode}: {best_metric_line(result)}, channels={','.join(result['channels'])}")

    lines.extend(["", "## Matched"])
    lines.append(f"- matched_channels={','.join(matched_channels)}")
    for electrode, result in matched_results.items():
        lines.append(f"- {electrode}: {best_metric_line(result)}")

    lines.extend(["", "## Caveats"])
    for electrode, notes in caveats.items():
        lines.append(f"- {electrode}: {' '.join(notes)}")

    if not matched_trial_budget.empty:
        lines.extend(["", "## Matched Trial Budget"])
        for row in matched_trial_budget.itertuples(index=False):
            lines.append(
                f"- {row.split} {row.word}/{row.session_token}: budget={row.trial_budget}, "
                f"agagcl={row.agagcl_available_trials}, pedot={row.pedot_available_trials}"
            )

    (report_dir / "comparison_summary.md").write_text("\n".join(lines))


def _run_mode(
    *,
    electrode: str,
    namespace: str,
    manifest_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    selected_recording_ids: set[str] | None,
    channels: tuple[str, ...],
) -> dict[str, object]:
    pipeline_summary = run_pipeline(
        electrode=electrode,
        namespace=namespace,
        manifest_df=manifest_df,
        allowed_channels=channels,
        selected_recording_ids=selected_recording_ids,
    )
    features_path = config.get_features_metadata_path(namespace)
    split_path = config.get_by_session_split_path(namespace)
    bundle = train_model(features_csv=features_path, split_path=split_path, save_path=config.get_model_path(namespace))
    features_df = load_feature_table(features_path)
    split_payload = _load_split_payload(split_path)
    _write_report_artifacts(namespace, bundle, features_df, split_payload)
    return {
        "pipeline_summary": pipeline_summary,
        "bundle": bundle,
        "features_df": features_df,
        "split_payload": split_payload,
        "summary_df": summary_df,
        "channels": channels,
        "namespace": namespace,
    }


def run_comparison() -> dict[str, object]:
    report_dir = config.get_report_dir("comparisons/agagcl_vs_pedot")
    report_dir.mkdir(parents=True, exist_ok=True)

    manifests = {electrode: build_recording_manifest(electrode) for electrode in ("agagcl", "pedot")}
    summaries = {electrode: summarize_recordings(manifest_df) for electrode, manifest_df in manifests.items()}
    caveats = {electrode: _qc_caveats(summary_df) for electrode, summary_df in summaries.items()}

    operational_channels = {
        electrode: derive_stable_channel_subset(summary_df)
        for electrode, summary_df in summaries.items()
    }
    operational_summaries = {
        electrode: filter_summary_for_channels(summary_df, operational_channels[electrode])
        for electrode, summary_df in summaries.items()
    }

    operational_results = {}
    for electrode in ("agagcl", "pedot"):
        operational_results[electrode] = _run_mode(
            electrode=electrode,
            namespace=f"{electrode}/operational",
            manifest_df=manifests[electrode],
            summary_df=summaries[electrode],
            selected_recording_ids=set(operational_summaries[electrode]["recording_id"].tolist()),
            channels=operational_channels[electrode],
        )

    common_keys = _recording_key_set(operational_summaries["agagcl"]) & _recording_key_set(operational_summaries["pedot"])
    pedot_common_summary = _summary_for_keys(summaries["pedot"], common_keys)
    ag_common_summary = _summary_for_keys(summaries["agagcl"], common_keys)
    pedot_common_channels = derive_stable_channel_subset(filter_summary_for_channels(pedot_common_summary))
    ag_common_channels = derive_stable_channel_subset(filter_summary_for_channels(ag_common_summary))
    matched_channels = tuple(sorted(set(pedot_common_channels) & set(ag_common_channels), key=int))

    matched_results = {}
    matched_keys = common_keys
    for electrode in ("agagcl", "pedot"):
        summary_for_mode = _summary_for_keys(summaries[electrode], matched_keys)
        filtered_for_mode = filter_summary_for_channels(summary_for_mode, matched_channels)
        matched_results[electrode] = _run_mode(
            electrode=electrode,
            namespace=f"{electrode}/matched",
            manifest_df=manifests[electrode],
            summary_df=summaries[electrode],
            selected_recording_ids=set(filtered_for_mode["recording_id"].tolist()),
            channels=matched_channels,
        )

    matched_feature_tables = {
        electrode: result["features_df"]
        for electrode, result in matched_results.items()
    }
    matched_split_payloads = {
        electrode: result["split_payload"]
        for electrode, result in matched_results.items()
    }
    subset_tables, subset_payloads, matched_trial_budget = build_matched_trial_subsets(
        matched_feature_tables,
        matched_split_payloads,
    )

    budgeted_results = {}
    for electrode in ("agagcl", "pedot"):
        namespace = f"{electrode}/matched_budget"
        config.get_metadata_dir(namespace).mkdir(parents=True, exist_ok=True)
        subset_tables[electrode].to_csv(config.get_features_metadata_path(namespace), index=False)
        _save_json(config.get_by_session_split_path(namespace), subset_payloads[electrode])
        _save_json(config.get_sampled_trial_ids_path(namespace), subset_payloads[electrode])
        bundle = train_model(
            save_path=config.get_model_path(namespace),
            features_df=subset_tables[electrode],
            split_payload=subset_payloads[electrode],
        )
        _write_report_artifacts(namespace, bundle, subset_tables[electrode], subset_payloads[electrode])
        budgeted_results[electrode] = {
            "bundle": bundle,
            "channels": matched_channels,
            "namespace": namespace,
        }

    matched_trial_budget.to_csv(report_dir / "matched_trial_budget.csv", index=False)
    _save_json(report_dir / "operational_channels.json", operational_channels)
    _save_json(report_dir / "matched_channels.json", {"matched_channels": matched_channels})
    _write_comparison_summary(report_dir, operational_results, budgeted_results, caveats, matched_trial_budget, matched_channels)

    return {
        "operational_results": operational_results,
        "matched_results": budgeted_results,
        "matched_trial_budget": matched_trial_budget,
        "matched_channels": matched_channels,
    }


def main() -> None:
    run_comparison()
    print(f"Wrote comparison artifacts to {config.get_report_dir('comparisons/agagcl_vs_pedot')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
