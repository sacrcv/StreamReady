"""
Accuracy-based evaluation task, used for OVO-Bench.

Reports accuracy -- the metric the StreamReady paper uses outside ProReady-QA --
broken down per task, and additionally reports ARS whenever the annotation
carries answer evidence windows.

How answers are scored:

* **Multiple choice** (OVO-Bench backward and real-time tasks) by option letter.
* **Counting** (OVO-Bench forward REC) numerically, so "3 times" matches "3".
* **Yes/No** (OVO-Bench forward SSR and CRR) as a binary, accepting a bare
  ``Y``/``N``. Both match the official OVO-Bench scorer.
* **Open-ended**, if present, by normalised text match. Where a source's official
  protocol is GPT judging, the number here is a deterministic lower bound and is
  labelled as such.
* **ARS** is added only when the annotation has evidence windows. OVO-Bench does
  not carry them, so accuracy is the meaningful number there -- which is what the
  paper reports. ProReady-QA is scored by the ``readiness_vqa`` task instead.

OVO-Bench aggregates differently from a plain mean: per-task accuracy is averaged
within each of its three modes (backward / real-time / forward) and those three
mode averages are then averaged. That is implemented here so the reported number
is comparable to the official one.
"""

import json
import logging
import os
from collections import defaultdict

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.metrics.answer_matching import score_answer
from lavis.metrics.ars import compute_ars, format_ars_report
from lavis.tasks.base_task import BaseTask

# OVO-Bench's three modes, used for its mode-averaged aggregate.
OVO_MODES = {
    "backward": ("EPM", "ASI", "HLD"),
    "realtime": ("STU", "OJR", "ATR", "ACR", "OCR", "FPD"),
    "forward": ("REC", "SSR", "CRR"),
}

# Task names whose official metric is GPT-judged, so a text match understates
# them. None of OVO-Bench's tasks are, but an added open-ended source may be.
GPT_JUDGED_TASKS = {"summary", "sub_scene", "sub_plot"}


@registry.register_task("video_qa")
class VideoQATask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True,
                 gamma_early=6.0, gamma_late=1.0, prompt=""):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric
        self.gamma_early = gamma_early
        self.gamma_late = gamma_late
        self.prompt = prompt

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            num_beams=run_cfg.get("num_beams", 1),
            # Multiple choice needs only a letter; the default is deliberately
            # short so a model cannot bury the answer in prose.
            max_len=run_cfg.get("max_len", 16),
            min_len=run_cfg.get("min_len", 1),
            evaluate=run_cfg.get("evaluate", False),
            report_metric=run_cfg.get("report_metric", True),
            gamma_early=run_cfg.get("gamma_early", 6.0),
            gamma_late=run_cfg.get("gamma_late", 1.0),
            prompt=run_cfg.get("prompt", ""),
        )

    def valid_step(self, model, samples):
        answers, answer_times = model.predict_answers(
            samples, num_beams=self.num_beams,
            max_len=self.max_len, min_len=self.min_len,
        )

        results = []
        for i in range(len(answers)):
            options = samples["options"][i] if "options" in samples else []
            gold_text = samples["answer_text"][i] if "answer_text" in samples \
                else samples["text_output"][i]
            gold_letter = samples["text_output"][i] if options else ""
            task = samples["task"][i] if "task" in samples else "unknown"

            rec = {
                "qa_id": samples["qa_id"][i],
                "video_id": samples["video_id"][i],
                "benchmark": samples["benchmark"][i] if "benchmark" in samples else "",
                "task": task,
                "question": samples["text_input"][i],
                "prediction": answers[i],
                "ground_truth": gold_text,
                "gold_letter": gold_letter,
                "num_options": len(options),
                "correct": bool(score_answer(answers[i], gold_text, gold_letter,
                                             options, task)),
            }

            # Carry timing only when the benchmark annotates it; ARS is skipped
            # otherwise rather than invented from a zero window.
            ev_s = float(samples["evidence_start"][i]) if "evidence_start" in samples else 0.0
            ev_e = float(samples["evidence_end"][i]) if "evidence_end" in samples else 0.0
            if ev_e > ev_s:
                t_a = answer_times[i]
                if t_a is None or t_a != t_a:
                    t_a = ev_s
                rec["answer_time"] = float(t_a)
                rec["evidence_start"] = ev_s
                rec["evidence_end"] = ev_e
                rec["turn_group"] = samples["dialogue_id"][i]
            results.append(rec)
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_epoch{epoch}_videoqa",
            remove_duplicate="qa_id",
        )
        if self.report_metric:
            return self._report_metrics(result_file, split_name)
        return {"agg_metrics": 0.0}

    @main_process
    def _report_metrics(self, result_file, split_name):
        with open(result_file, "r") as f:
            records = json.load(f)
        if not records:
            return {"agg_metrics": 0.0}

        benchmark = next((r.get("benchmark") for r in records if r.get("benchmark")), "")

        per_task = defaultdict(list)
        for r in records:
            per_task[r.get("task", "unknown")].append(float(bool(r["correct"])))

        overall = sum(float(bool(r["correct"])) for r in records) / len(records)
        task_acc = {t: sum(v) / len(v) for t, v in sorted(per_task.items())}

        # OVO-Bench: average within each mode, then across the three modes.
        aggregate = overall
        aggregate_name = "accuracy"
        if benchmark == "ovo_bench":
            mode_means = []
            for mode, tasks in OVO_MODES.items():
                present = [task_acc[t] for t in tasks if t in task_acc]
                if present:
                    mode_means.append(sum(present) / len(present))
            if mode_means:
                aggregate = sum(mode_means) / len(mode_means)
                aggregate_name = "mode-averaged accuracy"

        logging.info("%s evaluation (%s)", benchmark or "video_qa", split_name)
        logging.info("  questions          %d", len(records))
        logging.info("  accuracy           %.4f", overall)
        if aggregate_name != "accuracy":
            logging.info("  %-18s %.4f", aggregate_name, aggregate)
        logging.info("  per task:")
        for t, a in task_acc.items():
            logging.info("    %-28s n %-6d acc %.4f", t, len(per_task[t]), a)

        if any(t in GPT_JUDGED_TASKS for t in task_acc):
            logging.info("  note: these tasks are officially GPT-judged; the "
                         "text-match accuracy above is a lower bound")

        metrics = {
            "agg_metrics": aggregate * 100.0,
            "accuracy": overall * 100.0,
            **{f"acc_{t}": a * 100.0 for t, a in task_acc.items()},
        }

        # ARS where the benchmark annotates evidence windows.
        timed = [r for r in records if "evidence_end" in r]
        if timed:
            summary = compute_ars(timed, gamma_e=self.gamma_early,
                                  gamma_l=self.gamma_late)
            logging.info("  timing (%d of %d questions have evidence windows):",
                         len(timed), len(records))
            for line in format_ars_report(summary).splitlines():
                logging.info("    %s", line)
            metrics["ars"] = summary["ars"]
            if "effective_accuracy" in summary:
                metrics["effective_accuracy"] = summary["effective_accuracy"] * 100.0

        log_path = os.path.join(registry.get_path("output_dir"), "evaluate.txt")
        with open(log_path, "a") as f:
            f.write(json.dumps({"split": split_name, "benchmark": benchmark,
                                **{k: v for k, v in metrics.items()}}) + "\n")
        return metrics
