"""
Readiness-aware VQA task -- reports Accuracy, ARS and effective accuracy.

This is the task that makes StreamReady's evaluation differ from ordinary video
QA: the model is asked not only to answer but to declare *when* it is ready, and
both are scored. Each validation step records the generated answer together with
the timestep the readiness head fired, then :meth:`after_evaluation` scores the
whole split with :mod:`lavis.metrics.ars`.

Accuracy here is exact-match after light normalisation, following the convention
used for open-ended video QA in this codebase (the model generates text rather
than selecting from a closed label set). Counting answers (REC) are compared
numerically so that "3" and "3 times" agree.
"""

import json
import logging
import os
import re
import string

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.metrics.ars import compute_ars, format_ars_report
from lavis.tasks.base_task import BaseTask

_ARTICLES = {"a", "an", "the"}
_PUNC = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    text = str(text).lower().strip().translate(_PUNC)
    words = [w for w in text.split() if w not in _ARTICLES]
    return " ".join(words)


def _first_number(text: str):
    m = re.search(r"-?\d+(?:\.\d+)?", str(text))
    return float(m.group()) if m else None


def answers_match(pred: str, gold: str, task: str = "") -> bool:
    """Exact match after normalisation, with a numeric path for counting tasks."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    if task == "REC" or (_first_number(g) is not None and len(g.split()) <= 3):
        pn, gn = _first_number(p), _first_number(g)
        if pn is not None and gn is not None:
            return abs(pn - gn) < 1e-6
    # Gold is a short phrase fully contained in a longer generated answer.
    if g and len(g.split()) <= 4 and g in p:
        return True
    return False


@registry.register_task("readiness_vqa")
class ReadinessVQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        report_metric=True,
        gamma_early=6.0,
        gamma_late=1.0,
        prompt="",
    ):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric
        # Paper Sec. 5: gamma_e = 6, gamma_l = 1.
        self.gamma_early = gamma_early
        self.gamma_late = gamma_late
        self.prompt = prompt

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            num_beams=run_cfg.get("num_beams", 3),
            max_len=run_cfg.get("max_len", 30),
            min_len=run_cfg.get("min_len", 1),
            evaluate=run_cfg.get("evaluate", False),
            report_metric=run_cfg.get("report_metric", True),
            gamma_early=run_cfg.get("gamma_early", 6.0),
            gamma_late=run_cfg.get("gamma_late", 1.0),
            prompt=run_cfg.get("prompt", ""),
        )

    def valid_step(self, model, samples):
        """One record per *turn*, from dialogue-level batches.

        ProReady-QA items are whole dialogues answered in a single streaming
        pass, so a batch of D dialogues yields sum(turns) records rather than D.
        """
        answers, answer_times = model.predict_answers(
            samples,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
        )

        results = []
        for d in range(len(answers)):
            for i in range(len(answers[d])):
                gold = samples["text_output"][d][i]
                task = samples["task"][d][i]
                # A readiness trigger before the question was even asked is not a
                # legal response; clamp so ARS scores it as maximally early
                # rather than as a negative offset.
                t_answer = answer_times[d][i]
                q_ts = float(samples["question_timestamp"][d][i])
                if t_answer is None or t_answer != t_answer:   # None or NaN
                    t_answer = q_ts
                t_answer = max(float(t_answer), q_ts)

                results.append({
                    "qa_id": samples["qa_id"][d][i],
                    "video_id": samples["video_id"][d],
                    "turn_group": samples["dialogue_id"][d],
                    "turn_index": int(samples["turn_index"][d][i]),
                    "task": task,
                    "question": samples["text_input"][d][i],
                    "prediction": answers[d][i],
                    "ground_truth": gold,
                    "correct": bool(answers_match(answers[d][i], gold, task)),
                    "answer_time": t_answer,
                    "question_timestamp": q_ts,
                    "evidence_start": float(samples["evidence_start"][d][i]),
                    "evidence_end": float(samples["evidence_end"][d][i]),
                })
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_epoch{epoch}_readiness",
            remove_duplicate="qa_id",
        )
        if self.report_metric:
            return self._report_metrics(result_file, split_name)
        return {"agg_metrics": 0.0}

    @main_process
    def _report_metrics(self, result_file, split_name):
        with open(result_file, "r") as f:
            records = json.load(f)

        summary = compute_ars(
            records, gamma_e=self.gamma_early, gamma_l=self.gamma_late
        )

        logging.info("ProReady-QA readiness evaluation (%s)", split_name)
        for line in format_ars_report(summary).splitlines():
            logging.info("  %s", line)

        log_path = os.path.join(registry.get_path("output_dir"), "evaluate.txt")
        with open(log_path, "a") as f:
            f.write(json.dumps({"split": split_name, **{
                k: v for k, v in summary.items() if k != "per_task"
            }}) + "\n")

        # Effective accuracy is the headline number of the paper (Eq. 9), so it is
        # what checkpoint selection tracks.
        return {
            "agg_metrics": summary.get("effective_accuracy", summary["ars"]) * 100.0,
            "accuracy": summary.get("accuracy", 0.0) * 100.0,
            "ars": summary["ars"],
            "effective_accuracy": summary.get("effective_accuracy", 0.0) * 100.0,
            **{f"ars_{k}": v["ars"] for k, v in summary.get("per_task", {}).items()},
        }
