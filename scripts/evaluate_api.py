#!/usr/bin/env python3
"""Evaluate MedVQA on VQA-RAD test set using API mode (OpenAI gpt-4o).

Computes:
  - Yes/No accuracy (with flexible normalization)
  - BLEU-1/4
  - ROUGE-L
  - BERTScore (if bert-score installed)

Usage:
    python scripts/evaluate_api.py

Requires:
    - OPENAI_API_KEY in .env or environment
    - VQA-RAD dataset downloaded (run scripts/download_vqa_rad.py first)
"""

import json
import re
import sys
import time
from pathlib import Path

# Force UTF-8 for stdout/stderr (handles emoji in Windows consoles)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Progress reporting ──────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Load VQA-RAD test set ───────────────────────────────────────────────────
DATA_DIR = Path("data/raw/vqa_rad")
ANNOTATION_FILE = DATA_DIR / "VQA_RAD_Dataset.json"
IMAGES_DIR = DATA_DIR / "images"

if not ANNOTATION_FILE.exists():
    print(f"[ERROR] Annotation file not found: {ANNOTATION_FILE}")
    print("[HINT]  Run: python scripts/download_vqa_rad.py")
    sys.exit(1)


def load_test_set() -> list[dict]:
    """Load VQA-RAD test split samples."""
    with open(ANNOTATION_FILE) as f:
        all_annotations = json.load(f)

    test_samples = [s for s in all_annotations if s.get("split", "").lower() == "test"]

    for s in test_samples:
        img_file = s.get("filename", "")
        candidate = IMAGES_DIR / img_file
        if candidate.exists():
            s["image_path"] = str(candidate)
        else:
            found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                alt = IMAGES_DIR / (Path(img_file).stem + ext)
                if alt.exists():
                    s["image_path"] = str(alt)
                    found = True
                    break
            if not found:
                s["image_path"] = str(candidate)

    return test_samples


# ── Initialize API client ───────────────────────────────────────────────────
def get_api_client():
    from src.inference.api_llm import APILLMClient, APILLMConfig

    config = APILLMConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.0,
        max_tokens=64,
    )

    is_valid, err_msg = config.validate()
    if not is_valid:
        print(f"[ERROR] {err_msg}")
        print("[HINT]  Set OPENAI_API_KEY in your .env file")
        sys.exit(1)

    return APILLMClient(config)


# ── Smarter normalization ───────────────────────────────────────────────────


def extract_yes_no(text: str) -> str | None:
    """Extract yes/no from GPT-4o's verbose answers.

    Looks for 'yes' or 'no' anywhere in the text, handling:
    - "Yes, there is..." -> yes
    - "No evidence of..." -> no
    - "I'm unable to determine..." -> None (abstain)
    - "The image shows no..." -> no
    """
    text_clean = text.strip().lower()

    # Check for explicit negation patterns first
    negation_patterns = [
        r"\bno\b",
        r"\bnot\b",
        r"\bcan't\b",
        r"\bcannot\b",
        r"\bunable\b",
        r"\bnegative\b",
        r"\babsent\b",
        r"\bwithout\b",
        r"\bno evidence\b",
    ]

    affirmation_patterns = [
        r"\byes\b",
        r"\bpresent\b",
        r"\bvisible\b",
        r"\bconsistent\b",
        r"\bapparent\b",
    ]

    # First check for explicit "I'm unable" / "cannot determine"
    cannot_determine = re.search(
        r"\b(unable|cannot|can\'t)\s+.*\b(determine|assess|evaluate|comment)\b", text_clean
    )
    if cannot_determine:
        return None  # Abstained

    # Check for explicit yes
    if re.search(r"^\s*yes", text_clean) or text_clean.startswith("yes"):
        return "yes"

    # Check for explicit no at start
    if re.search(r"^\s*no", text_clean) or text_clean.startswith("no"):
        return "no"

    # Check for "The answer is yes/no"
    answer_is = re.search(r"\banswer\s+is\s+(yes|no)\b", text_clean)
    if answer_is:
        return answer_is.group(1)

    # Check for "shows no evidence" / "there is no"
    negation = any(re.search(p, text_clean) for p in negation_patterns)
    affirmation = any(re.search(p, text_clean) for p in affirmation_patterns)

    if negation and not affirmation:
        return "no"
    if affirmation and not negation:
        return "yes"

    return None  # Can't determine


def normalize_answer(text: str) -> str:
    """Normalize an answer for comparison."""
    return text.strip().lower().rstrip(".!?")


# ── Metrics ─────────────────────────────────────────────────────────────────


def compute_metrics(
    predictions: list[str], references: list[str], questions: list[str], is_yesno: list[bool]
):
    """Compute all evaluation metrics with smart yes/no extraction."""
    metrics = {}

    n = len(predictions)
    if n == 0:
        return metrics

    # ── 1. Yes/No accuracy with flexible extraction ──────────────────────
    yesno_preds = []
    yesno_refs = []
    yesno_raw = []
    yesno_abstained = 0

    for pred, ref, yn in zip(predictions, references, is_yesno):
        if yn:
            extracted = extract_yes_no(pred)
            yesno_raw.append(pred)
            if extracted is not None:
                yesno_preds.append(extracted)
                yesno_refs.append(normalize_answer(ref))
            else:
                yesno_abstained += 1

    if yesno_preds:
        correct = sum(1 for p, r in zip(yesno_preds, yesno_refs) if p == r)
        total = len(yesno_preds)
        all_yn = len([y for y in is_yesno if y])
        metrics["yesno_accuracy"] = correct / total if total > 0 else 0.0
        metrics["yesno_correct"] = correct
        metrics["yesno_total"] = total
        metrics["yesno_abstained"] = yesno_abstained
        metrics["yesno_all_count"] = all_yn
        print(
            f"\n  Yes/No: {correct}/{total} = {metrics['yesno_accuracy'] * 100:.1f}% "
            f"(abstained: {yesno_abstained}/{all_yn})"
        )

    # ── 2. BLEU scores ──────────────────────────────────────────────────
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

        pred_tokens = [p.lower().split() for p in predictions]
        ref_tokens = [[r.lower().split()] for r in references]
        smoothing = SmoothingFunction().method1

        for n_gram in range(1, 5):
            weights = tuple(1.0 / n_gram if i < n_gram else 0.0 for i in range(4))
            try:
                score = corpus_bleu(
                    ref_tokens, pred_tokens, weights=weights, smoothing_function=smoothing
                )
            except Exception:
                score = 0.0
            metrics[f"bleu_{n_gram}"] = score
            print(f"  BLEU-{n_gram}: {score:.4f}")
    except ImportError:
        print("  [SKIP] nltk — skipping BLEU")

    # ── 3. ROUGE-L ──────────────────────────────────────────────────────
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        f1_scores = []
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            f1_scores.append(result["rougeL"].fmeasure)
        metrics["rouge_l_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        print(f"  ROUGE-L F1: {metrics['rouge_l_f1']:.4f}")
    except ImportError:
        print("  [SKIP] rouge-score — skipping ROUGE-L")

    # ── 4. BERTScore ────────────────────────────────────────────────────
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            predictions, references, model_type="distilbert-base-uncased", lang="en", verbose=False
        )
        metrics["bertscore_f1"] = F1.mean().item()
        metrics["bertscore_precision"] = P.mean().item()
        metrics["bertscore_recall"] = R.mean().item()
        print(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}")
    except ImportError:
        print("  [SKIP] bert-score — skipping BERTScore")

    # ── 5. Keyword/entity recall for open-ended ─────────────────────────
    # Checks if key medical terms from the reference appear in the prediction
    if not all(is_yesno):
        open_preds = [p for p, y in zip(predictions, is_yesno) if not y]
        open_refs = [r for r, y in zip(references, is_yesno) if not y]
        terms_found = 0
        terms_total = 0
        for pred, ref in zip(open_preds, open_refs):
            pred_lower = pred.lower()
            ref_words = set(ref.lower().split())
            for word in ref_words:
                if len(word) > 3:  # Only meaningful words
                    terms_total += 1
                    if word in pred_lower:
                        terms_found += 1
        metrics["keyword_recall"] = terms_found / terms_total if terms_total > 0 else 0.0
        print(
            f"  Keyword Recall (open-ended): {metrics['keyword_recall'] * 100:.1f}% ({terms_found}/{terms_total} terms)"
        )

    return metrics


# ── Main evaluation loop ────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  MedVQA — API Mode Evaluation (OpenAI gpt-4o)")
    print("=" * 60)

    # 1. Load test set
    print("\n[1/4] Loading VQA-RAD test set...")
    test_samples = load_test_set()
    print(f"  Loaded {len(test_samples)} test samples")

    # Count yes/no vs open
    yn_count = sum(
        1 for s in test_samples if s.get("question_type", "").lower() in ["yes/no", "binary"]
    )
    open_count = len(test_samples) - yn_count
    print(f"  Yes/No: {yn_count}  Open-ended: {open_count}")

    # 2. Initialize API client
    print("\n[2/4] Initializing OpenAI gpt-4o client...")
    client = get_api_client()
    print("  [OK] Client ready")

    # 3. Run inference
    print(f"\n[3/4] Running inference on {len(test_samples)} test samples...")
    print("  (This will take ~15-20 minutes, ~$1-2 in API costs)")
    print("  ----------------------------------------------")

    predictions = []
    references = []
    questions_list = []
    is_yesno_list = []
    errors = 0
    _start_time = time.time()

    iterator = enumerate(test_samples)
    if tqdm:
        iterator = tqdm(list(iterator), desc="Evaluating", unit="sample")

    for idx, sample in iterator:
        question = sample["question"]
        answer = sample["answer"]
        image_path = sample.get("image_path", "")
        qtype = sample.get("question_type", "open")

        if not image_path or not Path(image_path).exists():
            errors += 1
            continue

        # Validate image
        try:
            from PIL import Image

            img = Image.open(image_path)
            img.verify()
        except Exception:
            errors += 1
            continue

        # Run API inference
        try:
            result = client.predict(image_path=image_path, question=question)
        except Exception as e:
            errors += 1
            if not tqdm:
                print(f"  [ERROR] Sample {idx}: {e}")
            continue

        predictions.append(result)
        references.append(answer)
        questions_list.append(question)
        is_yesno_list.append(qtype.lower() in ["yes/no", "binary"])

        # Progress update every 20 samples (without tqdm)
        if not tqdm and (idx + 1) % 20 == 0:
            elapsed = time.time() - _start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(test_samples) - idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{idx + 1}/{len(test_samples)}]  {rate:.1f} samples/s  ~{remaining:.0f}s remaining"
            )

    elapsed_total = time.time() - _start_time
    print(
        f"  Done in {elapsed_total:.0f}s. {len(predictions)}/{len(test_samples)} successful ({errors} errors)"
    )

    # 4. Compute metrics
    print("\n[4/4] Computing metrics...")

    if len(predictions) == 0:
        print("[ERROR] No successful predictions")
        sys.exit(1)

    metrics = compute_metrics(predictions, references, questions_list, is_yesno_list)

    # 5. Save results
    output = {
        "model": "openai/gpt-4o",
        "mode": "api",
        "dataset": "vqa-rad",
        "test_samples": len(test_samples),
        "successful": len(predictions),
        "errors": errors,
        "yesno_count": yn_count,
        "open_count": open_count,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": elapsed_total,
        "predictions": [
            {
                "question": questions_list[i],
                "reference": references[i],
                "prediction": predictions[i],
                "is_yesno": is_yesno_list[i],
            }
            for i in range(len(predictions))
        ],
    }

    output_path = Path("experiments") / "api_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_path}")

    # 6. Summary table
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY — GPT-4o on VQA-RAD")
    print("=" * 60)

    if "yesno_accuracy" in metrics:
        print(f"  Yes/No Accuracy:       {metrics['yesno_accuracy'] * 100:.2f}%")
        print(
            f"    ({metrics['yesno_correct']}/{metrics['yesno_total']} correct, "
            f"{metrics['yesno_abstained']} abstained of {metrics['yesno_all_count']} total)"
        )
    for n in range(1, 5):
        key = f"bleu_{n}"
        if key in metrics:
            print(f"  BLEU-{n}:                {metrics[key]:.4f}")
    if "rouge_l_f1" in metrics:
        print(f"  ROUGE-L F1:             {metrics['rouge_l_f1']:.4f}")
    if "bertscore_f1" in metrics:
        print(f"  BERTScore F1:           {metrics['bertscore_f1']:.4f}")
    if "keyword_recall" in metrics:
        print(f"  Keyword Recall (open):  {metrics['keyword_recall'] * 100:.1f}%")
    print(f"  Samples evaluated:      {len(predictions)}")
    print(f"  Errors:                 {errors}")
    print(f"  Duration:               {elapsed_total:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
