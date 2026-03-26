"""
eval_trigger.py —— RAG Agent 触发机制评估（支持大规模并发 + 断点续跑）

【数据格式兼容】
支持以下两种 JSON 格式（自动识别）：

格式A：key 本身带内容（你描述的格式）
    [
        {"用户: 你好": "", "客服: 你好呀": "", "用户: 有没有XX案例": "", "caseid": ["KT001"]},
        ...
    ]

格式B：固定 key，value 是内容
    [
        {"用户": "你好", "客服": "你好呀", "caseid": []},
        ...
    ]

【标注逻辑】
    caseid 非空 → label=1（应触发检索）
    caseid 为空 → label=0（不应触发）

【指标】Accuracy / Precision / Recall / F1（以 label=1 为正类，Recall 最重要）

【用法】
    # 小批量调试
    python eval_trigger.py --data /path/to/dialog.json --limit 50 --workers 5

    # 全量（4万条，建议 workers=20~50）
    python eval_trigger.py --data /path/to/dialog.json --workers 30 --output eval_results_trigger

    # 断点续跑（自动跳过已完成的）
    python eval_trigger.py --data /path/to/dialog.json --workers 30 --resume
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

from langchain_core.messages import HumanMessage

import rag_agent_local as rag_module
from rag_agent_local import agent


# ──────────────────────────────────────────────
# 数据加载：兼容两种格式
# ──────────────────────────────────────────────

def parse_dialog(item: dict) -> dict | None:
    """
    解析单条对话记录，返回 {"query": str, "label": int, "caseid": list}
    兼容：
      格式A：key 本身带内容，如 {"用户: 你好": "", "客服: 回答": "", "caseid": [...]}
      格式B：固定 key，如 {"用户": "你好", "客服": "回答", "caseid": [...]}
    """
    caseid = item.get("caseid", [])
    if caseid is None:
        caseid = []

    user_texts = []

    for key, val in item.items():
        if key == "caseid":
            continue

        # 格式A：key 以 "用户" 开头，内容在 key 里
        if key.startswith("用户"):
            # key 形如 "用户: 你好" 或 "用户：你好" 或 "用户你好"
            # 取冒号后面的部分，或整个 key 去掉"用户"前缀
            text = ""
            if ":" in key:
                text = key.split(":", 1)[1].strip()
            elif "：" in key:
                text = key.split("：", 1)[1].strip()
            else:
                text = key[2:].strip()  # 去掉"用户"两字
            # 如果 val 也有内容，也拼上
            if isinstance(val, str) and val.strip():
                text = (text + " " + val).strip()
            if text:
                user_texts.append(text)

        # 格式B：key == "用户"，内容在 val 里
        elif key == "用户" and isinstance(val, str) and val.strip():
            user_texts.append(val.strip())

    if not user_texts:
        return None

    query = " ".join(user_texts)
    label = 1 if caseid else 0
    return {"query": query, "label": label, "caseid": caseid}


def load_dialogs(path: str) -> list[dict]:
    """加载 JSON 文件，返回标准化样本列表"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    skipped = 0
    for idx, item in enumerate(raw):
        parsed = parse_dialog(item)
        if parsed is None:
            skipped += 1
            continue
        parsed["_idx"] = idx  # 保留原始索引，用于断点续跑
        samples.append(parsed)

    print(f"加载完成：共 {len(raw)} 条原始数据，有效 {len(samples)} 条，跳过 {skipped} 条（无用户发言）")
    return samples


# ──────────────────────────────────────────────
# 单条推理
# ──────────────────────────────────────────────

def predict_trigger(query: str) -> tuple[int, str]:
    """
    运行 Agent，返回 (predicted, agent_answer)
    predicted=1 → 调用了检索工具
    predicted=0 → 未调用
    """
    messages = [HumanMessage(content=query)]
    triggered = False
    final_answer = ""

    try:
        for event in agent.stream(
            {"messages": messages},
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            msg_type = last_msg.__class__.__name__
            if msg_type == "AIMessage":
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    triggered = True
                elif last_msg.content:
                    final_answer = str(last_msg.content)
    except Exception as e:
        final_answer = f"[ERROR] {e}"

    return int(triggered), final_answer


# ──────────────────────────────────────────────
# 指标计算
# ──────────────────────────────────────────────

def compute_metrics(labels: list[int], preds: list[int]) -> dict:
    tp = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    fp = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    fn = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    tn = sum(l == 0 and p == 0 for l, p in zip(labels, preds))

    accuracy  = (tp + tn) / len(labels) if labels else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total":     len(labels),
        "pos_label": sum(labels),
        "neg_label": len(labels) - sum(labels),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


# ──────────────────────────────────────────────
# 主流程（并发 + 断点续跑）
# ──────────────────────────────────────────────

def run_eval(data_path: str, limit: int | None, workers: int, output_dir: str, resume: bool):
    print(f"\n{'='*60}")
    print(f"  RAG Agent 触发机制评估")
    print(f"  数据文件 : {data_path}")
    print(f"  样本上限 : {limit if limit else '全部'}")
    print(f"  并发线程 : {workers}")
    print(f"  断点续跑 : {'开启' if resume else '关闭'}")
    print(f"{'='*60}\n")

    samples = load_dialogs(data_path)
    if limit:
        samples = samples[:limit]

    pos_cnt = sum(s["label"] for s in samples)
    neg_cnt = len(samples) - pos_cnt
    print(f"样本分布：正样本（应触发）{pos_cnt} 条，负样本（不触发）{neg_cnt} 条\n")

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = os.path.join(output_dir, f"trigger_detail_{ts}.json")
    report_path = os.path.join(output_dir, f"trigger_report_{ts}.txt")

    # 断点续跑：加载已完成的结果
    done_indices = set()
    existing_details = []
    if resume:
        # 找最新的 detail 文件
        existing_files = sorted(
            [f for f in os.listdir(output_dir) if f.startswith("trigger_detail_")],
            reverse=True
        )
        if existing_files:
            prev_path = os.path.join(output_dir, existing_files[0])
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            existing_details = prev_data.get("details", [])
            done_indices = {d["_idx"] for d in existing_details}
            print(f"断点续跑：已加载 {len(done_indices)} 条历史结果，跳过这些样本\n")

    # 过滤掉已完成的
    todo_samples = [s for s in samples if s["_idx"] not in done_indices]
    print(f"本次需处理：{len(todo_samples)} 条\n")

    # 并发推理
    results = list(existing_details)  # 从历史结果开始
    lock = Lock()
    done_count = len(existing_details)
    total = len(samples)
    start_time = time.time()

    def process_one(s: dict) -> dict:
        pred, answer = predict_trigger(s["query"])
        label = s["label"]
        if label == 1 and pred == 1:
            result_type = "TP"
        elif label == 0 and pred == 0:
            result_type = "TN"
        elif label == 1 and pred == 0:
            result_type = "FN"
        else:
            result_type = "FP"
        return {
            "_idx":    s["_idx"],
            "query":   s["query"],
            "caseid":  s["caseid"],
            "label":   label,
            "pred":    pred,
            "result":  result_type,
            "answer":  answer,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, s): s for s in todo_samples}
        for future in as_completed(futures):
            detail = future.result()
            with lock:
                results.append(detail)
                done_count += 1
                elapsed = time.time() - start_time
                speed = (done_count - len(existing_details)) / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / speed if speed > 0 else 0
                icon = "✅" if detail["result"] in ("TP", "TN") else "❌"
                print(
                    f"[{done_count:>6}/{total}] {icon} {detail['result']:<2}  "
                    f"速度:{speed:.1f}条/s  预计剩余:{eta/60:.1f}min  "
                    f"query: {detail['query'][:40]}..."
                )

                # 每 500 条自动保存一次（防止中途崩溃丢数据）
                if done_count % 500 == 0:
                    _save_results(results, detail_path, report_path, data_path)
                    print(f"  [自动保存] 已保存 {done_count} 条到 {detail_path}")

    # 最终保存
    _save_results(results, detail_path, report_path, data_path)

    # 打印汇总
    labels = [d["label"] for d in results]
    preds  = [d["pred"]  for d in results]
    metrics = compute_metrics(labels, preds)

    print(f"\n{'='*60}")
    print(f"  评估结果汇总（共 {metrics['total']} 条）")
    print(f"{'='*60}")
    print(f"  正样本: {metrics['pos_label']}  负样本: {metrics['neg_label']}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}  （触发时有多少是真正需要的）")
    print(f"  Recall   : {metrics['recall']:.4f}  ← 核心：该触发时有多少触发了")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"{'='*60}")
    print(f"\n详细结果: {detail_path}")
    print(f"报告文件: {report_path}")

    return metrics


def _save_results(details: list, detail_path: str, report_path: str, data_path: str):
    """保存详细结果和报告"""
    labels = [d["label"] for d in details]
    preds  = [d["pred"]  for d in details]
    metrics = compute_metrics(labels, preds)

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": details}, f, ensure_ascii=False, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"RAG Agent 触发机制评估报告\n")
        f.write(f"数据: {data_path}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n\n--- FN（漏检，该触发未触发）---\n")
        for d in details:
            if d["result"] == "FN":
                f.write(f"\nquery : {d['query'][:100]}\n")
                f.write(f"caseid: {d['caseid']}\n")
                f.write(f"answer: {d['answer'][:100]}\n")
        f.write("\n\n--- FP（误触发，不该触发却触发了）---\n")
        for d in details:
            if d["result"] == "FP":
                f.write(f"\nquery : {d['query'][:100]}\n")
                f.write(f"answer: {d['answer'][:100]}\n")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Agent 触发机制评估")
    parser.add_argument("--data",    required=True,            help="对话数据 JSON 文件路径")
    parser.add_argument("--limit",   type=int,  default=None,  help="最多评估前 N 条（调试用）")
    parser.add_argument("--workers", type=int,  default=10,    help="并发线程数（建议 10~50）")
    parser.add_argument("--output",  default="eval_results_trigger", help="结果输出目录")
    parser.add_argument("--resume",  action="store_true",      help="断点续跑，跳过已完成的样本")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] 数据文件不存在: {args.data}")
        sys.exit(1)

    run_eval(args.data, args.limit, args.workers, args.output, args.resume)
