# A-Mem 原结构修改对照（供手工替换）

## 我的理解和最终范围

保留原 GitHub 仓库的目录、文件名和主要职责分工。继续使用原有 memory_layer、robust layer、
dataset loader 和 evaluator，不另建工程，不整体重写 README。

本方案只修改五个现有文件。断点辅助函数写在原有 `test_advanced_robust.py` 内；
58 条记忆在 `load_dataset.py` 中按参数截取，不新增子集脚本或数据文件。
原始 `data/locomo10.json`、模型提示词、评分公式、LICENSE、Figure 和其他原文件保留。

实验范围：第一个样本的前 3 场完整对话，58 条记忆，同样的十道题，全部标注证据保留。
固定范围与 DeepSeek 后，对比 k=10 和 k=5；两组各重复 3～5 次。
本方案仍需真实运行、结果对比、报告和录制演示，不能仅凭代码改动认定作业完成。

## 如何使用这份对照

对照基准是原仓库提交 `0ee2a5a`（2026-11-04 的 README 更新）。
定位优先使用文件和函数名，行号仅用于这个基准；远端文件若已改动，行号可能变化。

`A-mem_原结构修改参考/` 中只有五个修改后的原文件，可以打开参照。
下文逐文件列出修改点和完整差异：`-` 是原代码，`+` 是替换或新增代码，空格开头是定位上下文。
差异块不是可以直接运行的 Python；手工修改时从对应参考文件复制完整代码块，去掉差异标记。
也可以比较原文件和参考文件后，替换确认过的整个文件。

需要更新的原文件：

| 文件 | 修改目的 |
|---|---|
| load_dataset.py | 按场次缩小记忆，并筛选证据完整的题目 |
| test_advanced_robust.py | 参数传递、断点续跑、API 检查和结果记录 |
| memory_layer_robust.py | 显式请求超时，避免双层重试 |
| utils.py | 补充 NLTK punkt_tab 资源 |
| README.md | 更新最前面的作业说明和运行命令 |

重要：前两个文件必须一起更新，因为评测入口会传入新增的加载参数。

## 修改后怎么运行

完成依赖安装和 DeepSeek 环境变量配置后，在原项目目录运行：

```bash
python test_advanced_robust.py --backend openai --model deepseek-chat --dataset data/locomo10.json --max_samples 1 --max_sessions 3 --max_questions 10 --retrieve_k 10 --checkpoint_every 10 --output results/fast58/k10_trial1.json
python test_advanced_robust.py --backend openai --model deepseek-chat --dataset data/locomo10.json --max_samples 1 --max_sessions 3 --max_questions 10 --retrieve_k 5 --checkpoint_every 10 --output results/fast58/k5_trial1.json
```

输出应有 `retained_turns: 58`、`source_qa_indices: [0,1,2,3,4,5,6,7,8,9]`、`total_questions: 10`。
后续将 trial1 改为 trial2、trial3 等输出名。相同范围的 k 对照会复用完整缓存。

`--max_sessions 0` 表示不截断对话；首个样本会恢复为 419 条记忆。
`--max_samples 0 --max_questions 0 --max_sessions 0` 可选取完整原始数据，但会有更大运行量。
加载函数不传限制参数时恢复全量，因此只运行这里给出的 robust 命令，不把原版
`test_advanced.py` 的默认运行当成快速实验。

缩短会话会生成不同缓存，不直接复用旧的 419 条断点。没有 Drive 时，运行时重置会清除缓存。

## 删除、移动和替换说明

本方案没有删除原仓库的任何项目文件。差异中的 `-` 表示旧位置的代码不再保留，
既可能是真正删除，也可能是移动到另一个位置，或者由新实现替换。

| 文件与位置 | 原代码或旧逻辑 | 操作类型 | 原因与替代方式 |
|---|---|---|---|
| load_dataset.py / parse_conversation | 直接遍历全部 conv_data.items() | 替换 | 改为按 session 编号排序并选取前 N 场；原始数据文件不变。 |
| load_dataset.py / load_locomo_dataset | if sample_idx >= 1: break | 替换 | 用 max_samples 参数控制样本数，避免限制写死。 |
| load_dataset.py / QA 循环 | if qa_idx >= 10: break | 替换 | 用保留下来的题数控制 max_questions；先筛选证据完整的题目，避免筛选前就凑满十个原始索引。 |
| load_dataset.py / QA 解析之后 | conversation = parse_conversation(...) | 移动并修改 | 移至 QA 解析之前，并传 max_sessions；先确定保留历史，才能判断题目证据是否全部保留。会话解析功能没有删除。 |
| load_dataset.py / 总体统计 | 无条件除以 len(samples)，以及无默认值的 min/max | 替换 | 增加空数据保护；统计仍输出，避免没有样本时除零或空序列报错。 |
| test_advanced_robust.py / 顶部导入 | cache_is_current、mark_cache | 删除这两个导入 | 新断点用数据、配置和构建代码的哈希及格式版本检查一致性，不再使用旧 .version 标记。memory_layer.py 内的原函数仍保留。 |
| test_advanced_robust.py / 顶部导入 | nltk、SentenceTransformer、pytorch_cos_sim、statistics、tqdm | 删除本文件中的未使用导入 | 全局重复初始化移除后，评测脚本不再使用这些名称；实际检索模型仍在原记忆层中，评分依赖仍在 utils.py 中。requirements.txt 不删这些依赖。 |
| test_advanced_robust.py / 顶部 | 全局 NLTK 下载块 | 删除重复初始化 | utils.py 保留资源下载并增加 punkt_tab；评分模块在 API 连通性检查之后加载，避免 API 不通时先做重复初始化。 |
| test_advanced_robust.py / 顶部 | 全局 sentence_model = SentenceTransformer(...) | 删除未使用对象 | 这个评测脚本的全局 sentence_model 没被问答流程使用。原记忆检索模型和 utils.py 中的评分模型仍保留，没有取消向量检索。 |
| test_advanced_robust.py / 顶部 | from utils import calculate_metrics, aggregate_metrics, score_prediction | 部分删除、部分移动 | calculate_metrics 在此脚本未直接调用；aggregate_metrics 和 score_prediction 移至 evaluate_dataset 的 API 检查之后导入。评分功能和公式未删除。 |
| test_advanced_robust.py / 导入区 | import pickle 的旧位置 | 移动 | 仍然导入 pickle，只是与断点所需的标准库放在一起；不要把旧位置的减号理解为取消 pickle。 |
| test_advanced_robust.py / load_or_build | 三个分开的缓存文件、旧缓存加载/重建/版本告警分支 | 整块替换 | 改用一个原子断点保存笔记、向量、关联、进度和演化状态，避免文件状态不一致；旧缓存目录和已有文件不会被程序删除，也不会自动迁移。 |
| test_advanced_robust.py / load_or_build | 完整遍历后才一次性写缓存的 for 循环 | 整块替换 | 改由同文件 build_memory_with_resume 遍历，并每 N 条保存。agent.add_memory 方法仍保留，记忆添加功能未删除。 |
| memory_layer_robust.py / RobustOpenAIController.__init__ | self.client = OpenAI(api_key=api_key) | 替换 | 改为显式 timeout=60.0、max_retries=0；外层原重试保留，避免两层重试叠加。 |
| utils.py | 无 | 仅新增 | 只增加 punkt_tab 下载，没有删除原资源下载、评分函数或公式。 |
| README.md / 顶部作业说明 | “确保预算内完成”的旧说法、旧的 Run command used 命令 | 替换 | 改为可验证的范围说明和 Suggested scoped run；没有实测依据的完成保证不保留。后面的上游正文原样保留。 |

函数签名加参数、初始化列表加字段、文件写入加 UTF-8，以及结果字典加元数据，
在差异中也会表现为一减一加；这些属于原语句的替换，不是功能删除。

缩小到 58 条记忆属于运行时选择范围，并未从 data/locomo10.json 中删除其余对话。
需要全量历史时使用 --max_sessions 0。保留的十道题是否仍含全部证据已做本地检查。

## 逐文件对照


### load_dataset.py

- 在 LoCoMoSample 数据类末尾添加可选 source_qa_indices，以便结果记录题目的原始索引。
- parse_conversation 增加 max_sessions 参数，按 session 编号保留前 N 场完整对话。
- load_locomo_dataset 增加样本、题目、场次参数，移除两条硬编码 break。先截取会话，再只保留全部标注证据仍在历史中的题目。
- 默认三个场次得到 58 条记忆和原来的第 0–9 题。max_sessions=None 时不做证据筛选，恢复原始数据范围。


定位（原文件行号 → 参考文件行号）：


- `class LoCoMoSample`：63 → 63
- `def parse_conversation`：92 → 93
- `def load_locomo_dataset`：111 → 118



```diff
--- a/load_dataset.py
+++ b/load_dataset.py
@@ -68,6 +68,7 @@
     event_summary: EventSummary
     observation: Observation
     session_summary: Dict[str, str]
+    source_qa_indices: Optional[List[int]] = None
 
 def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
     """Parse a single session's data, including turns with images by using their captions."""
@@ -89,10 +90,16 @@
         ))
     return Session(session_id=session_id, date_time=date_time, turns=turns)
 
-def parse_conversation(conv_data: dict) -> Conversation:
+def parse_conversation(conv_data: dict, max_sessions: Optional[int] = None) -> Conversation:
     """Parse conversation data."""
     sessions = {}
-    for key, value in conv_data.items():
+    session_keys = sorted(
+        (key for key, value in conv_data.items()
+         if key.startswith('session_') and isinstance(value, list)),
+        key=lambda key: int(key.split('_')[1]),
+    )
+    for key in session_keys[:max_sessions]:
+        value = conv_data[key]
         if key.startswith("session_") and isinstance(value, list):
             session_id = int(key.split("_")[1])
             date_time = conv_data.get(f"{key}_date_time")
@@ -108,16 +115,25 @@
         sessions=sessions
     )
 
-def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
+def load_locomo_dataset(file_path: Union[str, Path], max_samples: Optional[int] = None,
+                        max_questions: Optional[int] = None,
+                        max_sessions: Optional[int] = None) -> List[LoCoMoSample]:
     """
     Load the LoComo dataset from a JSON file, including image-based content by using captions.
     
     Args:
         file_path: Path to the JSON file containing the dataset
+        max_samples: Maximum number of conversations; None loads all.
+        max_questions: Maximum questions per conversation; None loads all.
+        max_sessions: Chronological session prefix; keep only QAs with contained evidence.
         
     Returns:
         List of LoCoMoSample objects containing the parsed data
     """
+    for name, limit in (("max_samples", max_samples), ("max_questions", max_questions), ("max_sessions", max_sessions)):
+        if limit is not None and limit <= 0:
+            raise ValueError(f"{name} must be positive or None")
+
     if isinstance(file_path, str):
         file_path = Path(file_path)
         
@@ -133,15 +149,28 @@
     qa_counts_per_sample = []
     
     for sample_idx, sample in enumerate(data):
-        if sample_idx >= 1: break
+        if max_samples is not None and sample_idx >= max_samples:
+            break
         try:
+            # Choose the history before selecting supported questions.
+            conversation = parse_conversation(sample['conversation'], max_sessions)
+            retained_ids = {turn.dia_id for session in conversation.sessions.values()
+                            for turn in session.turns}
             # Parse QA data
             qa_list = []
+            source_qa_indices = []
             sample_qa_count = 0
             sample_image_qa_count = 0
             
             for qa_idx, qa in enumerate(sample["qa"]):
-                if qa_idx >= 10: break
+                if max_sessions is not None:
+                    # All evidence must fit the fixed history; no answer-based retrieval hints.
+                    if (qa.get('category') not in (1, 2, 3, 4) or qa.get('answer') is None
+                            or not qa.get('evidence')
+                            or not set(qa['evidence']).issubset(retained_ids)):
+                        continue
+                if max_questions is not None and len(qa_list) >= max_questions:
+                    break
                 try:
                     # Check if QA has image evidence
                     has_image_evidence = False
@@ -168,6 +197,7 @@
                         adversarial_answer=qa.get("adversarial_answer")
                     )
                     qa_list.append(qa_obj)
+                    source_qa_indices.append(qa_idx)
                     sample_qa_count += 1
                     
                 except KeyError as e:
@@ -179,8 +209,7 @@
                     print(f"QA data: {qa}")
                     raise e
             
-            # Parse conversation
-            conversation = parse_conversation(sample["conversation"])
+            # Conversation was selected before QA filtering.
             
             # Parse event summary
             event_summary = EventSummary(events=sample["event_summary"])
@@ -198,7 +227,8 @@
                 conversation=conversation,
                 event_summary=event_summary,
                 observation=observation,
-                session_summary=session_summary
+                session_summary=session_summary,
+                source_qa_indices=source_qa_indices
             )
             samples.append(sample_obj)
             
@@ -220,9 +250,9 @@
     print("\nOverall Statistics:")
     print(f"Total QAs: {total_qa}")
     print(f"Total QAs with image evidence: {total_image_qa}")
-    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
-    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
-    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")
+    print(f"Average QAs per sample: {total_qa / len(samples) if samples else 0:.2f}")
+    print(f"Min QAs in a sample: {min(qa_counts_per_sample, default=0)}")
+    print(f"Max QAs in a sample: {max(qa_counts_per_sample, default=0)}")
     
     return samples
```


### test_advanced_robust.py

- 只增加标准库导入，断点辅助函数直接放进这个原有文件，不新增 memory_checkpoint.py。
- evaluate_dataset 新增范围参数及 API 预检查；load_or_build 改为同文件的断点恢复逻辑。笔记、向量、关联和演化状态在一个文件中原子保存。
- main 增加 --max_samples、--max_questions、--max_sessions、--checkpoint_every 并传入 evaluate_dataset。
- 结果增加 k、源数据哈希、保留会话、原题索引、构建失败记录和时间；模型生成、记忆演化提示词和评分方法保持沿用。


定位（原文件行号 → 参考文件行号）：


- `class RobustAdvancedMemAgent`：55 → 128
- `def evaluate_dataset`：205 → 278
- `def load_or_build`：246 → 342
- `def main`：394 → 491



```diff
--- a/test_advanced_robust.py
+++ b/test_advanced_robust.py
@@ -8,7 +8,7 @@
 """
 
 from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
-from memory_layer import FAILURES, valid_link, cache_is_current, mark_cache
+from memory_layer import FAILURES, valid_link, MEMORY_VERSION
 from llm_text_parsers import (
     parse_plain_text_answer,
     parse_relevant_parts,
@@ -17,39 +17,112 @@
 import os
 import json
 import argparse
+import hashlib
+import time
+import tempfile
+import pickle
 import logging
 from typing import List, Dict, Optional
 from pathlib import Path
 import numpy as np
 from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
-import nltk
-from sentence_transformers import SentenceTransformer
-from sentence_transformers.util import pytorch_cos_sim
-import statistics
 from collections import defaultdict
 import copy
-import pickle
 import random
-from tqdm import tqdm
-from utils import calculate_metrics, aggregate_metrics, score_prediction
 from datetime import datetime
 
-# Download required NLTK data
-try:
-    nltk.data.find('tokenizers/punkt')
-    nltk.data.find('wordnet')
-except LookupError:
-    nltk.download('punkt')
-    nltk.download('wordnet')
-
-# Initialize SentenceTransformer model (this will be reused)
-try:
-    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
-except Exception as e:
-    print(f"Warning: Could not load SentenceTransformer model: {e}")
-    sentence_model = None
-
 logger = logging.getLogger("amem_robust")
+
+
+# Checkpoint helpers live in this original file to retain the repository layout.
+def memory_identity(turns, configuration):
+    """Bind a cache to the conversation, backend, model and construction settings."""
+    payload = json.dumps({"turns": turns, "configuration": configuration},
+                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
+    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
+
+def _save_checkpoint(path, state):
+    """Replace one self-contained snapshot only after it has been written fully."""
+    path = Path(path)
+    path.parent.mkdir(parents=True, exist_ok=True)
+    temp_path = None
+    try:
+        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
+                                         suffix=".tmp", delete=False) as handle:
+            temp_path = handle.name
+            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
+            handle.flush()
+            os.fsync(handle.fileno())
+        os.replace(temp_path, path)
+    finally:
+        if temp_path and os.path.exists(temp_path):
+            os.unlink(temp_path)
+
+def build_memory_with_resume(system, turns, checkpoint_path, identity,
+                             checkpoint_every=10, failures=None, logger=None):
+    """Resume from the last completed snapshot, keeping evolved notes and embeddings.
+
+    A failed/in-progress add_note is never saved. At most checkpoint_every - 1
+    successfully constructed notes need rebuilding after an interruption.
+    """
+    if checkpoint_every <= 0:
+        raise ValueError("checkpoint_every must be positive")
+    logger = logger or logging.getLogger(__name__)
+    failures = failures if failures is not None else {}
+    initial_failures = dict(failures)
+    previous_failures = {}
+    path = Path(checkpoint_path)
+    completed = 0
+    resumed_from = 0
+
+    if path.exists():
+        with path.open("rb") as handle:
+            state = pickle.load(handle)
+        if state.get("format_version") != 1 or state.get("identity") != identity:
+            raise ValueError(f"Checkpoint does not match this experiment: {path}")
+        completed = state["completed"]
+        embeddings = state["embeddings"]
+        if not (0 <= completed <= len(turns)
+                and completed == len(state["memories"]) == len(state["corpus"])
+                and (len(embeddings) if embeddings is not None else 0) == completed):
+            raise ValueError(f"Inconsistent memory checkpoint: {path}")
+        system.memories = state["memories"]
+        system.retriever.corpus = state["corpus"]
+        system.retriever.embeddings = embeddings
+        system.retriever.document_ids = state["document_ids"]
+        system.evo_cnt = state["evo_cnt"]
+        previous_failures = state.get("build_failures", {})
+        resumed_from = completed
+        logger.info("Resuming memory from %d/%d completed turns", completed, len(turns))
+
+    def snapshot():
+        retriever = system.retriever
+        if not (len(system.memories) == len(retriever.corpus) == completed
+                and (len(retriever.embeddings) if retriever.embeddings is not None else 0) == completed):
+            raise ValueError("Memory and embedding index are out of sync; checkpoint was not replaced")
+        build_failures = dict(previous_failures)
+        for key, count in failures.items():
+            delta = count - initial_failures.get(key, 0)
+            if delta:
+                build_failures[key] = build_failures.get(key, 0) + delta
+        state = {
+            "format_version": 1, "identity": identity, "completed": completed,
+            "memories": system.memories, "corpus": retriever.corpus,
+            "embeddings": retriever.embeddings, "document_ids": retriever.document_ids,
+            "evo_cnt": system.evo_cnt, "build_failures": build_failures,
+        }
+        _save_checkpoint(path, state)
+        logger.info("Saved memory checkpoint: %d/%d turns", completed, len(turns))
+        return build_failures
+
+    for content, timestamp in turns[completed:]:
+        system.add_note(content, time=timestamp)
+        completed += 1
+        if completed % checkpoint_every == 0:
+            snapshot()
+    build_failures = snapshot()
+    return {"completed": completed, "resumed_from": resumed_from,
+            "build_failures": build_failures, "checkpoint": str(path)}
 
 
 class RobustAdvancedMemAgent:
@@ -195,7 +268,7 @@
     eval_logger.addHandler(console_handler)
 
     if log_file:
-        file_handler = logging.FileHandler(log_file)
+        file_handler = logging.FileHandler(log_file, encoding="utf-8")
         file_handler.setFormatter(formatter)
         eval_logger.addHandler(file_handler)
 
@@ -207,8 +280,11 @@
                      temperature_c5: float = 0.5, retrieve_k: int = 10,
                      sglang_host: str = "http://localhost", sglang_port: int = 30000,
                      legacy_c5_reference: bool = False, seed: int = 42,
-                     refresh_index: bool = False, memory_scope: str = "conversation"):
+                     refresh_index: bool = False, memory_scope: str = "conversation",
+                     max_samples: Optional[int] = 1, max_questions: Optional[int] = 10,
+                     checkpoint_every: int = 10, max_sessions: Optional[int] = 3):
     """Evaluate the robust agent on the LoComo dataset."""
+    evaluation_started = time.perf_counter()
     timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
     log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{memory_scope}_{timestamp}.log"
     log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
@@ -219,13 +295,21 @@
     eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")
     FAILURES.clear()
 
-    samples = load_locomo_dataset(dataset_path)
+    samples = load_locomo_dataset(dataset_path, max_samples, max_questions, max_sessions)
     eval_logger.info(f"Loaded {len(samples)} samples")
 
     if ratio < 1.0:
         num_samples = max(1, int(len(samples) * ratio))
         samples = samples[:num_samples]
         eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
+
+    if not samples or not any(sample.qa for sample in samples):
+        raise ValueError("The selected dataset contains no evaluation questions")
+    eval_logger.info("Checking LLM connectivity before building memories")
+    RobustLLMController(backend=backend, model=model, sglang_host=sglang_host,
+                        sglang_port=sglang_port, check_connection=True)
+    # Load metric models/resources only after the API check succeeds.
+    from utils import aggregate_metrics, score_prediction
 
     results = []
     all_metrics = []
@@ -237,57 +321,46 @@
     error_num = 0
     memories_dir = os.path.join(
         os.path.dirname(__file__),
-        "cached_memories_robust_{}_{}{}".format(backend, model, "_refresh" if refresh_index else ""),
+        "cached_memories_resumable",
     )
     os.makedirs(memories_dir, exist_ok=True)
     allow_categories = [1, 2, 3, 4, 5]
     cached_samples, stale_cache_samples = [], []
+    memory_builds = {}
+    qa_elapsed_seconds = 0.0
+    source_digest = hashlib.sha256()
+    for name in ("memory_layer.py", "memory_layer_robust.py", "llm_text_parsers.py"):
+        source_digest.update((Path(__file__).parent / name).read_bytes())
+    memory_configuration = {
+        "backend": backend, "model": model, "embedding_model": "all-MiniLM-L6-v2",
+        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") if backend == "openai"
+                    else f"{sglang_host}:{sglang_port}" if backend in ("sglang", "vllm") else "ollama",
+        "refresh_index": refresh_index, "memory_version": MEMORY_VERSION,
+        "source_sha256": source_digest.hexdigest(),
+    }
 
     def load_or_build(sample_idx, sample):
-        """An agent holding this conversation's memory, loaded from the cache or built and cached."""
+        """Build the full conversation, or resume its last atomic checkpoint."""
         agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                        sglang_host, sglang_port, refresh_index)
 
-        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
-        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
-        retriever_cache_embeddings_file = os.path.join(
-            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
+        turns = [("Speaker " + turn.speaker + "says : " + turn.text, session.date_time)
+                 for session in sample.conversation.sessions.values() for turn in session.turns]
+        identity = memory_identity(turns, memory_configuration)
+        checkpoint_path = os.path.join(memories_dir, f"sample_{sample_idx}_{identity}.pkl")
+        eval_logger.info("Sample %d requires %d memory notes; questions=%d",
+                         sample_idx, len(turns), len(sample.qa))
+        build_started = time.perf_counter()
+        memory_builds[str(sample_idx)] = build_memory_with_resume(
+            agent.memory_system, turns, checkpoint_path, identity,
+            checkpoint_every=checkpoint_every, failures=FAILURES, logger=eval_logger,
         )
-
-        if os.path.exists(memory_cache_file):
-            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
-            with open(memory_cache_file, 'rb') as f:
-                cached_memories = pickle.load(f)
-            agent.memory_system.memories = cached_memories
-            if os.path.exists(retriever_cache_file):
-                eval_logger.info(f"Found retriever cache files")
-                agent.memory_system.retriever = agent.memory_system.retriever.load(
-                    retriever_cache_file, retriever_cache_embeddings_file
-                )
-            else:
-                eval_logger.info(f"No retriever cache found, loading from memory")
-                agent.memory_system.consolidate_memories()  # the document format of this memory system
-            eval_logger.info(f"Successfully loaded {len(cached_memories)} memories")
+        memory_builds[str(sample_idx)]["elapsed_seconds"] = time.perf_counter() - build_started
+        if memory_builds[str(sample_idx)]["resumed_from"]:
             cached_samples.append(sample_idx)
-            if not cache_is_current(memory_cache_file):
-                stale_cache_samples.append(sample_idx)
-                eval_logger.warning(f"Memories of sample {sample_idx} were cached by another version of the memory "
-                                    f"layer; delete {memories_dir} to rebuild them")
-        else:
-            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new memories.")
-
-            for _, turns in sample.conversation.sessions.items():
-                for turn in turns.turns:
-                    turn_datatime = turns.date_time
-                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
-                    agent.add_memory(conversation_tmp, time=turn_datatime)
-
-            memories_to_cache = agent.memory_system.memories
-            with open(memory_cache_file, 'wb') as f:
-                pickle.dump(memories_to_cache, f)
-            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
-            mark_cache(memory_cache_file)
-            eval_logger.info(f"Successfully cached {len(memories_to_cache)} memories")
+        if memory_builds[str(sample_idx)]["build_failures"]:
+            eval_logger.warning("Memory construction has absorbed failures: %s",
+                                memory_builds[str(sample_idx)]["build_failures"])
         return agent
 
     if memory_scope == "pooled":
@@ -309,9 +382,12 @@
 
                 # the category-5 option order comes from the question's own generator
                 rng = random.Random(f"{seed}:{sample_idx}:{qa_idx}")
+                answer_started = time.perf_counter()
                 prediction, user_prompt, raw_context = agent.answer_question(
                     qa.question, qa.category, qa.final_answer, rng
                 )
+                answer_elapsed_seconds = time.perf_counter() - answer_started
+                qa_elapsed_seconds += answer_elapsed_seconds
 
                 # Parse the prediction (handles both JSON and plain text)
                 prediction = parse_plain_text_answer(prediction)
@@ -336,6 +412,7 @@
                     "category": qa.category,
                     "adversarial_answer": qa.adversarial_answer,
                     "metrics": metrics,
+                    "answer_elapsed_seconds": answer_elapsed_seconds,
                 }
                 results.append(result)
 
@@ -354,6 +431,25 @@
         },
         "aggregate_metrics": aggregate_results,
         "ratio": ratio,
+        "backend": backend,
+        "max_samples": max_samples,
+        "max_questions": max_questions,
+        "max_sessions": max_sessions,
+        "dataset_source_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
+        "subset_selection": {sample.sample_id: {
+            "session_ids": list(sample.conversation.sessions),
+            "source_qa_indices": sample.source_qa_indices,
+            "retained_turns": sum(len(session.turns) for session in sample.conversation.sessions.values()),
+        } for sample in samples},
+        "retrieve_k": retrieve_k,
+        "memory_configuration": memory_configuration,
+        "decoding": {"temperature": 0.7, "temperature_c5": temperature_c5,
+                     "max_tokens": 1000 if backend != "ollama" else "provider default",
+                     "top_p": "provider default"},
+        "timing": {"total_elapsed_seconds": time.perf_counter() - evaluation_started,
+                   "qa_answer_elapsed_seconds": qa_elapsed_seconds},
+        "checkpoint_every": checkpoint_every,
+        "memory_builds": memory_builds,
         "seed": seed,
         "legacy_c5_reference": legacy_c5_reference,
         "refresh_index": refresh_index,
@@ -370,7 +466,8 @@
     eval_logger.info(f"LLM failures absorbed in this run: {dict(FAILURES) or 'none'}")
 
     if output_path:
-        with open(output_path, 'w') as f:
+        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
+        with open(output_path, 'w', encoding="utf-8") as f:
             json.dump(final_results, f, indent=2)
         eval_logger.info(f"Results saved to {output_path}")
 
@@ -403,6 +500,14 @@
                         help="Path to save evaluation results")
     parser.add_argument("--ratio", type=float, default=1.0,
                         help="Ratio of dataset to evaluate (0.0 to 1.0)")
+    parser.add_argument("--max_samples", type=int, default=1,
+                        help="Maximum conversations (default: 1; 0: all)")
+    parser.add_argument("--max_questions", type=int, default=10,
+                        help="Maximum questions per conversation (default: 10; 0: all)")
+    parser.add_argument("--max_sessions", type=int, default=3,
+                        help="Keep earliest N sessions and QAs with contained evidence (0: all)")
+    parser.add_argument("--checkpoint_every", type=int, default=10,
+                        help="Save memory construction every N completed turns")
     parser.add_argument("--backend", type=str, default="openai",
                         help="Backend to use (openai, ollama, sglang, or vllm)")
     parser.add_argument("--temperature_c5", type=float, default=0.5,
@@ -431,6 +536,11 @@
 
     if args.ratio <= 0.0 or args.ratio > 1.0:
         raise ValueError("Ratio must be between 0.0 and 1.0")
+    if (args.max_samples < 0 or args.max_questions < 0 or args.max_sessions < 0
+            or args.checkpoint_every <= 0):
+        raise ValueError("Sample/question limits must be nonnegative and checkpoint_every positive")
+    if args.retrieve_k <= 0:
+        raise ValueError("retrieve_k must be positive")
 
     dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
     output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None
@@ -440,6 +550,8 @@
         args.backend, args.temperature_c5, args.retrieve_k,
         args.sglang_host, args.sglang_port, args.legacy_c5_reference, args.seed,
         args.refresh_index, args.memory_scope,
+        args.max_samples or None, args.max_questions or None, args.checkpoint_every,
+        args.max_sessions or None,
     )
```


### memory_layer_robust.py

- 只替换 RobustOpenAIController.__init__ 中创建 OpenAI 客户端的一行：timeout=60.0、max_retries=0。
- 保留原有外层最多两次重试，避免 SDK 重试与外层重试相乘。超时不等于整个实验时长限制。


定位（原文件行号 → 参考文件行号）：


- `class RobustOpenAIController`：97 → 97
- `self.client = OpenAI`：108 → 109



```diff
--- a/memory_layer_robust.py
+++ b/memory_layer_robust.py
@@ -105,7 +105,8 @@
             api_key = os.getenv('OPENAI_API_KEY')
         if api_key is None:
             raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
-        self.client = OpenAI(api_key=api_key)
+        # The retry decorator owns retries; avoid multiplying them with SDK retries.
+        self.client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
 
     @retry_llm_call(max_retries=2)
     def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
```


### utils.py

- 在原有 NLTK 资源下载处增加 punkt_tab；不改 F1、BLEU、ROUGE 等评分公式。


定位（原文件行号 → 参考文件行号）：


- `nltk.download('punkt'`：22 → 22



```diff
--- a/utils.py
+++ b/utils.py
@@ -20,6 +20,7 @@
 # Download required NLTK data
 try:
     nltk.download('punkt', quiet=True)
+    nltk.download('punkt_tab', quiet=True)
     nltk.download('wordnet', quiet=True)
 except Exception as e:
     print(f"Error downloading NLTK data: {e}")
```


### README.md

- 只修改最前面的学生复现说明及运行命令，补充范围、k 对照和未实测状态。
- 从 # Agentic Memory 开始的上游 README 正文保留原样；不新增或迁移上游文档。


定位（原文件行号 → 参考文件行号）：


- `Reproducibility Note`：1 → 1
- `# Agentic Memory`：24 → 28



```diff
--- a/README.md
+++ b/README.md
@@ -10,15 +10,19 @@
 > os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
 > ```
 > 
-> **2. Modification (Timeout Prevention):**
-> Because building the memory cache for the full conversation timed out (due to heavy API serial calls), we modified **`load_dataset.py`** to add a hard slicing guard. This limits the evaluation to the first sample and the first 10 questions (`if qa_idx >= 10: break`), ensuring the evaluation script can complete within the compute budget.
+> **2. Scoped evaluation and resumable construction:**
+> The scoped evaluation keeps the first three complete sessions of the first LoCoMo sample: 58 memory turns and the original first ten questions, whose annotated evidence is retained. Limits are configurable, and memory construction saves atomic checkpoints every ten completed turns. This reduces the workload and permits resuming interrupted construction; actual runtime and accuracy must be measured.
 > 
-> **Run command used:**
+> **Suggested scoped run:**
 > ```bash
-> python test_advanced_robust.py --backend openai --model deepseek-chat --dataset data/locomo10.json --output test_final.json
+> python test_advanced_robust.py --backend openai --model deepseek-chat --dataset data/locomo10.json --max_samples 1 --max_sessions 3 --max_questions 10 --retrieve_k 10 --checkpoint_every 10 --output results/fast58/k10_trial1.json
 > ```
 > 
 > *Note: The API key is loaded via Colab Secrets and is NOT hardcoded in any script.*
+>
+> **Target and measured modification:** Verify the memory-based QA pipeline on this fixed subset, then change only `retrieve_k` from 10 to 5. Run 3–5 trials per setting and report F1, BLEU-1, and QA time with mean/standard deviation across runs. The robust memory method, parsers, and scorers are inherited from upstream.
+>
+> **Status and limitations:** Local structural checks are available; real DeepSeek results and modification impact have not been supplied. Shortened history and a different LLM mean these results are not directly comparable to the paper's full LoCoMo evaluation. Do not claim a successful reproduction without measured outputs.
 
 
 # Agentic Memory 🧠
```

## GitHub 提交和作业状态

只上传这五个原文件到原仓库对应位置。可以分两次提交：

1. load_dataset.py + test_advanced_robust.py：范围参数和断点恢复。
2. memory_layer_robust.py + utils.py + README.md：运行兼容和说明。

不要把本对照文档、参考文件夹或 ZIP 当成项目运行依赖。
此方案不依赖之前生成的辅助模块、新数据文件或笔记本目录。
如果你是修改之前的 Colab 内嵌代码，需要替换里面的原文件；重新上传旧笔记本不会自动使用新 GitHub 代码。
更简单的做法是新建 Colab，在同一目录克隆更新后的原仓库，再运行上述命令。

F1/BLEU-1 沿用原评分程序，报告运行之间的均值和标准差；JSON 内原有 std 是题目间的波动。
检查 llm_failures 和 memory_builds 的 build_failures，不把错误降级输出当成正常实测。
目前尚无真实 API 成功完成的证据；需要补齐真实结果、改动效果、报告和演示链接。
