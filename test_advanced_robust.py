"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4o-mini --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
"""

from memory_layer_robust import RobustLLMController, RobustAgenticMemorySystem
from memory_layer import FAILURES, valid_link, MEMORY_VERSION
from llm_text_parsers import (
    parse_plain_text_answer,
    parse_relevant_parts,
    parse_keywords_response,
)
import os
import json
import argparse
import hashlib
import time
import tempfile
import pickle
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from collections import defaultdict
import copy
import random
from datetime import datetime

logger = logging.getLogger("amem_robust")


# Checkpoint helpers live in this original file to retain the repository layout.
def memory_identity(turns, configuration):
    """Bind a cache to the conversation, backend, model and construction settings."""
    payload = json.dumps({"turns": turns, "configuration": configuration},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _save_checkpoint(path, state):
    """Replace one self-contained snapshot only after it has been written fully."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temp_path = handle.name
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def build_memory_with_resume(system, turns, checkpoint_path, identity,
                             checkpoint_every=10, failures=None, logger=None):
    """Resume from the last completed snapshot, keeping evolved notes and embeddings.

    A failed/in-progress add_note is never saved. At most checkpoint_every - 1
    successfully constructed notes need rebuilding after an interruption.
    """
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    logger = logger or logging.getLogger(__name__)
    failures = failures if failures is not None else {}
    initial_failures = dict(failures)
    previous_failures = {}
    path = Path(checkpoint_path)
    completed = 0
    resumed_from = 0

    if path.exists():
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if state.get("format_version") != 1 or state.get("identity") != identity:
            raise ValueError(f"Checkpoint does not match this experiment: {path}")
        completed = state["completed"]
        embeddings = state["embeddings"]
        if not (0 <= completed <= len(turns)
                and completed == len(state["memories"]) == len(state["corpus"])
                and (len(embeddings) if embeddings is not None else 0) == completed):
            raise ValueError(f"Inconsistent memory checkpoint: {path}")
        system.memories = state["memories"]
        system.retriever.corpus = state["corpus"]
        system.retriever.embeddings = embeddings
        system.retriever.document_ids = state["document_ids"]
        system.evo_cnt = state["evo_cnt"]
        previous_failures = state.get("build_failures", {})
        resumed_from = completed
        logger.info("Resuming memory from %d/%d completed turns", completed, len(turns))

    def snapshot():
        retriever = system.retriever
        if not (len(system.memories) == len(retriever.corpus) == completed
                and (len(retriever.embeddings) if retriever.embeddings is not None else 0) == completed):
            raise ValueError("Memory and embedding index are out of sync; checkpoint was not replaced")
        build_failures = dict(previous_failures)
        for key, count in failures.items():
            delta = count - initial_failures.get(key, 0)
            if delta:
                build_failures[key] = build_failures.get(key, 0) + delta
        state = {
            "format_version": 1, "identity": identity, "completed": completed,
            "memories": system.memories, "corpus": retriever.corpus,
            "embeddings": retriever.embeddings, "document_ids": retriever.document_ids,
            "evo_cnt": system.evo_cnt, "build_failures": build_failures,
        }
        _save_checkpoint(path, state)
        logger.info("Saved memory checkpoint: %d/%d turns", completed, len(turns))
        return build_failures

    for content, timestamp in turns[completed:]:
        system.add_note(content, time=timestamp)
        completed += 1
        if completed % checkpoint_every == 0:
            snapshot()
    build_failures = snapshot()
    return {"completed": completed, "resumed_from": resumed_from,
            "build_failures": build_failures, "checkpoint": str(path)}


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k, temperature_c5,
                 sglang_host="http://localhost", sglang_port=30000, refresh_index=False):
        self.memory_system = RobustAgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            refresh_index=refresh_index,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        """Select relevant parts of conversation memories — plain text, no JSON schema."""
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

Conversation memories:
{memories_text}

Question: {query}

Return only the relevant parts of the conversation that would help answer this specific question.
If no parts are relevant, return the input unchanged."""

        response = self.retriever_llm.llm.get_completion(prompt)
        return parse_relevant_parts(response)

    def generate_query_llm(self, question):
        """Generate query keywords — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    def answer_question(self, question: str, category: int, answer: str,
                        rng: Optional[random.Random] = None) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        context = raw_context

        assert category in [1, 2, 3, 4, 5]

        if category == 5:
            answer_tmp = list()
            if (rng or random).random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:"""
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
            temperature = 0.7
        elif category == 3:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
            temperature = 0.7

        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt, temperature=temperature,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            FAILURES["answer"] += 1  # scored as an empty prediction; reported at the end
            response = ""
        return response, user_prompt, raw_context


def pool_memories(agents: List[RobustAdvancedMemAgent]) -> RobustAdvancedMemAgent:
    """One agent whose memory is the union of the given agents' memories (--memory_scope pooled).
    Notes and their retriever rows (the same texts and embeddings) are appended conversation by
    conversation, and each note's links are offset to its position in the pooled list."""
    memories, corpus, embeddings = {}, [], []
    for agent in agents:
        notes = list(agent.memory_system.memories.values())
        retriever = agent.memory_system.retriever
        assert len(notes) == len(retriever.corpus) == len(retriever.embeddings), "memory and retriever out of sync"
        offset = len(memories)
        for note in notes:
            note = copy.copy(note)
            note.links = [link + offset for link in note.links if valid_link(link, len(notes))]
            memories[note.id] = note
        corpus.extend(retriever.corpus)
        embeddings.append(retriever.embeddings)
    embeddings = np.vstack(embeddings)
    assert len(memories) == len(corpus) == len(embeddings), "duplicate note ids across conversations"
    pooled = agents[0]
    pooled.memory_system.memories = memories
    pooled.memory_system.retriever.corpus = corpus
    pooled.memory_system.retriever.embeddings = embeddings
    pooled.memory_system.retriever.document_ids = {doc: i for i, doc in enumerate(corpus)}
    return pooled


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     temperature_c5: float = 0.5, retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     legacy_c5_reference: bool = False, seed: int = 42,
                     refresh_index: bool = False, memory_scope: str = "conversation",
                     max_samples: Optional[int] = 1, max_questions: Optional[int] = 10,
                     checkpoint_every: int = 10, max_sessions: Optional[int] = 3):
    """Evaluate the robust agent on the LoComo dataset."""
    evaluation_started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{memory_scope}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST memory layer (no JSON schema dependency)")
    FAILURES.clear()

    samples = load_locomo_dataset(dataset_path, max_samples, max_questions, max_sessions)
    eval_logger.info(f"Loaded {len(samples)} samples")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    if not samples or not any(sample.qa for sample in samples):
        raise ValueError("The selected dataset contains no evaluation questions")
    eval_logger.info("Checking LLM connectivity before building memories")
    RobustLLMController(backend=backend, model=model, sglang_host=sglang_host,
                        sglang_port=sglang_port, check_connection=True)
    # Load metric models/resources only after the API check succeeds.
    from utils import aggregate_metrics, score_prediction

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    i = 0
    error_num = 0
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_resumable",
    )
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1, 2, 3, 4, 5]
    cached_samples, stale_cache_samples = [], []
    memory_builds = {}
    qa_elapsed_seconds = 0.0
    source_digest = hashlib.sha256()
    for name in ("memory_layer.py", "memory_layer_robust.py", "llm_text_parsers.py"):
        source_digest.update((Path(__file__).parent / name).read_bytes())
    memory_configuration = {
        "backend": backend, "model": model, "embedding_model": "all-MiniLM-L6-v2",
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") if backend == "openai"
                    else f"{sglang_host}:{sglang_port}" if backend in ("sglang", "vllm") else "ollama",
        "refresh_index": refresh_index, "memory_version": MEMORY_VERSION,
        "source_sha256": source_digest.hexdigest(),
    }

    def load_or_build(sample_idx, sample):
        """Build the full conversation, or resume its last atomic checkpoint."""
        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5,
                                       sglang_host, sglang_port, refresh_index)

        turns = [("Speaker " + turn.speaker + "says : " + turn.text, session.date_time)
                 for session in sample.conversation.sessions.values() for turn in session.turns]
        identity = memory_identity(turns, memory_configuration)
        checkpoint_path = os.path.join(memories_dir, f"sample_{sample_idx}_{identity}.pkl")
        eval_logger.info("Sample %d requires %d memory notes; questions=%d",
                         sample_idx, len(turns), len(sample.qa))
        build_started = time.perf_counter()
        memory_builds[str(sample_idx)] = build_memory_with_resume(
            agent.memory_system, turns, checkpoint_path, identity,
            checkpoint_every=checkpoint_every, failures=FAILURES, logger=eval_logger,
        )
        memory_builds[str(sample_idx)]["elapsed_seconds"] = time.perf_counter() - build_started
        if memory_builds[str(sample_idx)]["resumed_from"]:
            cached_samples.append(sample_idx)
        if memory_builds[str(sample_idx)]["build_failures"]:
            eval_logger.warning("Memory construction has absorbed failures: %s",
                                memory_builds[str(sample_idx)]["build_failures"])
        return agent

    if memory_scope == "pooled":
        # every evaluated conversation's memory in one store: retrieval must also reject the
        # other conversations' notes (issue #27); built from the same per-conversation caches
        pooled_agent = pool_memories([load_or_build(i, s) for i, s in enumerate(samples)])
        eval_logger.info(f"Pooled memory: {len(pooled_agent.memory_system.memories)} notes "
                         f"from {len(samples)} conversations")

    for sample_idx, sample in enumerate(samples):
        agent = pooled_agent if memory_scope == "pooled" else load_or_build(sample_idx, sample)

        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")

        for qa_idx, qa in enumerate(sample.qa):
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1

                # the category-5 option order comes from the question's own generator
                rng = random.Random(f"{seed}:{sample_idx}:{qa_idx}")
                answer_started = time.perf_counter()
                prediction, user_prompt, raw_context = agent.answer_question(
                    qa.question, qa.category, qa.final_answer, rng
                )
                answer_elapsed_seconds = time.perf_counter() - answer_started
                qa_elapsed_seconds += answer_elapsed_seconds

                # Parse the prediction (handles both JSON and plain text)
                prediction = parse_plain_text_answer(prediction)

                reference, metrics = score_prediction(prediction, qa, legacy_c5_reference)

                eval_logger.info(f"Question {total_questions}: {qa.question}")
                eval_logger.info(f"Prediction: {prediction}")
                eval_logger.info(f"Reference: {reference}")
                eval_logger.info(f"User Prompt: {user_prompt}")
                eval_logger.info(f"Category: {qa.category}")
                eval_logger.info(f"Raw Context: {raw_context}")

                all_metrics.append(metrics)
                all_categories.append(qa.category)

                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": reference,
                    "category": qa.category,
                    "adversarial_answer": qa.adversarial_answer,
                    "metrics": metrics,
                    "answer_elapsed_seconds": answer_elapsed_seconds,
                }
                results.append(result)

                if total_questions % 10 == 0:
                    eval_logger.info(f"Processed {total_questions} questions")

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": model,
        "dataset": dataset_path,
        "memory_layer": "robust",
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "ratio": ratio,
        "backend": backend,
        "max_samples": max_samples,
        "max_questions": max_questions,
        "max_sessions": max_sessions,
        "dataset_source_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "subset_selection": {sample.sample_id: {
            "session_ids": list(sample.conversation.sessions),
            "source_qa_indices": sample.source_qa_indices,
            "retained_turns": sum(len(session.turns) for session in sample.conversation.sessions.values()),
        } for sample in samples},
        "retrieve_k": retrieve_k,
        "memory_configuration": memory_configuration,
        "decoding": {"temperature": 0.7, "temperature_c5": temperature_c5,
                     "max_tokens": 1000 if backend != "ollama" else "provider default",
                     "top_p": "provider default"},
        "timing": {"total_elapsed_seconds": time.perf_counter() - evaluation_started,
                   "qa_answer_elapsed_seconds": qa_elapsed_seconds},
        "checkpoint_every": checkpoint_every,
        "memory_builds": memory_builds,
        "seed": seed,
        "legacy_c5_reference": legacy_c5_reference,
        "refresh_index": refresh_index,
        "memory_scope": memory_scope,
        "pooled_notes": len(pooled_agent.memory_system.memories) if memory_scope == "pooled" else None,
        "cached_samples": cached_samples,
        "stale_cache_samples": stale_cache_samples,
        # failures absorbed in this run (memories loaded from the cache were built earlier);
        # "answer" counts questions scored as empty predictions because the call failed
        "llm_failures": dict(FAILURES),
        "individual_results": results,
    }
    eval_logger.info(f"Error number: {error_num}")
    eval_logger.info(f"LLM failures absorbed in this run: {dict(FAILURES) or 'none'}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--max_samples", type=int, default=1,
                        help="Maximum conversations (default: 1; 0: all)")
    parser.add_argument("--max_questions", type=int, default=10,
                        help="Maximum questions per conversation (default: 10; 0: all)")
    parser.add_argument("--max_sessions", type=int, default=3,
                        help="Keep earliest N sessions and QAs with contained evidence (0: all)")
    parser.add_argument("--checkpoint_every", type=int, default=10,
                        help="Save memory construction every N completed turns")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, or vllm)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                        help="Temperature for category 5 questions")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--legacy_c5_reference", action="store_true",
                        help="Score category 5 against the adversarial answer, as the paper's evaluation did "
                             "(default: against 'Not mentioned in the conversation', the correct response)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the order of the two category-5 answer options")
    parser.add_argument("--refresh_index", action="store_true",
                        help="Re-embed neighbour memories as soon as evolution updates them (default: their "
                             "embeddings refresh only every evo_threshold evolutions, as in the paper). Builds "
                             "its own memory cache")
    parser.add_argument("--memory_scope", choices=["conversation", "pooled"], default="conversation",
                        help="conversation: each question is answered from its own conversation's memory "
                             "(LoCoMo's protocol, as in the paper). pooled: from one memory holding all "
                             "evaluated conversations, so retrieval must also reject other conversations' "
                             "notes (uses the same memory caches)")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    if (args.max_samples < 0 or args.max_questions < 0 or args.max_sessions < 0
            or args.checkpoint_every <= 0):
        raise ValueError("Sample/question limits must be nonnegative and checkpoint_every positive")
    if args.retrieve_k <= 0:
        raise ValueError("retrieve_k must be positive")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    evaluate_dataset(
        dataset_path, args.model, output_path, args.ratio,
        args.backend, args.temperature_c5, args.retrieve_k,
        args.sglang_host, args.sglang_port, args.legacy_c5_reference, args.seed,
        args.refresh_index, args.memory_scope,
        args.max_samples or None, args.max_questions or None, args.checkpoint_every,
        args.max_sessions or None,
    )


if __name__ == "__main__":
    main()
