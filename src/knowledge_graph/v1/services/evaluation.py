"""
RAGAS Evaluation Module (packaged under knowledge_graph.v1).
"""

import time
from ..utils.logger_config import get_retrieval_logger
from ..config.config import config

# Direct imports since we're using asyncio loop
from ragas import evaluate
from ragas.metrics import ContextRelevance, answer_relevancy, faithfulness
from datasets import Dataset
from ragas.run_config import RunConfig

# Bedrock imports for AWS integration
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# OpenAI imports for fallback
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = get_retrieval_logger()
logger.propagate = False


CUSTOM_CONTEXT_RELEVANCE_TEMPLATE = """You are an expert relevance evaluator. Your task is to score the relevance of the Context to the Question on a scale of 0 to 2.
Use only the information provided in the Context and Question.

**SCORING GUIDE:**
- **2 (Directly Answers):** The context fully and directly answers the question.
- **1.5 (Mostly Relevant):** Good explanation, but missing some details.
- **1 (Partially Relevant):** Contains related information (like training methods, background) but doesn't fully answer the core question.
- **0.5 (Tangentially Relevant):** Mentions the core concept but doesn't explain it.
- **0 (Irrelevant):** Does NOT mention the core concept from the question at all.

**CRITICAL SCORING RULES:**
- **Tangential content (history, consequences, impacts) = 0.5.** It mentions the topic but doesn't define it.
- **Related information (background, training methods) = 1.0.** It provides useful but incomplete information.
- **If the core concept is NOT mentioned = 0.**

You must provide the relevance score of 0, 0.5, 1, 1.5, or 2, nothing else.
Do not explain.

### Question: {query}

### Context: {context}

Do not try to explain.
Analyzing Context and Question, the Relevance score is """


def _make_bedrock(model_name: str, temperature: float = 0.0):
    return ChatBedrockConverse(
        model_id=model_name,  # type: ignore
        temperature=temperature,
        region_name=config.amazon_bedrock.region_name,
    )  # type: ignore


def _make_bedrock_embeddings(model_name: str):
    return BedrockEmbeddings(
        model_id=model_name,
        region_name=config.amazon_bedrock.region_name,
    )


class RAGASEvaluator:
    def __init__(self, use_local_models=False, timeout_seconds=600):
        self.evaluate = evaluate
        self.Dataset = Dataset
        self.run_config = RunConfig(timeout=timeout_seconds)
        logger.debug(f"RAGAS timeout set to {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")

        if use_local_models:
            logger.debug("Initializing RAGAS with AWS Bedrock models...")
            try:
                bedrock_llm = _make_bedrock(config.amazon_bedrock.llm_model_name, temperature=config.amazon_bedrock.temperature)
                bedrock_embeddings = _make_bedrock_embeddings("amazon.titan-embed-text-v2:0")
                self.llm = LangchainLLMWrapper(bedrock_llm)
                self.embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)
                logger.debug("Using AWS Bedrock Titan-Embed-Text-V2 for embeddings")

                from ragas.metrics import AnswerRelevancy, Faithfulness
                custom_context_relevance = ContextRelevance(llm=self.llm)
                custom_context_relevance.template_relevance1 = CUSTOM_CONTEXT_RELEVANCE_TEMPLATE
                custom_answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
                custom_faithfulness = Faithfulness(llm=self.llm)
                self.metrics = [
                    custom_context_relevance,
                    custom_answer_relevancy,
                    custom_faithfulness,
                ]
                logger.debug("AWS Bedrock GPT-OSS 20B models configured successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock models: {e}")
                logger.info("Falling back to default metrics (will require OpenAI API key)")
                self.metrics = [ContextRelevance(), answer_relevancy, faithfulness]
        else:
            logger.debug("Using OpenAI GPT-4 models...")
            try:
                openai_llm = ChatOpenAI(model=config.llm.model_name, temperature=config.llm.eval_temperature)
                openai_embeddings = OpenAIEmbeddings(model=config.llm.openai_embedding_model_name)
                self.llm = LangchainLLMWrapper(openai_llm)
                self.embeddings = LangchainEmbeddingsWrapper(openai_embeddings)
                from ragas.metrics import AnswerRelevancy, Faithfulness
                default_context_relevance = ContextRelevance(llm=self.llm)
                custom_answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
                custom_faithfulness = Faithfulness(llm=self.llm)
                self.metrics = [
                    default_context_relevance,
                    custom_answer_relevancy,
                    custom_faithfulness,
                ]
                logger.debug("OpenAI GPT-4 configured with DEFAULT RAGAS prompt (baseline comparison)")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI GPT-4: {e}")
                logger.info("Falling back to default RAGAS metrics")
                self.metrics = [ContextRelevance(), answer_relevancy, faithfulness]

    def safe_faithfulness_evaluation(self, query, retrieved_contexts, generated_answer):
        try:
            logger.debug("Attempting safe faithfulness evaluation...")
            faithfulness_llm = _make_bedrock(config.amazon_bedrock.llm_model_name)
            faithfulness_llm.temperature = config.llm.eval_temperature
            faithfulness_llm_wrapper = LangchainLLMWrapper(faithfulness_llm)
            from ragas.metrics import Faithfulness
            safe_faithfulness = Faithfulness(llm=faithfulness_llm_wrapper)
            eval_data = {
                "question": [query],
                "contexts": [retrieved_contexts],
                "answer": [generated_answer],
            }
            dataset = self.Dataset.from_dict(eval_data)
            result = self.evaluate(dataset, metrics=[safe_faithfulness], run_config=self.run_config)
            faithfulness_score = result.scores[0].get('faithfulness', None)
            logger.debug(f"Safe faithfulness evaluation completed: {faithfulness_score}")
            return faithfulness_score
        except Exception as e:
            logger.error(f"Safe faithfulness evaluation failed: {e}")
            logger.info("Faithfulness evaluation skipped due to parsing issues")
            return None

    def prepare_evaluation_data(self, query, retrieved_contexts, generated_answer, ground_truth=None):
        eval_data = {
            "question": [query],
            "contexts": [retrieved_contexts],
            "answer": [generated_answer],
        }
        if ground_truth:
            eval_data["ground_truth"] = [ground_truth]
        return eval_data

    def evaluate_individual_context(self, query, retrieved_contexts, generated_answer):
        individual_results = []
        try:

            logger.debug(f"Evaluating {len(retrieved_contexts)} contexts individually")

            for i, context in enumerate(retrieved_contexts):
                start_time = time.time()
                single_context_data = {
                    "question": [query],
                    "contexts": [[context]],
                    "answer": [generated_answer],
                }
                dataset = self.Dataset.from_dict(single_context_data)
                context_metrics = [self.metrics[0]]
                result = self.evaluate(dataset, metrics=context_metrics, run_config=self.run_config)
                eval_time = time.time() - start_time
                context_relevancy = result.scores[0]['nv_context_relevance']
                single_result = {
                    'context_index': i,
                    'context': context[:100] + "..." if len(context) > 100 else context,
                    'context_full': context,
                    'context_relevancy': context_relevancy,
                    'evaluation_time': eval_time,
                }
                individual_results.append(single_result)

            return individual_results
        except Exception as e:
            logger.error(f"Error in individual context evaluation: {e}")
            return []

    def evaluate_single_query(self, query, retrieved_contexts, generated_answer, ground_truth=None):
        try:
            start_time = time.time()
            if not query or not retrieved_contexts or not generated_answer:
                logger.error("Missing required inputs for evaluation")
                return {}
            if not isinstance(retrieved_contexts, list):
                logger.warning("Retrieved contexts should be a list. Converting to list.")
                retrieved_contexts = [retrieved_contexts]
            eval_data = self.prepare_evaluation_data(query, retrieved_contexts, generated_answer, ground_truth)
            dataset = self.Dataset.from_dict(eval_data)
            logger.debug(f"Query: {query}")
            logger.debug(f"Number of Retrieved Contexts: {len(retrieved_contexts)}")
            logger.debug(f"Answer length: {len(generated_answer)} characters")
            logger.debug(f"Metrics to evaluate: {[metric.name for metric in self.metrics]}")

            result = self.evaluate(dataset, metrics=self.metrics, run_config=self.run_config)
            eval_time = time.time() - start_time
            scores = result.scores[0]
            logger.debug(f"Result from RAGAS directly: {scores}")
            clean_result = {}
            for metric_name, score in scores.items():
                if score != score:  # NaN
                    logger.warning(f"{metric_name} returned NaN - setting to 0.0")
                    clean_result[metric_name] = 0.0
                else:
                    clean_result[metric_name] = float(score)
            logger.debug("RAGAS EVALUATION RESULTS:")
            for metric, score in clean_result.items():
                logger.debug(f"{metric}: {score:.4f}")
            logger.debug(f"Evaluation completed in {eval_time:.2f} seconds")

            return clean_result
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Contexts length: {len(retrieved_contexts) if retrieved_contexts else 0}")
            logger.error(f"Answer length: {len(generated_answer) if generated_answer else 0}")
            return {}


