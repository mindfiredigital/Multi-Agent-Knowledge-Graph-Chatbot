"""
Multi-Agent Retrieval System with Blackboard Architecture (packaged under knowledge_graph.v1).
"""

import os
import ast
import time
import threading
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

from ..config.prompt_manager import prompt_manager
# import ollama
import openai
from typing import List, Dict, Any, Optional, Tuple
from ..utils.logger_config import get_retrieval_logger
from ..config.config import config

# Add LangChain imports for Bedrock integration
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

# Load environment variables from .env file
load_dotenv()

# Lazy initialization of OpenAI client
_client = None

def get_openai_client():
    """Get or create OpenAI client with lazy initialization."""
    global _client
    if _client is None:
        try:
            _client = OpenAI()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    return _client

# For backward compatibility, provide lazy client access
def client():
    return get_openai_client()
# ====================================
# CONFIGURATION SETTINGS
# ====================================

RERANKING_CONFIG = {
    "ignore_formatting": True,
    "preprocess_content": True,
    "enhance_query": True,
}

# ====================================
# LOGGING CONFIGURATION
# ====================================

logger = get_retrieval_logger()
logger.propagate = False

# ====================================
# EMBEDDING MODEL INITIALIZATION
# ====================================

model = SentenceTransformer(config.llm.embedding_model_name, device='cpu')

# ====================================
# BEDROCK RERANKING INTEGRATION
# ====================================

import boto3


def _make_bedrock_reranker(model_name: str, temperature: float = config.amazon_bedrock.temperature):
    import base64
    from botocore.config import Config  # noqa: F401

    aws_bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

    if aws_bearer_token:
        try:
            if aws_bearer_token.startswith('"') and aws_bearer_token.endswith('"'):
                aws_bearer_token = aws_bearer_token.strip('"')

            decoded_bytes = base64.b64decode(aws_bearer_token)
            try:
                decoded_token = decoded_bytes.decode('utf-8')
            except UnicodeDecodeError:
                decoded_token = decoded_bytes.decode('latin-1')
                start_idx = 0
                for i, char in enumerate(decoded_token):
                    if char.isalnum() or char in [':', '-', '_', '+', '/', '=']:
                        start_idx = i
                        break
                decoded_token = decoded_token[start_idx:]

            if ':' in decoded_token:
                access_key, secret_key = decoded_token.split(':', 1)
                access_key = access_key.strip()
                secret_key = secret_key.strip()
                logger.debug(f"Successfully parsed AWS Bearer Token for Bedrock reranker")
                client = boto3.client(
                    'bedrock-runtime',
                    region_name=config.amazon_bedrock.region_name,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key
                )
                return client
        except Exception as e:
            logger.warning(f"Failed to parse AWS Bearer Token: {e}")

    return boto3.client('bedrock-runtime', region_name=config.amazon_bedrock.region_name)


class BedrockReranker:
    def __init__(self):
        try:
            self.bedrock_client = _make_bedrock_reranker(config.amazon_bedrock.reranker_model_name)
            self.model_id = config.amazon_bedrock.reranker_model_name
            logger.debug("Bedrock Amazon Rerank model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock Amazon rerank: {e}")
            raise e

    def predict(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        try:
            if not query_doc_pairs:
                return []
            query = query_doc_pairs[0][0]
            documents = [pair[1] for pair in query_doc_pairs]
            import json
            request_body = {
                "documents": documents,
                "query": query,
                "top_n": len(documents)
            }
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json'
            )
            response_body = json.loads(response['body'].read())
            scores = []
            if 'results' in response_body:
                score_map = {}
                for result in response_body['results']:
                    index = result.get('index', 0)
                    relevance_score = result.get('relevance_score', 0.0)
                    score_map[index] = relevance_score
                for i in range(len(documents)):
                    scores.append(score_map.get(i, 0.0))
            else:
                scores = [0.0] * len(documents)
            logger.debug(f"Bedrock Amazon reranking completed for {len(documents)} documents")
            return scores
        except Exception as e:
            logger.error(f"Error in Bedrock Amazon reranking: {e}")
            return [0.0] * len(query_doc_pairs)


# Global variable to hold the reranking model - will be loaded lazily
reranking_model = None

def _load_reranking_model():
    """Load the reranking model lazily when first needed."""
    global reranking_model
    if reranking_model is not None:
        return

    try:
        logger.debug("Loading Amazon Reranker v1 model via AWS Bedrock...")
        reranking_model = BedrockReranker()
        logger.debug("Amazon Reranker v1 model loaded successfully via Bedrock")
    except Exception as e:
        logger.error(f"Failed to load Amazon Reranker v1: {e}")
        logger.info("Falling back to Qwen3 reranker")
        try:
            reranking_model = CrossEncoder(config.llm.reranker_model_name)
            if reranking_model.tokenizer.pad_token is None:
                reranking_model.tokenizer.pad_token = reranking_model.tokenizer.eos_token
                logger.debug("Set pad_token to eos_token for Qwen3 reranker")
            logger.debug("Qwen3 reranking model loaded successfully on CPU")
        except Exception as fallback_error:
            logger.error(f"Failed to load Qwen3 reranker: {fallback_error}")
            logger.info("Falling back to MS-Marco reranker")
            reranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
            if reranking_model.tokenizer.pad_token is None:
                reranking_model.tokenizer.pad_token = reranking_model.tokenizer.eos_token


class Blackboard:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
        self.query_posted = False
        self.partial_results = []
        self.final_answer = None
        self.active_agents = 0
        self.processed_by_agents = set()

    def post_query(self, query: str, query_embedding: List[float]) -> None:
        with self.lock:
            self.data["query"] = query
            self.data["query_embedding"] = query_embedding
            self.data["status"] = "query_posted"
            self.query_posted = True
        logger.debug(f"Query posted to blackboard: {query}")

    def read_query(self) -> Tuple[Optional[str], Optional[List[float]]]:
        with self.lock:
            return self.data.get("query", None), self.data.get("query_embedding", None)

    def post_partial_result(self, agent_id: str, result: Dict[str, Any]) -> None:
        with self.lock:
            self.partial_results.append({
                "agent_id": agent_id,
                "result": result
            })

    def read_partial_results(self) -> List[Dict[str, Any]]:
        with self.lock:
            return self.partial_results.copy()

    def post_final_answer(self, answer: str) -> None:
        with self.lock:
            self.final_answer = answer
            self.data["status"] = "completed"
            logger.debug(f"Final answer posted to blackboard: {answer}")


class BlackboardAgent(threading.Thread):
    def __init__(self, driver, agent_id: str, domain_name: str, root_node: str, project_name: str, blackboard: Blackboard):
        super().__init__()
        self.driver = driver
        self.agent_id = agent_id
        self.domain_name = domain_name
        self.root_node = root_node
        self.project_name = project_name
        self.blackboard = blackboard
        self.running = True

    def run(self) -> None:
        while self.running:
            try:
                if self.blackboard.query_posted and self.agent_id not in self.blackboard.processed_by_agents:
                    self.blackboard.processed_by_agents.add(self.agent_id)
                    if self.should_activate():
                        self.blackboard.active_agents += 1
                        self.process_query()
                    else:
                        self.blackboard.active_agents += 0  # No activation
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id}: {e}")

    def stop(self) -> None:
        self.running = False
        logger.debug(f"Agent {self.agent_id} stopped.")

    def should_activate(self) -> bool:
        query, query_embedding = self.blackboard.read_query()
        if not query or not query_embedding:
            return False
        confidence = self.calculate_confidence(query_embedding, project_name=self.project_name)
        should_work = confidence > config.search.confidence_threshold
        return should_work

    def calculate_confidence(self, query_embedding: List[float], project_name: str) -> float:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:Entity {name:$root_node, project_name:$project_name})
                WHERE n.embedding IS NOT NULL
                RETURN n.embedding AS embedding
                """,
                root_node=self.root_node, project_name=project_name
            ).single()

            if result and result["embedding"]:
                root_embedding = result["embedding"]
                dot_product = sum(a * b for a, b in zip(query_embedding, root_embedding))
                norm_a = sum(a * a for a in query_embedding) ** 0.5
                norm_b = sum(b * b for b in root_embedding) ** 0.5
                if norm_a > 0 and norm_b > 0:
                    similarity = dot_product / (norm_a * norm_b)
                    return max(0, similarity)
            return 0.0

    def process_query(self) -> None:
        query, query_embedding = self.blackboard.read_query()
        logger.debug(f"Agent {self.agent_id} is searching the domain {self.domain_name}")
        if not query_embedding:
            logger.warning(f"Agent {self.agent_id} has no query embedding to process")
            return
        domain_results = self.search_domain_only(query_embedding)
        if domain_results:
            self.blackboard.post_partial_result(self.agent_id, {
                "domain_name": self.domain_name,
                "nodes": domain_results,
                "confidence": self.calculate_confidence(query_embedding, project_name=self.project_name)
            })

    def search_domain_only(self, query_embedding: List[float], k: int = config.search.domain_search_k, threshold: float = config.search.domain_threshold) -> List[Dict[str, Any]]:
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n:Entity {name:$root_node, project_name:$project_name})-[*0..5]->(node:Entity)
                    WHERE node.embedding IS NOT NULL
                    WITH node, 
                        reduce(dot = 0.0, i in range(0, size(node.embedding)-1) | 
                               dot + node.embedding[i] * $query_embedding[i]) as dot_product,
                        sqrt(reduce(norm_a = 0.0, i in range(0, size(node.embedding)-1) | 
                                   norm_a + node.embedding[i] * node.embedding[i])) as norm_a,
                        sqrt(reduce(norm_b = 0.0, i in range(0, size($query_embedding)-1) | 
                                   norm_b + $query_embedding[i] * $query_embedding[i])) as norm_b
                    WITH node, dot_product / (norm_a * norm_b) AS similarity
                    WHERE similarity >= $threshold
                    RETURN node.name AS name, node.content AS content, node.type AS type,
                        node.project_name AS project_name, similarity
                    ORDER BY similarity DESC
                    """,
                    project_name=self.project_name, query_embedding=query_embedding,
                    root_node=self.root_node, threshold=threshold
                )

                records = list(result)
                domain_results = [{
                    "name": record["name"],
                    "content": record["content"],
                    "project_name": record["project_name"],
                    "type": record["type"],
                    "similarity": record["similarity"]
                } for record in records]

                return domain_results[:k]
        except Exception as e:
            logger.error(f"Error occurred while searching domain '{self.domain_name}': {e}")
            return []


class BlackboardController:
    def __init__(self, driver, blackboard: Blackboard):
        self.driver = driver
        self.blackboard = blackboard

    def wait_for_responses(self, timeout: int) -> List[Dict[str, Any]]:
        start_time = time.time()
        logger.debug(f"Waiting for responses. Number of active agents: {self.blackboard.active_agents}")
        while time.time() - start_time < timeout:
            partial_results = self.blackboard.read_partial_results()
            if len(partial_results) >= self.blackboard.active_agents and self.blackboard.active_agents > 0:
                logger.debug(f"All responses received. Number of active agents: {self.blackboard.active_agents}")
                return partial_results
            time.sleep(0.01)
        if self.blackboard.active_agents == 0:
            logger.debug("No active agents")
        else:
            logger.warning(f"Timeout occurred waiting for agent responses after {timeout} seconds")
        return self.blackboard.read_partial_results()

    def synthesize_results(self, partial_results: List[Dict[str, Any]], project_name: str, query: str) -> Dict[str, Any]:
        try:
            from ..services.evaluation import RAGASEvaluator
            evaluator = RAGASEvaluator(use_local_models=True, timeout_seconds=600)
            logger.debug("RAGAS evaluator initialized successfully with AWS Bedrock and extended timeout")
        except Exception as e:
            logger.warning(f"Could not initialize RAGAS evaluator: {e}")
            evaluator = None

        response = {"message": ""}
        if not partial_results:
            response["message"] = "No partial results found. So no related information regarding it."
            response["success"] = False
            return response

        start_time = time.time()
        all_content_list = get_all_content(self.driver, partial_results, project_name, query)
        all_content = "\n\n".join(all_content_list)
        end_time = time.time()

        if evaluator is not None:
            logger.debug("Starting RAGAS individual context evaluation with dummy answer")
            answer = "This evaluation focuses on context-query relevance only."
            individual_results = evaluator.evaluate_individual_context(
                query, all_content_list, answer
            )
            logger.debug(f"Individual context relevance scores: {individual_results}")
            ordered_contexts = [item['context_full'] for item in sorted(
                individual_results,
                key=lambda x: x['context_relevancy'], reverse=True
            )]
            reordered_content = "\n\n".join(ordered_contexts)
        else:
            logger.warning("RAGAS evaluator not available, skipping context relevance evaluation")
            reordered_content = all_content

        logger.debug(f"Content retrieval time: {end_time - start_time:.2f} seconds")

        start_time = time.time()
        final_answer = response_from_llm(query, reordered_content)
        end_time = time.time()
        logger.debug(f"LLM response time: {end_time - start_time:.2f} seconds")

        if "error" in final_answer:
            response["error"] = final_answer["error"]
            response["success"] = final_answer["success"]
        else:
            response["message"] = final_answer["answer"]
            response["success"] = final_answer["success"]
            if evaluator is not None:
                try:
                    logger.debug("Starting RAGAS overall evaluation...")
                    eval_start = time.time()
                    final_score = evaluator.evaluate_single_query(
                        query, all_content_list, final_answer["answer"]
                    )
                    eval_time = time.time() - eval_start
                    logger.debug(f"Final score from RAGAS: {final_score}")
                    logger.debug(f"RAGAS overall evaluation completed in {eval_time:.2f} seconds")
                    response["evaluation"] = {
                        "individual_contexts": individual_results,
                        "overall_score": final_score,
                        "evaluation_time": eval_time
                    }
                except Exception as e:
                    logger.error(f"RAGAS evaluation failed: {e}")
                    response["evaluation_error"] = str(e)
        return response


def find_children_nodes(driver, parent_name: str, project_name: str) -> List[Dict[str, Any]]:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Entity {name: $parent_name, project_name:$project_name})-[:RELATION]->(child)
            RETURN child.name AS name, child.content AS content
            """,
            parent_name=parent_name, project_name=project_name
        )
        records = list(result)
        return [{
            "name": record["name"],
            "content": record["content"]
        } for record in records]


def find_parent_node(driver, child_name, project_name):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (child:Entity {name: $child_name, project_name:$project_name})<-[:RELATION]-(parent:Entity)
            RETURN parent.name AS name, parent.content AS content
            """,
            child_name=child_name, project_name=project_name
        )
        records = list(result)
        return [{
            "name": record["name"],
            "content": record["content"]
        } for record in records]


def preprocess_content_for_reranking(content: str) -> str:
    if not content:
        return content
    cleaned = ' '.join(content.split())
    cleaned = cleaned.replace(' .', '.').replace(' ,', ',').replace(' ;', ';')
    cleaned = cleaned.replace('- ', '').replace('• ', '')
    import re
    cleaned = re.sub(r'\s+\.\s+([A-Z])', r'. \1', cleaned)
    cleaned = re.sub(r"(\w+)\s+'(\w+)", r"\1'\2", cleaned)
    cleaned = re.sub(r'(\w+)\s+\.', r'\1.', cleaned)
    cleaned = re.sub(r'\b[A-Z]\s+(?=[A-Z][a-z])', '', cleaned)
    return cleaned.strip()


def reranking_content(query: str, all_nodes: List[Dict[str, Any]], topk: int, ignore_formatting: bool = True) -> List[dict]:
    if not all_nodes:
        return []
    if len(all_nodes) == 1:
        logger.debug("Only one node exists, skipping reranking")
        return all_nodes
    start_time = time.time()
    logger.debug(f"Starting reranking for {len(all_nodes)} nodes (ignore_formatting={ignore_formatting})")
    try:
        query_doc_pairs = []
        for node in all_nodes:
            content = node['content'] if node['content'] else 'No content available'
            if ignore_formatting and RERANKING_CONFIG["preprocess_content"]:
                cleaned_content = preprocess_content_for_reranking(content)
                doc_text = f"Topic: {node['name']}\nContent: {cleaned_content}"
            else:
                doc_text = f"Topic: {node['name']}\nContent: {content}"
            query_doc_pairs.append((query, doc_text))
        rerank_start = time.time()
        # Ensure the reranking model is loaded
        _load_reranking_model()

        scores = []
        for query_doc_pair in query_doc_pairs:
            try:
                score = reranking_model.predict([query_doc_pair])
                scores.append(score[0])
            except Exception as pair_error:
                logger.warning(f"Error processing individual pair: {pair_error}")
                scores.append(0.0)
        logger.debug(f"Reranking scores computed: {scores}")
        rerank_time = time.time() - rerank_start
        logger.debug(f"Reranking model prediction took {rerank_time:.2f} seconds")
        scored_nodes = []
        for i, (node, score) in enumerate(zip(all_nodes, scores)):
            scored_nodes.append({
                "name": node["name"],
                "content": node["content"],
                "rerank_score": float(score),
                "original_position": i
            })
        reranked_nodes = sorted(scored_nodes, key=lambda x: x["rerank_score"], reverse=True)
        if topk:
            reranked_nodes = reranked_nodes[:topk]
            logger.debug(f"Returning top {topk} reranked nodes")
        total_time = time.time() - start_time
        logger.debug(f"Reranking completed in {total_time:.2f} seconds")
        logger.debug(f"Reranked nodes: {[(node['name'],node['original_position']) for node in reranked_nodes]}")
        return reranked_nodes
    except Exception as e:
        logger.error(f"Error occurred during reranking: {e}")
        logger.info("Falling back to original order")
        return all_nodes


def get_all_content(driver, partial_results: List[Dict[str, Any]], project_name: str, query: str) -> List[dict]:
    all_content = []
    all_unique_nodes = set()
    all_nodes = []
    for agent_result in partial_results:
        result = agent_result["result"]
        nodes = result["nodes"]
        for node in nodes:
            children = find_children_nodes(driver, node["name"], project_name)
            parents = find_parent_node(driver, node["name"], project_name)
            if node["name"] not in all_unique_nodes:
                all_unique_nodes.add(node["name"])
                all_nodes.append(node)
            for child in children:
                if child["name"] not in all_unique_nodes:
                    all_unique_nodes.add(child["name"])
                    all_nodes.append(child)
            for parent in parents:
                if parent["name"] not in all_unique_nodes:
                    all_unique_nodes.add(parent["name"])
                    all_nodes.append(parent)
    reranked_nodes = reranking_content(
        query,
        all_nodes,
        topk=config.llm.rerank_top_k,
        ignore_formatting=RERANKING_CONFIG["ignore_formatting"]
    )
    for node in reranked_nodes:
        all_content.append(f"Topic: {node['name']}\nContent: {node['content'] if node['content'] else 'No content available'}")
    logger.debug(f"All nodes being sent to llm: {all_unique_nodes}")
    content_preview = "\n\n".join(all_content)[:500]
    logger.debug(f"Content supplied to LLM: {content_preview}...")
    return all_content


def response_from_llm(query: str, content: str) -> Dict[str, Any]:
    start_time = time.time()
    logger.debug(f"Starting LLM response generation for query: {query[:50]}...")
    logger.debug(f"Content length: {len(content)} characters")

    # update the call to use OpenAI client
    response = client().chat.completions.create(  # Replace ollama.chat
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": prompt_manager.get_retrieval_response_system_prompt()},
            {"role": "user", "content": prompt_manager.get_retrieval_response_user_prompt(query=query, content=content)},
        ],
        temperature=config.search.retrieval_temperature
    )
    llm_call_time = time.time() - start_time
    logger.debug(f"LLM call completed in {llm_call_time:.2f} seconds")
    logger.debug("Changed temperature to 0.5")
    # OpenAI SDK returns an object; extract the first choice message content
    try:
        content_message = response.choices[0].message.content if getattr(response, "choices", None) else ""
    except Exception:
        content_message = ""
    llm_response = (content_message or "").strip()
    logger.debug(f"Raw response from llm: {repr(llm_response)}")
    try:
        response_dict = ast.literal_eval(llm_response)
        return response_dict if response_dict else {"success": False, "answer": "No response generated"}
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse LLM response as Python dictionary: {e}")
        import re
        answer_match = re.search(r'"answer":\s*"([^"]*(?:\\.[^"]*)*)', llm_response, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            logger.debug("Successfully extracted answer from malformed Python dict")
            success_match = re.search(r'"success":\s*(True|False)', llm_response)
            success_val = True if success_match and success_match.group(1) == "True" else False
            return {"success": success_val, "answer": answer_content}
        else:
            logger.warning("No answer field found in response")
            return {"success": False, "answer": f"Response parsing error, raw content: {llm_response}"}


def embed_query(query: str) -> List[float]:
    return model.encode(query, convert_to_tensor=True).tolist()


def discover_root_nodes(driver, project_name: str) -> List[Dict[str, str]]:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {name: $name})-[:CONTAINS]->(child:Entity)
            RETURN child.name AS name
            """,
            name=project_name
        )
        root_candidates = []
        records = list(result)
        for record in records:
            root_candidates.append({
                "name": record["name"]
            })
        logger.debug(f"Root candidates found for project '{project_name}': {root_candidates}")
        return root_candidates if root_candidates else []


class TrueBlackboardSystem:
    def __init__(self, driver, project_name):
        self.driver = driver
        self.project_name = project_name
        self.blackboard = Blackboard()
        self.agents = []
        self.controller = BlackboardController(self.driver, self.blackboard)
        self.discover_and_create_agents()

    def discover_and_create_agents(self):
        root_nodes = discover_root_nodes(self.driver, self.project_name)
        if not root_nodes:
            logger.warning("No root nodes found. Cannot create agents.")
            return
        for root in root_nodes:
            agent = BlackboardAgent(
                driver=self.driver,
                agent_id=f"{root['name']}",
                domain_name=root["name"],
                root_node=root["name"],
                project_name=self.project_name,
                blackboard=self.blackboard
            )
            self.agents.append(agent)
            agent.start()
            logger.debug(f"Started autonomous agent: {agent.agent_id}")

    def process_query(self, query):
        response = {}
        query_embedding = embed_query(query)
        self.blackboard.post_query(query, query_embedding)
        partial_results = self.controller.wait_for_responses(timeout=config.search.activation_timeout)
        if self.blackboard.active_agents == 0:
            logger.debug(f"No active agents remaining.")
            return {"success": False, "message": "No agents are experts in this topic."}
        final_answer = self.controller.synthesize_results(partial_results, self.project_name, query)
        self.blackboard.post_final_answer(final_answer["message"])
        return final_answer

    def shutdown(self):
        logger.info("Shutting down TrueBlackboardSystem...")
        for agent in self.agents:
            agent.stop()
            agent.join()
        logger.info("All agents stopped. System shutdown complete.")


