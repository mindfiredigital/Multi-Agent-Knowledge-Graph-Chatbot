"""
Knowledge Graph Ingestion (packaged under knowledge_graph.v1).
"""

import ast
import os
import time

from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from ..utils.logger_config import get_ingestion_logger
from ..config.config import config
from ..config.prompt_manager import prompt_manager

load_dotenv()

logger = get_ingestion_logger()

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

# Initialize model lazily too
_model = None

def get_embedding_model():
    """Get or create embedding model with lazy initialization."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(config.llm.embedding_model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {e}")
    return _model

# For backward compatibility - lazy initialization
def get_model():
    """Get the embedding model (backward compatibility function)."""
    return get_embedding_model()

# Keep the old name for backward compatibility but make it lazy
model = None


def merge_content_intelligently(existing_content, new_content, entity_name):
    logger.debug(f"Original: {existing_content}")
    logger.debug(f"Merging for {entity_name}:")
    logger.debug(f"  Existing: {existing_content}")
    logger.debug(f"  New: {new_content}")
    try:
        response = client().chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": prompt_manager.get_merge_system_prompt()},
                {"role": "user", "content": prompt_manager.get_merge_user_prompt(entity_name, existing_content, new_content)},
            ],
            temperature=config.llm.temperature
        )
        merged_content = response.choices[0].message.content.strip()
        new_info_found = existing_content != merged_content
        logger.debug(f"Merge result for {entity_name}: new_info={new_info_found}")
        return merged_content, new_info_found
    except Exception as e:
        logger.error(f"Error merging content: {e}")
        return existing_content, False


def finding_duplicate_nodes(driver, subject, object_, new_content, project_name):
    with driver.session() as session:
        result = session.run(
            "MATCH (s:Entity {name: $subject, project_name: $project_name})-[r]->(o:Entity {name: $object, project_name: $project_name}) "
            "RETURN s, r, o, o.content AS content",
            subject=subject,
            object=object_,
            project_name=project_name
        )
        relationship_record = result.single()
        if relationship_record:
            existing_content = relationship_record["content"]
            logger.debug(f"Existing content for {subject} - {object_}: {existing_content}")
            merged_content = existing_content
            if existing_content and new_content:
                entity_name = relationship_record["o"]["name"]
                merged_content, new_info_found = merge_content_intelligently(
                    existing_content, new_content, entity_name
                )
                content_embedding = embed_content(merged_content)
                with driver.session() as session:
                    session.run(
                        "MATCH (o:Entity {name: $object, project_name: $project_name}) "
                        "SET o.content = $merged_content, o.embedding = $embedding",
                        object=object_,
                        merged_content=merged_content,
                        embedding=content_embedding,
                        project_name=project_name
                    )
                logger.debug(f"Updated content for {object_} with new_info={new_info_found}")
                return new_info_found, True
            if not existing_content:
                return False, True
            else:
                logger.debug(f"No new information found for {subject} - {object_}. Skipping update.")
                return False, True
        else:
            logger.debug(f"No duplicate found for {subject} - {object_}.")
            return False, False


def find_children_nodes(driver, node_name, project_name):
    with driver.session() as session:
        result = session.run(
            "MATCH (n:Entity {name: $name, project_name: $project_name})-[:RELATION]->(child) "
            "RETURN child.name as name, child.content as content",
            name=node_name,
            project_name=project_name
        )
        children = [{"name": record["name"], "content": record["content"]} for record in result]
        logger.debug(f"Found {len(children)} children for node '{node_name}': {[child['name'] for child in children]}")
        if not children:
            logger.debug(f"No children found for node: {node_name}")
            return []
        return children


def update_root_nodes_with_children_summary(driver, project_name):
    logger.debug(f"Starting root node content update for project '{project_name}'")
    root_nodes = find_root_nodes(driver, project_name)
    if not root_nodes:
        logger.debug("No root nodes found to update")
        return 0
    updated_count = 0
    with driver.session() as session:
        for root_node_info in root_nodes:
            root_node_name = root_node_info["name"]
            logger.debug(f"Processing root node: {root_node_name}")
            children_nodes = find_children_nodes(driver, root_node_name, project_name)
            if not children_nodes:
                logger.debug(f"No children found for root node: {root_node_name}")
                continue
            all_children_content = []
            for child in children_nodes:
                if child["content"] is not None:
                    all_children_content.append(f"{child['name']}: {child['content']}")
                else:
                    all_children_content.append(f"{child['name']}: No content available")
            if all_children_content:
                combined_children_content = "\n".join(all_children_content)
                try:
                    response = client().chat.completions.create(
                        model="chatgpt-4o-latest",
                        messages=[
                            {"role": "system", "content": prompt_manager.get_summary_system_prompt()},
                            {"role": "user", "content": prompt_manager.get_summary_user_prompt(root_node_name, combined_children_content)},
                        ],
                        temperature=config.llm.temperature
                    )
                    summary_content = response.choices[0].message.content.strip()
                    if summary_content:
                        summary_embedding = embed_content(summary_content)
                        session.run(
                            "MATCH (n:Entity {name: $name, project_name: $project_name}) "
                            "SET n.content = $content, n.embedding = $embedding, n.type = 'root_node'",
                            name=root_node_name,
                            content=summary_content,
                            embedding=summary_embedding,
                            project_name=project_name
                        )
                        updated_count += 1
                        logger.debug(f"Updated root node '{root_node_name}' with children summary")
                except Exception as e:
                    logger.error(f"Error generating summary for root node '{root_node_name}': {e}")
                    continue
    logger.debug(f"Successfully updated {updated_count} root nodes with children summaries")
    return updated_count


def find_parent_nodes(driver, node_name, project_name):
    with driver.session() as session:
        result = session.run(
            "MATCH (n:Entity {name: $name, project_name: $project_name}) <-[:RELATION]-(parent:Entity) "
            "RETURN parent.name as name, parent.content as content, parent.type as type",
            name=node_name,
            project_name=project_name
        )
        if not result:
            logger.debug(f"No parent nodes found for {node_name}.")
            return []
        return [record["name"] for record in result]


def find_root_nodes(driver, project_name):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Entity {project_name: $project_name})
            OPTIONAL MATCH (n)-[:RELATION]->(child:Entity {project_name: $project_name})
            OPTIONAL MATCH (parent:Entity {project_name: $project_name})-[r]->(n)
            WITH n, count(child) AS child_count, count(parent) AS parent_count
            WHERE child_count >= 1 AND parent_count = 0
            RETURN n.name AS name, child_count
            """,
            project_name=project_name
        )
        root_candidates = []
        records = list(result)
        for record in records:
            root_candidates.append({
                "name": record["name"],
                "child_count": record["child_count"]
            })
        logger.debug(f"Root candidates found: {root_candidates}")
        return root_candidates


def link_root_nodes_to_project(project_name, session, driver):
    session.run(
        "MERGE (p:Project {name: $project_name}) "
        "ON CREATE SET p.created_at = datetime(), p.type = 'project' "
        "ON MATCH SET p.updated_at = datetime()",
        project_name=project_name
    )
    logger.debug(f"Created/Updated project node for '{project_name}'")
    root_nodes = find_root_nodes(driver, project_name)
    if root_nodes:
        for root_node in root_nodes:
            logger.debug(f"Found root node for project '{project_name}': {root_node['name']}")
    linked_count = 0
    if not root_nodes:
        logger.debug(f"No root nodes found to link to project '{project_name}'")
        return 0
    for root_node in root_nodes:
        root_name = root_node["name"]
        session.run(
            "MATCH (p:Project {name: $project_name}) "
            "MATCH (r:Entity {name: $root_name, project_name: $project_name}) "
            "MERGE (p)-[:CONTAINS]->(r)",
            project_name=project_name,
            root_name=root_name
        )
        linked_count += 1
        logger.debug(f"Linked root node '{root_name}' to project '{project_name}'")
    logger.debug(f"Successfully linked {linked_count} root nodes to project '{project_name}'")
    return linked_count


def embed_content(text):
    embedding = get_embedding_model().encode(text, convert_to_tensor=True)
    return embedding.tolist()


def new_extract_triplets(chunks: list):
    for chunk in chunks:
        chunk["triplets"] = [chunk.get("parent_heading", None), "contains", chunk.get("heading", None)]
    return chunks


def new_save_triplets(driver, triplets_data, project_name):
    start_time_again = time.time()
    logger.debug("Starting to save triplets to Neo4j database")
    with driver.session() as session:
        for triplet in triplets_data:
            subject, relation, object_ = triplet["triplets"]
            new_content = triplet['content']
            new_info, duplicate_exist = finding_duplicate_nodes(driver, subject, object_, new_content, project_name)
            if duplicate_exist:
                logger.debug(f"Duplicate found: {subject} - {relation} - {object_}")
            else:
                session.run(
                    "MERGE (s:Entity {name: $subject, project_name: $project_name}) "
                    "MERGE (o:Entity {name: $object, project_name: $project_name}) "
                    "MERGE (s)-[r:RELATION {type: $relation}]->(o)",
                    subject=subject,
                    relation=relation,
                    object=object_,
                    project_name=project_name
                )
                if new_content is not None:
                    content_embedding = embed_content(new_content)
                    session.run(
                        "MATCH (n:Entity {name: $name, project_name: $project_name}) "
                        "SET n.content = $content, n.type = 'content_node', n.embedding = $embedding",
                        name=object_,
                        content=new_content,
                        embedding=content_embedding,
                        project_name=project_name
                    )
        logger.debug("Starting bottom-up summary generation for root nodes")
        updated_root_count = update_root_nodes_with_children_summary(driver, project_name)
        logger.debug(f"Updated {updated_root_count} root nodes with comprehensive children summaries")
        if project_name:
            logger.debug(f"Linking root nodes to project '{project_name}'")
            linked_count = link_root_nodes_to_project(project_name, session, driver)
            logger.debug(f"Linked {linked_count} root nodes to project '{project_name}'")
    end_time_again = time.time()
    logger.debug(f"Triplets saved to Neo4j database in {end_time_again - start_time_again:.2f} seconds")
    logger.debug("Triplets saved to Neo4j database successfully")


def new_invoke_ingestion(driver, project_name: str, summarized_extracted_json: list):
    logger.info(f"Starting new ingestion for project: {project_name}")
    logger.debug(f"Extracted JSON data: {summarized_extracted_json}")
    start = time.time()
    chunks = summarized_extracted_json
    if not chunks:
        logger.info("Please provide info to process")
        return None
    triplets_data = new_extract_triplets(chunks)
    if triplets_data:
        logger.debug(f"Extracted {len(triplets_data)} triplets")
        logger.debug(f"Sample chunk: {triplets_data[0] if triplets_data else 'None'}")
        new_save_triplets(driver, triplets_data, project_name)
    else:
        logger.warning("No triplets extracted from the input text")
    end = time.time()
    logger.info(f"New ingestion process completed in {end - start:.2f} seconds")


