# memory.py
import os
import shutil
import datetime
import time
import math
import numpy as np
from typing import List, Dict, Tuple

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings # 注意 import 的变化

import config
from llm_wrappers import LLMManager, EmbeddingManager

# --- 直接在此处定义Tools和Prompts，避免循环依赖 ---

EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "从文本中提取实体及其类型。",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "提取出的实体。"},
                            "entity_type": {"type": "string", "description": "实体的类型 (例如: Person, Place, Thing)。"}
                        },
                        "required": ["entity", "entity_type"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
}

RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": "基于文本在实体间建立关系。",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "源实体。"},
                            "relationship": {"type": "string", "description": "实体间的关系。"},
                            "destination": {"type": "string", "description": "目标实体。"}
                        },
                        "required": ["source", "relationship", "destination"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
}

DELETE_MEMORY_TOOL_GRAPH = {
    "type": "function",
    "function": {
        "name": "delete_graph_memory",
        "description": "从知识图谱中删除特定的记忆或关系。",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "要删除的关系的源实体。"},
                "relationship": {"type": "string", "description": "要删除的关系类型。"},
                "destination": {"type": "string", "description": "要删除的关系的目标实体。"}
            },
            "required": ["source", "relationship", "destination"]
        }
    }
}

EXTRACT_RELATIONS_PROMPT = """
你是一位知识图谱专家。你的任务是从文本中提取给定实体之间的关系。
关系应为 (源实体, 关系, 目标实体) 的格式。
非常重要：你只能使用列表中提供的实体，不要创建新实体。
对于像 "我", "我的" 这样的自我指涉，如果实体列表中有 'USER_ID'，请使用它。
"""

def get_delete_messages(existing_rels: str, new_text: str, user_id: str) -> Tuple[str, str]:
    system_prompt = f"""
你是一位知识图谱的记忆管理专家。
你的任务是根据新的输入信息，判断是否应删除任何现有的记忆（关系）。
如果新文本明确地与现有记忆矛盾、取代或使其失效，则应删除该记忆。
例如，如果现有记忆是“用户住在上海”，而新文本是“我搬到了北京”，你就应该删除旧的记忆。

用户 '{user_id}' 的现有相关记忆:
{existing_rels}

请分析新文本，并决定哪些现有记忆现在已过时，应被删除。
只列出需要删除的关系。如果不需要删除任何记忆，则不要调用函数。
    """
    user_prompt = f"要处理的新信息: \"{new_text}\""
    return system_prompt, user_prompt


class KnowledgeAndMemoryManager:
    """统一管理知识图谱(Neo4j)和向量事实库(FAISS)，并包含记忆的生命周期管理。"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = LLMManager()
        self.embedder = EmbeddingManager()
        self.faiss_index_path = os.path.join(config.FAISS_INDEX_BASE_DIR, self.user_id)
        
        self.graph = self._init_neo4j()
        self.faiss_db = self._load_or_create_faiss_db()
        # 初始化时执行一次遗忘检查
        self.prune_memory()

    def _init_neo4j(self) -> Neo4jGraph:
        """初始化Neo4j连接和索引。"""
        try:
            graph = Neo4jGraph(
                url=config.NEO4J_URL, username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD, database=config.NEO4J_DATABASE
            )
            embedding_dim = self.embedder.get_embedding_dimension()
            graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (n:__Entity__) REQUIRE n.id IS UNIQUE;")
            graph.query("CREATE INDEX IF NOT EXISTS FOR (n:__Entity__) ON (n.user_id);")
            graph.query(f"CREATE VECTOR INDEX `entity_embeddings` IF NOT EXISTS FOR (n:__Entity__) ON (n.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dim}, `vector.similarity_function`: 'cosine'}}}}")
            print(f"Neo4j connection successful for user '{self.user_id}'.")
            return graph
        except Exception as e:
            print(f"致命错误: 无法连接到 Neo4j。请检查配置和数据库状态。错误: {e}")
            raise

    def _load_or_create_faiss_db(self) -> FAISS:
        """加载或创建用户的FAISS索引。"""
        if os.path.exists(self.faiss_index_path):
            try:
                print(f"[FAISS] 正在从本地加载 '{self.user_id}' 的向量数据库...")
                return FAISS.load_local(self.faiss_index_path, self.embedder.embedding_model, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"[FAISS][警告]: 加载索引失败，将重建。错误: {e}")
        
        # 创建一个空的索引以避免错误
        dummy_doc = [Document(page_content="initialization")]
        db = FAISS.from_documents(dummy_doc, self.embedder.embedding_model)
        db.save_local(self.faiss_index_path)
        print(f"[FAISS] 已为 '{self.user_id}' 创建新的向量数据库。")
        return db

    def get_combined_context(self, query: str, k_graph: int = 3, k_facts: int = 3) -> Tuple[str, List[Document]]:
        """
        检索知识图谱和事实库，合并并格式化上下文，供LLM使用。
        """
        print(f"[Memory] 正在为查询 '{query}' 合并上下文...")
        
        # 1. 从两个来源搜索
        graph_results = self.search_knowledge_graph(query, top_k=k_graph)
        fact_docs = self.search_facts(query, k=k_facts)

        # 2. 合并和格式化结果
        combined_docs = []
        context_parts = []

        # 格式化知识图谱结果
        if graph_results:
            context_parts.append("--- 来自知识图谱的记忆 ---")
            for rel in graph_results:
                content = f"关系: {rel.get('source')} -> {rel.get('relationship')} -> {rel.get('destination')}"
                metadata = {"source": "knowledge_graph"}
                combined_docs.append(Document(page_content=content, metadata=metadata))
                context_parts.append(content)
        
        # 格式化事实库结果
        if fact_docs:
            context_parts.append("\n--- 来自事实库的记忆 ---")
            for doc in fact_docs:
                combined_docs.append(doc) # 已经是Document对象
                context_parts.append(doc.page_content)

        if not context_parts:
            print("[Memory] 未在任何记忆库中找到相关信息。")
            return "没有在记忆中找到相关信息。", []

        # 3. 创建最终的上下文文本并返回
        final_context_str = "\n".join(context_parts)
        print(f"[Memory] 合并后的上下文长度: {len(final_context_str)} characters.")
        return final_context_str, combined_docs


    # --- 知识图谱相关方法 ---
    def _extract_entities(self, text: str) -> Dict[str, str]:
        """从文本中提取实体及其类型。"""
        messages = [
            {"role": "system", "content": f"你是一个专业的实体提取器。从文本中提取实体及其类型。对于任何像'我'、'我的'这样的自我指涉，请使用实体ID '{self.user_id}' 和类型 'User'。"},
            {"role": "user", "content": text}
        ]
        response = self.llm.generate_response(messages, tools=[EXTRACT_ENTITIES_TOOL])
        entity_map = {}
        if response and response['tool_calls']:
            for tool_call in response['tool_calls']:
                if tool_call['name'] == 'extract_entities':
                    for entity in tool_call['arguments']['entities']:
                        entity_map[entity['entity'].lower().replace(" ", "_")] = entity['entity_type'].lower().replace(" ", "_")
        return entity_map

    def _extract_relationships(self, text: str, entities: List[str]) -> List[Dict]:
        """从文本中提取实体间的关系。"""
        system_prompt = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", self.user_id)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"实体: {entities}\n\n文本: {text}"}
        ]
        response = self.llm.generate_response(messages, tools=[RELATIONS_TOOL])
        relationships = []
        if response and response['tool_calls']:
             for tool_call in response['tool_calls']:
                if tool_call['name'] == 'establish_relationships':
                    relationships = tool_call['arguments']['entities']
        return self._sanitize_relationships(relationships)
        
    def _sanitize_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """清理关系，将名称转换为小写和下划线格式。"""
        sanitized = []
        for rel in relationships:
            sanitized.append({
                "source": rel["source"].lower().replace(" ", "_"),
                "relationship": rel["relationship"].upper().replace(" ", "_"),
                "destination": rel["destination"].lower().replace(" ", "_"),
            })
        return sanitized

    def _get_entities_to_delete(self, existing_rels: List[Dict], new_text: str) -> List[Dict]:
        """【深度思考环节】判断哪些现有关系应该被新文本所取代或删除。"""
        if not existing_rels:
            return []
        existing_rels_str = "\n".join([f"({r['source']})-[{r['relationship']}]->({r['destination']})" for r in existing_rels])
        system_prompt, user_prompt = get_delete_messages(existing_rels_str, new_text, self.user_id)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.llm.generate_response(messages, tools=[DELETE_MEMORY_TOOL_GRAPH])
        to_delete = []
        if response and response['tool_calls']:
            for tool_call in response['tool_calls']:
                if tool_call['name'] == 'delete_graph_memory':
                    to_delete.append(tool_call['arguments'])
        return self._sanitize_relationships(to_delete)

    def add_knowledge_graph_memory(self, text: str):
        """处理文本并智能地更新知识图谱。"""
        print(f"\n--- [KG Memory] 用户 '{self.user_id}': 开始处理记忆: '{text}' ---")
        entities_map = self._extract_entities(text)
        if not entities_map:
            print("[KG Memory] 未提取到实体，流程中止。")
            return
        new_relationships = self._extract_relationships(text, list(entities_map.keys()))
        if not new_relationships:
            print("[KG Memory] 未提取到新关系，流程中止。")
            return
        all_entity_names = list(entities_map.keys())
        for rel in new_relationships:
            all_entity_names.extend([rel['source'], rel['destination']])
        related_nodes = [f'"{name}_{self.user_id}"' for name in set(all_entity_names)]
        if related_nodes:
            query = f"MATCH (n:__Entity__)-[r]-(m:__Entity__) WHERE n.id IN [{','.join(related_nodes)}] RETURN n.name AS source, type(r) AS relationship, m.name AS destination"
            existing_relationships = self._sanitize_relationships(self.graph.query(query))
        else:
            existing_relationships = []
        to_delete = self._get_entities_to_delete(existing_relationships, text)
        for rel in to_delete:
            self.graph.query("MATCH (n:__Entity__ {id: $source_id})-[r]->(m:__Entity__ {id: $dest_id}) WHERE type(r) = $rel_type DELETE r", params={"source_id": f"{rel['source']}_{self.user_id}", "dest_id": f"{rel['destination']}_{self.user_id}", "rel_type": rel['relationship']})
        for rel in new_relationships:
            source_id, dest_id = f"{rel['source']}_{self.user_id}", f"{rel['destination']}_{self.user_id}"
            source_type, dest_type = entities_map.get(rel['source'], 'Thing'), entities_map.get(rel['destination'], 'Thing')
            self.graph.query(f"MERGE (n:__Entity__ {{id: $source_id}}) ON CREATE SET n.name = $source_name, n.user_id = $user_id, n.embedding = $source_embedding, n.type = '{source_type}', n.timestamp = $timestamp, n.strength = 1.0 ON MATCH SET n.embedding = $source_embedding, n.timestamp = $timestamp MERGE (m:__Entity__ {{id: $dest_id}}) ON CREATE SET m.name = $dest_name, m.user_id = $user_id, m.embedding = $dest_embedding, m.type = '{dest_type}', m.timestamp = $timestamp, m.strength = 1.0 ON MATCH SET m.embedding = $dest_embedding, m.timestamp = $timestamp MERGE (n)-[r:{rel['relationship']}]->(m)", params={"source_id": source_id, "source_name": rel['source'], "user_id": self.user_id, "source_embedding": self.embedder.embedding_model.embed_query(rel['source']), "dest_id": dest_id, "dest_name": rel['destination'], "dest_embedding": self.embedder.embedding_model.embed_query(rel['destination']), "timestamp": time.time()})
        print(f"--- [KG Memory] 记忆处理完成。删除了 {len(to_delete)} 条，新增/更新了 {len(new_relationships)} 条关系。 ---")

    def search_knowledge_graph(self, query: str, top_k: int = 5) -> List[Dict]:
        """在知识图谱中进行混合搜索。"""
        print(f"--- [KG Memory] 正在搜索: '{query}' ---")
        query_embedding = self.embedder.embedding_model.embed_query(query)
        results = self.graph.query("CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $query_embedding) YIELD node, score WHERE node.user_id = $user_id MATCH (node)-[r]-(related_node) RETURN node.name AS source, type(r) AS relationship, related_node.name AS destination, score ORDER BY score DESC LIMIT $top_k", params={"query_embedding": query_embedding, "user_id": self.user_id, "top_k": top_k})
        return results if results else []

    # --- 事实库 (FAISS) 相关方法 ---
    def add_fact_memory(self, doc: Document):
        """向FAISS中添加一个事实、摘要或画像。"""
        doc.metadata.setdefault("id", f"fact_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        doc.metadata.setdefault("timestamp", time.time())
        doc.metadata.setdefault("strength", 1.0)
        self.faiss_db.add_documents([doc])
        self.faiss_db.save_local(self.faiss_index_path)
        print(f"[FAISS Memory] 已添加文档 (类型: {doc.metadata.get('type')}, ID: {doc.metadata.get('id')})")

    def search_facts(self, query: str, k: int = 5) -> List[Document]:
        """在FAISS中搜索相似的事实或摘要。"""
        if self.faiss_db.index.ntotal == 0: return []
        return self.faiss_db.similarity_search(query, k=k)
    
    def get_all_facts_of_type(self, doc_type: str) -> List[Document]:
        """获取FAISS中所有特定类型的文档。"""
        if self.faiss_db.index.ntotal == 0: return []
        all_docs = [self.faiss_db.docstore.search(self.faiss_db.index_to_docstore_id[i]) for i in range(self.faiss_db.index.ntotal)]
        return [doc for doc in all_docs if doc and doc.metadata.get('type') == doc_type]

    def reinforce_memory(self, used_docs: List[Document]):
        """根据用于生成答案的记忆块，加固相应记忆的强度。"""
        print(f"[Memory Lifecycle] 正在为 {len(used_docs)} 条记忆增强强度...")
        faiss_updated = False
        for doc in used_docs:
            source = doc.metadata.get("source")
            if source == "knowledge_graph":
                # 对于KG，我们假设实体名称是唯一的，并更新其强度
                # 这是一个简化，更复杂的情况需要实体ID
                self.graph.query("MATCH (n:__Entity__ {name: $name, user_id: $user_id}) SET n.strength = n.strength + 0.1, n.timestamp = $timestamp", params={"name": doc.page_content.split(" -> ")[0].replace("关系: ", ""), "user_id": self.user_id, "timestamp": time.time()})
            elif source == "faiss":
                doc_id = doc.metadata.get("id")
                if doc_id and self.faiss_db.docstore.search(doc_id):
                    # FAISS本身不易更新元数据，我们将通过重建索引来模拟
                    # 在实际生产中，会使用更高级的向量数据库
                    doc.metadata["strength"] = doc.metadata.get("strength", 1.0) + 0.1
                    doc.metadata["timestamp"] = time.time()
                    faiss_updated = True
        if faiss_updated:
            print("[Memory Lifecycle] FAISS中有记忆强度更新，正在重建索引...")
            all_docs = [self.faiss_db.docstore.search(self.faiss_db.index_to_docstore_id[i]) for i in range(self.faiss_db.index.ntotal)]
            shutil.rmtree(self.faiss_index_path)
            self.faiss_db = FAISS.from_documents(all_docs, self.embedder.embedding_model)
            self.faiss_db.save_local(self.faiss_index_path)
            
    def prune_memory(self):
        """根据艾宾浩斯遗忘曲线修剪记忆。"""
        print(f"[Memory Lifecycle] 正在为用户 '{self.user_id}' 检查需要遗忘的记忆...")
        current_time = time.time()
        # 1. 修剪知识图谱
        all_nodes = self.graph.query("MATCH (n:__Entity__ {user_id: $user_id}) RETURN n.id AS id, n.strength AS strength, n.timestamp AS timestamp", params={"user_id": self.user_id})
        nodes_to_delete = []
        for node in all_nodes:
            if not node['timestamp']: continue
            strength = node.get('strength') or 1.0
            S = max(1.0, strength) * (86400 * 7) # 7天基准
            retention = math.exp(-(current_time - node['timestamp']) / S)
            if retention < config.FORGETTING_THRESHOLD:
                nodes_to_delete.append(node['id'])
        if nodes_to_delete:
            self.graph.query("MATCH (n:__Entity__) WHERE n.id IN $ids DETACH DELETE n", params={"ids": nodes_to_delete})
            print(f"[Memory Lifecycle] 从知识图谱中遗忘了 {len(nodes_to_delete)} 个实体。")
            
        # 2. 修剪FAISS
        if self.faiss_db.index.ntotal > 0:
            all_docs = [self.faiss_db.docstore.search(self.faiss_db.index_to_docstore_id[i]) for i in range(self.faiss_db.index.ntotal)]
            retained_docs = []
            for doc in all_docs:
                if doc and doc.page_content != "initialization":
                    # 摘要和画像通常更重要，给它们更长的半衰期
                    base_days = 30 if doc.metadata.get('type') in ["daily_summary", "overall_personality"] else 7
                    strength = doc.metadata.get('strength') or 1.0
                    timestamp = doc.metadata.get('timestamp') or current_time
                    S = max(1.0, strength) * (86400 * base_days)
                    retention = math.exp(-(current_time - timestamp) / S)
                    if retention >= config.FORGETTING_THRESHOLD:
                        retained_docs.append(doc)
            if len(retained_docs) < len(all_docs) - 1: # -1 for the init doc
                print(f"[Memory Lifecycle] 从FAISS中遗忘了 {len(all_docs) - 1 - len(retained_docs)} 个事实。正在重建索引...")
                shutil.rmtree(self.faiss_index_path)
                self.faiss_db = FAISS.from_documents(retained_docs if retained_docs else [Document(page_content="initialization")], self.embedder.embedder)
                self.faiss_db.save_local(self.faiss_index_path)

    def clear_all_memory(self):
        """清除当前用户的所有记忆。"""
        print(f"--- [Memory] 正在清除用户 '{self.user_id}' 的所有记忆 ---")
        self.graph.query("MATCH (n:__Entity__ {user_id: $user_id}) DETACH DELETE n", params={"user_id": self.user_id})
        if os.path.exists(self.faiss_index_path):
            shutil.rmtree(self.faiss_index_path)
        self.faiss_db = self._load_or_create_faiss_db()
        print(f"[Memory] 用户 '{self.user_id}' 的所有记忆已清除。")