# memory_manager.py
import faiss
import numpy as np
import os
import json
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import HuggingFaceEmbeddings # 注意 import 的变化

from llm_wrappers import LLMManager, EmbeddingManager
from graphs.tools import EXTRACT_ENTITIES_TOOL, RELATIONS_TOOL, DELETE_MEMORY_TOOL_GRAPH
from graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
import config

class MemoryAssistant:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.llm = LLMManager()
        self.embedder = EmbeddingManager()
        
        # 初始化 Neo4j Graph
        try:
            self.graph = Neo4jGraph(
                url=config.NEO4J_URL,
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
                database=config.NEO4J_DATABASE
            )
            # 创建必要的索引以提高性能
            self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (n:__Entity__) REQUIRE n.id IS UNIQUE;")
            self.graph.query("CREATE INDEX IF NOT EXISTS FOR (n:__Entity__) ON (n.user_id);")
            self.graph.query("CREATE VECTOR INDEX `entity_embeddings` IF NOT EXISTS FOR (n:__Entity__) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}")
            print("Neo4j 连接成功并已创建索引。")
        except Exception as e:
            print(f"错误：无法连接到 Neo4j 数据库。请检查您的配置和数据库状态。")
            print(f"详细错误: {e}")
            raise

    def _extract_entities(self, text: str):
        """从文本中提取实体及其类型。"""
        messages = [
            {"role": "system", "content": f"You are an expert entity extractor. Extract entities and their types from the text. For any self-references like 'I', 'me', 'my', use the entity ID '{self.user_id}' with the type 'User'."},
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

        

    def _extract_relationships(self, text: str, entities: list):
        """从文本中提取实体间的关系。"""
        system_prompt = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", self.user_id).replace("CUSTOM_PROMPT", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Entities: {entities}\n\nText: {text}"}
        ]
        response = self.llm.generate_response(messages, tools=[RELATIONS_TOOL])
        
        relationships = []
        if response and response['tool_calls']:
             for tool_call in response['tool_calls']:
                if tool_call['name'] == 'establish_relationships':
                    relationships = tool_call['arguments']['entities']
        return self._sanitize_relationships(relationships)
        
    def _sanitize_relationships(self, relationships: list):
        """清理关系，将名称转换为小写和下划线格式。"""
        sanitized = []
        for rel in relationships:
            sanitized.append({
                "source": rel["source"].lower().replace(" ", "_"),
                "relationship": rel["relationship"].upper(),
                "destination": rel["destination"].lower().replace(" ", "_"),
            })
        return sanitized

    def _get_entities_to_delete(self, existing_rels: list, new_text: str):
        """判断哪些现有关系应该被新文本所取代或删除。"""
        if not existing_rels:
            return []
            
        existing_rels_str = "\n".join([f"{r['source']} -- {r['relationship']} -- {r['destination']}" for r in existing_rels])
        system_prompt, user_prompt = get_delete_messages(existing_rels_str, new_text, self.user_id)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.llm.generate_response(messages, tools=[DELETE_MEMORY_TOOL_GRAPH])
        
        to_delete = []
        if response and response['tool_calls']:
            for tool_call in response['tool_calls']:
                if tool_call['name'] == 'delete_graph_memory':
                    to_delete.append(tool_call['arguments'])
        return self._sanitize_relationships(to_delete)

    def add_memory(self, text: str):
        """
        处理一段文本，提取信息，并更新知识图谱。

        这是一个多步骤的过程：
        1. 从文本中提取所有实体。
        2. 基于这些实体，从文本中提取它们之间的关系。
        3. 搜索图中已存在的、与新实体相关的关系。
        4. 判断哪些旧关系需要被删除。
        5. 从图中删除过时的关系。
        6. 将新的关系（包括实体节点）添加或合并到图中。
        """
        print(f"\n--- 正在为用户 '{self.user_id}' 添加新记忆 ---")
        print(f"输入文本: \"{text}\"")

        # 1. 提取实体
        entities_map = self._extract_entities(text)
        if not entities_map:
            print("未提取到实体，流程终止。")
            return
        print(f"提取到的实体: {entities_map}")

        # 2. 提取关系
        new_relationships = self._extract_relationships(text, list(entities_map.keys()))
        print(f"提取到的新关系: {new_relationships}")

        # 3. 搜索相关旧关系
        all_entity_names = list(entities_map.keys())
        for rel in new_relationships:
            all_entity_names.append(rel['source'])
            all_entity_names.append(rel['destination'])
        
        existing_relationships = self.search_memory(query=" ".join(set(all_entity_names)), search_type='graph_only')
        print(f"查询到的现有关系: {existing_relationships}")

        # 4. 判断要删除的关系
        to_delete = self._get_entities_to_delete(existing_relationships, text)
        print(f"计划删除的关系: {to_delete}")

        # 5. 执行删除
        for rel in to_delete:
            self.graph.query(
                "MATCH (n:__Entity__ {id: $source_id})-[r]->(m:__Entity__ {id: $dest_id}) WHERE type(r) = $rel_type DELETE r",
                params={
                    "source_id": f"{rel['source']}_{self.user_id}",
                    "dest_id": f"{rel['destination']}_{self.user_id}",
                    "rel_type": rel['relationship']
                }
            )
        print(f"成功删除 {len(to_delete)} 条关系。")
        
        # 6. 添加新关系
        for rel in new_relationships:
            source_id = f"{rel['source']}_{self.user_id}"
            dest_id = f"{rel['destination']}_{self.user_id}"
            source_type = entities_map.get(rel['source'], 'Thing')
            dest_type = entities_map.get(rel['destination'], 'Thing')

            self.graph.query(
                f"""
                MERGE (n:__Entity__ {{id: $source_id}})
                ON CREATE SET n.name = $source_name, n.user_id = $user_id, n.embedding = $source_embedding, n:{source_type}
                MERGE (m:__Entity__ {{id: $dest_id}})
                ON CREATE SET m.name = $dest_name, m.user_id = $user_id, m.embedding = $dest_embedding, m:{dest_type}
                MERGE (n)-[r:{rel['relationship']}]->(m)
                RETURN n.name, type(r), m.name
                """,
                params={
                    "source_id": source_id,
                    "source_name": rel['source'],
                    "user_id": self.user_id,
                    "source_embedding": self.embedder.embed_text(rel['source']).tolist(),
                    "dest_id": dest_id,
                    "dest_name": rel['destination'],
                    "dest_embedding": self.embedder.embed_text(rel['destination']).tolist(),
                }
            )
        print(f"成功添加/更新 {len(new_relationships)} 条关系。")
        print("--- 记忆更新完成 ---")


    def search_memory(self, query: str, top_k: int = 5, search_type: str = 'hybrid'):
        """
        根据查询在知识图谱中搜索相关信息。

        Args:
            query (str): 用户的查询文本。
            top_k (int): 返回结果的数量。
            search_type (str): 'hybrid' (默认) 或 'graph_only'。

        Returns:
            list: 相关的关系三元组列表。
        """
        print(f"\n--- 正在为用户 '{self.user_id}' 搜索记忆 ---")
        print(f"查询: \"{query}\"")
        
        query_embedding = self.embedder.embed_text(query).tolist()
        
        # 使用向量索引进行初步召回
        try:
            results = self.graph.query(
                """
                CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $query_embedding)
                YIELD node, score
                WHERE node.user_id = $user_id
                MATCH (node)-[r]-(related_node)
                RETURN node.name AS source, type(r) AS relationship, related_node.name AS destination, score
                ORDER BY score DESC
                """,
                params={
                    "query_embedding": query_embedding,
                    "user_id": self.user_id,
                    "top_k": top_k
                }
            )
            print(f"向量搜索召回了 {len(results)} 条初步结果。")

            if search_type == 'graph_only':
                return [{"source": r['source'], "relationship": r['relationship'], "destination": r['destination']} for r in results]

            # 对结果进行重排序
            documents_to_rerank = [f"{r['source']} {r['relationship']} {r['destination']}" for r in results]
            reranked_docs = self.embedder.rerank_results(query, documents_to_rerank)
            
            # 创建一个从文档内容到原始结果的映射
            doc_to_result_map = {doc: result for doc, result in zip(documents_to_rerank, results)}

            # 按重排顺序整理最终结果
            final_results = []
            for doc in reranked_docs:
                original_result = doc_to_result_map[doc]
                final_results.append({
                    "source": original_result['source'],
                    "relationship": original_result['relationship'],
                    "destination": original_result['destination']
                })
            
            print(f"重排序后返回 {len(final_results)} 条结果。")
            print("--- 搜索完成 ---")
            return final_results

        except Exception as e:
            print(f"在搜索过程中发生错误: {e}")
            return []

    def clear_memory(self):
        """为当前用户清除所有记忆。"""
        print(f"正在为用户 '{self.user_id}' 清除所有记忆...")
        self.graph.query(
            "MATCH (n:__Entity__ {user_id: $user_id}) DETACH DELETE n",
            params={"user_id": self.user_id}
        )
        print("记忆已清除。")

    