# llm_wrappers.py
import json
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import config
from langchain_community.embeddings import HuggingFaceEmbeddings

class LLMManager:
    """封装与大语言模型 (LLM) 的交互。"""
    def __init__(self):
        self.client = OpenAI(
            base_url=config.DEEPSEEK_API_URL,
            api_key=config.DEEPSEEK_API_KEY,
        )
        self.model = config.DEEPSEEK_MODEL_NAME

    def generate_response(self, messages: list, tools: list = None, tool_choice: str = "auto", temperature: float = 0.5, use_json_format: bool = False):
        """调用LLM生成响应，并支持函数调用和JSON格式。"""
        params = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        if use_json_format:
            params["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**params)
        
        if not response.choices or not response.choices[0].message:
            return {"content": "模型未返回有效响应。", "tool_calls": []}
            
        message = response.choices[0].message
        
        # 处理可能的JSON字符串响应
        content = message.content
        if use_json_format:
            try:
                # 尝试直接解析content，因为Ollama有时会将JSON对象作为字符串返回
                parsed_json = json.loads(content)
                return {"content": parsed_json, "tool_calls": []}
            except (json.JSONDecodeError, TypeError):
                # 如果解析失败，则按原样返回，让调用者处理
                pass
        
        if message.tool_calls:
            parsed_tool_calls = []
            for tool_call in message.tool_calls:
                arguments = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                parsed_tool_calls.append({"name": tool_call.function.name, "arguments": arguments})
            return {"content": message.content, "tool_calls": parsed_tool_calls}
        else:
            return {"content": content, "tool_calls": []}


class EmbeddingManager:
    """管理嵌入模型和重排序模型。"""
    def __init__(self):
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}  # 或 'cuda'
            )
            #self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            self.reranker_model = CrossEncoder(config.RERANKER_MODEL_PATH)
        except Exception as e:
            print(f"致命错误: 加载嵌入或重排序模型失败: {e}")
            raise


    def get_embedding_dimension(self) -> int:
        # 这个方法可能仍被Neo4j索引需要，但获取方式变了
        # HuggingFaceEmbeddings 没有 get_sentence_embedding_dimension
        # 我们直接返回一个已知的值，或者从模型配置读取
        return 384 # 比如 all-MiniLM-L6-v2 的维度是 384
    
    '''def get_embedding_dimension(self) -> int:
        return self.embedding_model.get_sentence_embedding_dimension()'''

    def rerank_documents(self, query: str, documents: list) -> list:
        if not documents or not self.reranker_model:
            return documents
        doc_contents = [doc.page_content for doc in documents]
        pairs = [[query, doc] for doc in doc_contents]
        scores = self.reranker_model.predict(pairs)
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs]