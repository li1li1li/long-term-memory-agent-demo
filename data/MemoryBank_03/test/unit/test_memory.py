# tests/unit/test_memory.py
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from assistant.memory import ShortTermCache, StableHuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

class TestShortTermCache(unittest.TestCase):

    def setUp(self):
        """在每个测试前运行，准备测试环境。"""
        # 注意：为避免每次测试都加载大模型，这里可以替换为模拟对象(mock)
        self.embeddings = StableHuggingFaceEmbeddings(EMBEDDING_MODEL_NAME)
        self.cache = ShortTermCache(capacity=3, embeddings_model=self.embeddings)

    def test_add_and_capacity(self):
        """测试添加项目和容量限制是否正常工作。"""
        self.assertEqual(len(self.cache.memory), 0)
        self.cache.add("q1", "r1")
        self.cache.add("q2", "r2")
        self.cache.add("q3", "r3")
        self.assertEqual(len(self.cache.memory), 3)
        self.assertIn("q3", self.cache.memory[-1].page_content)
        self.cache.add("q4", "r4")
        self.assertEqual(len(self.cache.memory), 3)
        self.assertNotIn("q1", [doc.page_content for doc in self.cache.memory])
        self.assertIn("q4", self.cache.memory[-1].page_content)

if __name__ == '__main__':
    unittest.main()