# rerank 一般策略
# 1. 大模型rerank
# 2. cohere 模型 交叉熵重排；bge reanker模型

from typing import List, Union, Dict, Any
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        """
        初始化重排序器
        Args:
            model: 重排序模型
        """
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query: str, docs: List[str], top_k=3) -> List[str]:
        """
        对文档进行重排序
        Args:
            query: 查询文本
            docs: 待重排序的文档列表
            top_k: 返回前k个结果
        Returns:
            重排序后的文档列表
        """
        try:
            # 准备输入数据
            pairs = [[query, doc] for doc in docs]
            
            # 计算相关性分数
            scores = self.model.predict(pairs)
            
            # 将文档和分数配对并排序
            doc_score_pairs = list(zip(docs, scores))
            ranked_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            
            # 提取前 top_k 个文档
            ranked_docs = [doc for doc, _ in ranked_pairs[:top_k]]
            
            return ranked_docs
            
        except Exception as e:
            print(f"重排序过程中出错: {str(e)}")
            return docs[:top_k]  # 发生错误时返回前 top_k 个原始文档
    
if __name__ == "__main__":
    reranker = Reranker()
    query = "什么是天气"
    docs = ["天气预报", "天气预警", "天气变化", "天气的概念是气象变化"]
    result = reranker.rerank(query, docs)
    print("查询:", query)
    print("\n重排序结果:")
    for i, doc in enumerate(result, 1):
        print(f"{i}. {doc}")