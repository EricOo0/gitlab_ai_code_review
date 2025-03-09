from query_rewrite import QureyRewrite
from milvus import VectorStore
from summary import Summarizer
from rerank import Reranker
from llm import LLMService
import sys
import os

def load_documents(data_dir):
    """
    从指定目录加载文档
    Args:
        data_dir: 数据目录路径
    Returns:
        文档列表
    """
    documents = []
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('.md'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        'title': filename,
                        'content': content
                    })
        return documents
    except Exception as e:
        print(f"加载文档出错: {str(e)}")
        return []

def main():
    try:
        # 1.指令改写
        query = "ai code review 怎么实现"
        print("\n=== 1. 查询改写 ===")
        qr = QureyRewrite()
        res = qr.rewrite(query)
        if not res:
            print("查询改写失败")
            return
        print("改写结果:", res)
        
        # 2. 向量召回
        print("\n=== 2. 向量召回 ===")
        try:
            vs = VectorStore()
            collection_name = "demo_collection"
            
            # 创建或重置集合
            print(f"创建新集合: {collection_name}")
            vs.create_collection(collection_name)
            
            # 加载并导入文档
            print("正在导入文档...")
            documents = load_documents("./data")
            if not documents:
                print("没有找到可导入的文档")
                return
                
            print(f"找到 {len(documents)} 个文档，正在插入...")
            vs.insert(collection_name, documents)
            print("文档导入完成")
            
            # 执行查询
            doc = []
            for i, r in enumerate(res): 
                new_query = r.get("query")
                print(f"查询 {i+1}:", new_query)
                ret_doc = vs.query(collection_name, new_query)
                if ret_doc:
                    doc.extend(ret_doc)
            if not doc:
                print("未找到相关文档")
                return
            print("召回文档数:", len(doc))
            
        except Exception as e:
            print(f"向量召回出错: {str(e)}")
            return
            
        # 3. rerank
        print("\n=== 3. 重排序 ===")
        try:
            reranker = Reranker()
            reranked_docs = reranker.rerank(query, [d.get('content', '') for d in doc])
            print("重排序结果:", reranked_docs[:3])  # 显示前3个结果
        except Exception as e:
            print(f"重排序出错: {str(e)}")
            return
            
        # 4. 总结
        print("\n=== 4. 生成摘要 ===")
        try:
            summarizer = Summarizer()
            summary = summarizer.summarize(query, res, reranked_docs[:3])
            print("摘要:", summary)
        except Exception as e:
            print(f"生成摘要出错: {str(e)}")
            return
            
        # 5. LLM 问答
        print("\n=== 5. LLM 问答 ===")
        try:
            llm = LLMService()
            response = llm.get_response(
                query=query,
                rewritten_queries=res,
                relevant_docs=reranked_docs[:3]  # 使用重排序后的前三个文档
            )
            if response:
                print("\nAI 助手回答:")
                print("-" * 50)
                print(response)
                print("-" * 50)
            else:
                print("未能获取 AI 回答")
                
        except Exception as e:
            print(f"LLM 问答出错: {str(e)}")
            return
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        return

if __name__ == "__main__":
    main()
        