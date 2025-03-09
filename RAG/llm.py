from typing import List, Dict, Any, Optional
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage

class LLMService:
    def __init__(self):
        """初始化 LLM 服务"""
        self.model = ChatOpenAI(
            openai_api_base="https://api.siliconflow.cn/v1/",
            openai_api_key="sk-xxxx",
            model_name="Pro/deepseek-ai/DeepSeek-V3",
            temperature=0.7,
            max_tokens=2000
        )
        
    def _construct_messages(self, 
                         query: str, 
                         rewritten_queries: List[Dict[str, str]], 
                         relevant_docs: List[str]) -> List[Dict[str, str]]:
        """
        构建消息列表
        Args:
            query: 原始查询
            rewritten_queries: 改写后的查询列表
            relevant_docs: 相关文档列表
        Returns:
            消息列表
        """
        # 构建用户消息内容
        user_content = f"""作为一个 AI 代码审查专家，请基于以下信息回答用户的问题。

用户原始问题：
{query}

问题已被分解为以下几个方面：
"""
        
        # 添加改写后的查询
        for i, q in enumerate(rewritten_queries, 1):
            user_content += f"{i}. {q.get('query', '')}\n"
            
        user_content += "\n根据知识库检索到的相关文档：\n"
        
        # 添加相关文档
        for i, doc in enumerate(relevant_docs, 1):
            user_content += f"文档 {i}:\n{doc}\n\n"
            
        user_content += """请根据以上信息，给出一个全面、专业且结构化的回答。回答应该：
1. 直接针对用户的问题
2. 包含具体的实现步骤和方法
3. 如果相关，提供代码示例或工具建议
4. 注意可行性和最佳实践
5. 使用清晰的结构和小标题组织内容

请用中文回答。
"""
        
        # 构建消息列表
        messages = [
            SystemMessage(content="你是一个专业的 AI 代码审查专家，精通各种编程语言和代码审查最佳实践。"),
            HumanMessage(content=user_content)
        ]
        
        return messages
        
    def get_response(self, 
                     query: str, 
                     rewritten_queries: List[Dict[str, str]], 
                     relevant_docs: List[str]) -> Optional[str]:
        """
        获取 LLM 回答
        Args:
            query: 原始查询
            rewritten_queries: 改写后的查询列表
            relevant_docs: 相关文档列表
        Returns:
            LLM 的回答
        """
        try:
            # 构建消息
            messages = self._construct_messages(query, rewritten_queries, relevant_docs)
            
            # 调用 API
            response = self.model.invoke(messages)
            
            # 返回回答内容
            return response.content
            
        except Exception as e:
            print(f"获取 LLM 回答时出错: {str(e)}")
            return None
            
if __name__ == "__main__":
    # 测试代码
    llm = LLMService()
    test_query = "如何实现 AI 代码审查？"
    test_rewritten = [
        {"query": "AI 代码审查的工作原理和实现方法"},
        {"query": "常见的 AI 代码审查工具比较"}
    ]
    test_docs = [
        "AI 代码审查是通过机器学习模型分析代码质量的过程...",
        "常见的 AI 代码审查工具包括 GitHub Copilot, SonarQube 等..."
    ]
    
    response = llm.get_response(test_query, test_rewritten, test_docs)
    if response:
        print("\nLLM 回答:")
        print("-" * 50)
        print(response)
        print("-" * 50)
    else:
        print("获取回答失败")
