

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

summary_prompt=PromptTemplate.from_template("""
你是一个高效的摘要生成代理，专注于整合用户的查询和背景信息。请根据下面的内容生成一个综合摘要，摘要需要具备以下特点：
1. 概括用户的主要查询意图和需求；
2. 融合重写后的查询中更明确的检索意图；
3. 提取相关文档中的关键信息和背景知识；
4. 突出总结出对解答问题最重要的部分，帮助用户快速了解问题的核心。

以下是输入内容：
原始查询：{origin_query}

重写后的查询：{rewrite_query}

相关文档内容：
{related_docs}

请生成一个结构清晰、内容全面的摘要：                        
""")

class Summarizer:
    def __init__(self):
        self.model = ChatOpenAI(
            openai_api_base="https://api.siliconflow.cn/v1/",
            openai_api_key="sk-xxx",
            model_name="Pro/deepseek-ai/DeepSeek-V3"
        )
    def summarize(self, origin_query,rewrite_query,related_docs):
        prompt = summary_prompt.format(origin_query=origin_query,rewrite_query=rewrite_query,related_docs=related_docs)
        result = self.model.invoke(prompt)
        return result.content
        
