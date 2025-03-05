from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from typing import List, Dict
import json5
# 用户意图识别 & 上下文整理 & 查询重写
re_write_prompt=PromptTemplate.from_template("""
            你是一个专业的信息检索专家。当前我们的任务是重写用户的查询，
            使其更适合用于检索一个包含结构化知识数据的知识库。
            
            请确保重写后的查询具备以下特点：
            1. 关键词增强(Keyword Boosting):提取问题中的核心实体和术语，增加权重或补充同义词；
            ---
            示例：原问题： "如何解决程序运行慢" ； "重写后： "优化Python代码运行速度的方法（性能调优、算法复杂度、多线程）"
            ---
            2. 意图显式化（Intent Explicitization）: 将隐式需求转为显式查询，添加限定条件；
            ---
            示例：原问题： "苹果的最新产品" ；重写后： "苹果公司（Apple Inc.）2023年发布的消费电子产品型号及参数"
            ---
            3. 上下文补全（Context Completion）: 在不修改问题本意的前提下，为短问题添加隐含的上下文信息；
            ---
            示例： 原问题： "怎么治疗糖尿病脚烂？"；重写后： "糖尿病患者出现足部溃疡后的标准化临床治疗方案"
            ---
            4.问题分解（Query Decomposition）：将复杂问题拆分为多个原子问题；
            ---
            示例：原问题： "如何在北京开一家咖啡店并申请营业执照？" ；重写后： ["北京市餐饮行业开店选址要求；","北京市个体工商户营业执照申请流程"]
            ---
            5. 重写后的问题应该更符合信息检索的习惯，便于从知识库中找到准确的答案。
            
            用户原查询：{query}
            
            请重写这个查询并按照以下格式回答,new_query为子问题列表：
            {{"new_query": ["<new_query>"], "reason": "<reason>"}}
            """
            )
@tool
def extract_query(new_query:str,reason:str)-> str:
    """
    从用户问题中提取多个子查询关键词，用于分步检索知识库。
    
    Args:
        new_query (str): 原始用户问题，需包含多个隐含子问题。
        
    Returns:
        Dict[str, List[str]]: 包含以下键值：
            - "queries": 提取的子查询列表（如 ["子查询1", "子查询2"]）
            - "reason": 分解查询的原因说明
    
    Example:
        extract_query("如何评估机器学习的准确率和效率？")
        {'queries': ['机器学习准确率指标', '机器学习效率优化方法'], 'reason': '问题涉及两个独立评估维度'}
    """
    
    return f"""{"queries": new_query, "reason": reason}"""

llm_tool = ChatOpenAI(
            openai_api_base="https://api.siliconflow.cn/v1/",
            openai_api_key="sk-nsrodawrccxulpmixfrwcogpaotypwhtpjcnxqgfnzafhtdk",
            model_name="deepseek-ai/DeepSeek-V2.5"
).bind_tools([extract_query])

class QureyRewrite:
    def __init__(self):
        self.model = ChatOpenAI(
            openai_api_base="https://api.siliconflow.cn/v1/",
            openai_api_key="sk-nsrodawrccxulpmixfrwcogpaotypwhtpjcnxqgfnzafhtdk",
            model_name="Pro/deepseek-ai/DeepSeek-V3"
        )
        self.tool = llm_tool

    def rewrite(self, query):
        try:
            prompt = re_write_prompt.format(query=query)
            result = self.model.invoke(prompt)
            print("1.",result,"\n")
            res2 = self.tool.invoke(result.content)
            print("2.",res2,"\n")
            print("2.5",res2.tool_calls)
            extracted_data = []
            try:
                for data in res2.tool_calls:
                    print("3.",data['args'],"\n")
                    if isinstance(data['args'], str):
                        args_dict = json5.loads(data['args'])
                    else:
                        args_dict = data['args']  # 已经是字典
                    extracted_data.append({
                        "query" : args_dict.get("new_query"),
                        "reason" : args_dict.get("reason")
                    })
                print("3.",extracted_data,"\n")
            except Exception as e:
                print(f"json解析错误: {str(e)}")
            return extracted_data
        except Exception as e:
            print(f"未知错误: {str(e)}")
            
if __name__ == "__main__":
    qr = QureyRewrite()
    res = qr.rewrite("查询全球变暖的影响")
    print(res)
