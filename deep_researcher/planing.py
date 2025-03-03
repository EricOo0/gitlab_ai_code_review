from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
import json
from typing import Dict
evaluationPrompt=PromptTemplate.from_template( """
You are an evaluator that determines if a question requires freshness, plurality, and/or completeness checks.

<evaluation_types>
1. freshness - Checks if the question is time-sensitive or requires very recent information
2. plurality - Checks if the question asks for multiple items, examples, or a specific count or enumeration
3. completeness - Checks if the question explicitly mentions multiple named elements that all need to be addressed
</evaluation_types>

<rules>
1. Freshness Evaluation:
   - Required for questions about current state, recent events, or time-sensitive information
   - Required for: prices, versions, leadership positions, status updates
   - Look for terms: "current", "latest", "recent", "now", "today", "new"
   - Consider company positions, product versions, market data time-sensitive

2. Plurality Evaluation:
   - ONLY apply when completeness check is NOT triggered
   - Required when question asks for multiple examples, items, or specific counts
   - Check for: numbers ("5 examples"), list requests ("list the ways"), enumeration requests
   - Look for: "examples", "list", "enumerate", "ways to", "methods for", "several"
   - Focus on requests for QUANTITY of items or examples

3. Completeness Evaluation:
   - Takes precedence over plurality check - if completeness applies, set plurality to false
   - Required when question EXPLICITLY mentions multiple named elements that all need to be addressed
   - This includes:
     * Named aspects or dimensions: "economic, social, and environmental factors"
     * Named entities: "Apple, Microsoft, and Google", "Biden and Trump"
     * Named products: "iPhone 15 and Samsung Galaxy S24"
     * Named locations: "New York, Paris, and Tokyo"
     * Named time periods: "Renaissance and Industrial Revolution"
   - Look for explicitly named elements separated by commas, "and", "or", bullets
   - Example patterns: "comparing X and Y", "differences between A, B, and C", "both P and Q"
   - DO NOT trigger for elements that aren't specifically named   
</rules>

<examples>
<example-1>
谁发明了微积分？牛顿和莱布尼兹各自的贡献是什么？
<think>
这是关于微积分历史的问题，不需要最新信息。问题特别提到了牛顿和莱布尼兹两个人，要求分析他们各自的贡献，所以我需要全面回答这两部分内容。完整性比较重要，而不是提供多个不同答案。
</think>
<output>
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-1>

<example-2>
fam PLEASE help me calculate the eigenvalues of this 4x4 matrix ASAP!! [matrix details] got an exam tmrw 😭
<think>
This is a math question about eigenvalues which doesn't change over time, so I don't need fresh info. A 4x4 matrix has multiple eigenvalues, so I'll need to provide several results. The student just wants the eigenvalues calculated, not asking me to address multiple specific topics.
</think>
<output>
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-2>

<example-3>
Quelles sont les principales différences entre le romantisme et le réalisme dans la littérature du 19ème siècle?
<output>
<think>
C'est une question sur l'histoire littéraire, donc je n'ai pas besoin d'informations récentes. Je dois comparer deux mouvements spécifiques: le romantisme et le réalisme. Ma réponse doit couvrir ces deux éléments, donc l'exhaustivité est importante ici. La pluralité n'est pas la priorité dans ce cas.
</think>
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-3>

<example-4>
Shakespeare の最も有名な悲劇を5つ挙げ、簡単にあらすじを説明してください。
<think>
シェイクスピアの悲劇についての質問だから、最新情報は必要ないな。「5つ挙げ」とはっきり書いてあるから、複数の回答が必要だ。どの悲劇を選ぶかは私次第で、特定の作品について比較するよう求められているわけじゃないから、完全性よりも複数性が重要だな。
</think>
<output>
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-4>

<example-5>
What are the current interest rates for mortgage loans from Bank of America, Wells Fargo, and Chase Bank in the US?
<think>
This is asking about 'current' interest rates, so I definitely need up-to-date info. The person wants rates from three specific banks: Bank of America, Wells Fargo, and Chase. I need to cover all three to properly answer, so addressing these specific elements is more important than providing multiple different answers.
</think>
<output>
"needsFreshness": true,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-5>

<example-6>
2025年に注目すべき人工知能の3つのトレンドは何ですか？
<think>
これは将来のAIトレンドについての質問だから、最新の情報が必要だね。「3つの」と明確に数を指定しているから、複数の回答が求められている。特定のトレンドについて詳しく説明するというより、重要なトレンドを3つ挙げることが大事そうだから、複数性の方が完全性より重要だな。
</think>
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-6>

<example-7>
Was sind die besten Strategien für nachhaltiges Investieren in der heutigen Wirtschaft?
<think>
Hier geht's um Investieren in der 'heutigen Wirtschaft', also brauche ich aktuelle Informationen. Die Frage ist nach 'Strategien' im Plural gestellt, daher sollte ich mehrere Beispiele nennen. Es werden keine bestimmten Aspekte genannt, die ich alle behandeln muss - ich soll einfach verschiedene gute Strategien vorschlagen. Aktualität und mehrere Antworten sind hier wichtig.
</think>
<output>
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-7>

<example-8>
请解释赤壁之战的历史背景、主要参与者以及战略意义，这对中国历史产生了什么影响？
<think>
这是关于历史事件的问题，不需要最新信息。问题清楚地列出了几个需要我回答的方面：历史背景、主要参与者、战略意义和历史影响。我需要涵盖所有这些特定方面，而不是提供多个不同的答案。这里完整性比复数性更重要。
</think>
<output>
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-8>
</examples>
user:{query}
""")
prompt=PromptTemplate.from_template(""" 
You are an expert search query generator with deep psychological understanding. 
You optimize user queries by extensively analyzing potential user intents and
generating comprehensive search subquery.

<rules>
1. Start with deep intent analysis:
   - Direct intent (what they explicitly ask)
   - Implicit intent (what they might actually want)
   - Related intents (what they might need next)
   - Prerequisite knowledge (what they need to know first)
   - Common pitfalls (what they should avoid)
   - Expert perspectives (what professionals would search for)
   - Beginner needs (what newcomers might miss)
   - Alternative approaches (different ways to solve the problem)

2. For each identified intent:
   - Generate queries in original language
   - Generate queries in English (if not original)
   - Generate queries in most authoritative language
   - Use appropriate operators and filters

3. Query structure rules:
   - Use exact match quotes for specific phrases
   - Split queries for distinct aspects
   - Add operators only when necessary
   - Ensure each query targets a specific intent
   - Remove fluff words but preserve crucial qualifiers

<query-operators>
A query can't only have operators; and operators can't be at the start a query;

- "phrase" : exact match for phrases
- +term : must include term; for critical terms that must appear
- -term : exclude term; exclude irrelevant or ambiguous terms
- filetype:pdf/doc : specific file type
- site:example.com : limit to specific site
- lang:xx : language filter (ISO 639-1 code)
- loc:xx : location filter (ISO 3166-1 code)
- intitle:term : term must be in title
- inbody:term : term must be in body text
</query-operators>

</rules>

<examples>
<example-1>
Input Query: 宝马二手车价格
<think>
让我以用户的角度思考...

我在查询宝马二手车价格，但我内心真正关注的是什么？

主要顾虑：
- 我想买宝马是因为它代表身份地位，但我担心负担能力
- 我不想因为买了一辆无法维护的旧豪车而显得愚蠢
- 我需要知道我是否得到了好价格或被骗
- 我担心购买后出现昂贵的意外支出

更深层次的焦虑：
- 我真的能负担得起维修保养费用吗？
- 人们会因为我买了旧宝马而不是新的普通车而评判我吗？
- 如果我陷入困境怎么办？
- 我对车的知识足够应对这种情况吗？

专业级考量：
- 哪些型号有众所周知的问题？
- 除了购买价格外，真正的拥有成本是多少？
- 谈判的关键点在哪里？
- 机械师在这些特定型号中会关注什么？

关于多语言扩展的思考：
- 宝马是德国品牌，德语搜索可能提供更专业的维修和问题信息
- 英语搜索可能有更广泛的全球用户体验和价格比较
- 保留中文搜索针对本地市场情况和价格区间
- 多语言搜索能够获取不同文化视角下的二手宝马评价
</think>
queries: [
  "宝马 二手车 价格区间 评估 lang:zh",
  "宝马 各系列 保值率 对比",
  "二手宝马 维修成本 真实体验",
  "买二手宝马 后悔 经历",
  "二手宝马 月收入 工资要求",
  "修宝马 坑 避免",
  "BMW used car price guide comparison",
  "BMW maintenance costs by model year",
  "living with used BMW reality",
  "BMW ownership regret stories",
  "expensive BMW repair nightmares avoid",
  "BMW versus new Toyota financial comparison",
  "BMW Gebrauchtwagen Preisanalyse lang:de",
  "BMW Langzeitqualität Erfahrung",
  "BMW Werkstatt Horror Geschichten",
  "BMW Gebrauchtwagen versteckte Kosten"
]
</example-1>

<example-2>
Input Query: Python Django authentication best practices
<think>
Let me think as the user seeking Django authentication best practices...

Surface-level request:
- I'm looking for standard Django authentication practices
- I want to implement "best practices" for my project
- I need technical guidance on secure authentication

Deeper professional concerns:
- I don't want to mess up security and get blamed for a breach
- I'm worried my implementation isn't "professional enough"
- I need to look competent in code reviews
- I don't want to rebuild this later when we scale

Underlying anxieties:
- Am I out of my depth with security concepts?
- What if I miss something critical that leads to a vulnerability?
- How do real companies actually implement this in production?
- Will this code embarrass me when more experienced developers see it?

Expert-level considerations:
- I need to anticipate future architecture questions from senior devs
- I want to avoid common security pitfalls in authentication flows
- I need to handle edge cases I haven't thought of yet
- How do I balance security with user experience?

Reasoning for multilingual expansion:
- Although Django documentation is primarily in English, Spanish is widely spoken in many developer communities
- Security concepts might be better explained in different languages with unique perspectives
- Including queries in multiple languages will capture region-specific best practices and case studies
- Spanish or Portuguese queries might reveal Latin American enterprise implementations with different security constraints
- Language-specific forums may contain unique discussions about authentication issues not found in English sources
</think>
queries: [
  "Django authentication security best practices site:docs.djangoproject.com",
  "Django auth implementation patterns security",
  "authentication security breach postmortem",
  "how to explain authentication architecture interview",
  "authentication code review feedback examples",
  "startup authentication technical debt lessons",
  "Django auth security testing methodology",
  "Django autenticación mejores prácticas lang:es",
  "Django seguridad implementación profesional",
  "authentication mistakes junior developers",
  "when to use third party auth instead of building",
  "signs your authentication implementation is amateur",
  "authentication decisions you'll regret",
  "autenticação Django arquitetura empresarial lang:pt",
  "Django authentication scalability issues",
  "Python Django Authentifizierung Sicherheit lang:de"
]
</example-2>

<example-3>
Input Query: KIリテラシー向上させる方法
<think>
ユーザーとしての私の考えを整理してみます...

表面的な質問：
- AIリテラシーを高める方法を知りたい
- 最新のAI技術について学びたい
- AIツールをより効果的に使いたい

本当の関心事：
- 私はAIの急速な発展についていけていないのではないか
- 職場でAIに関する会話に参加できず取り残されている
- AIが私の仕事を奪うのではないかと不安
- AIを使いこなせないと将来的に不利になる

潜在的な懸念：
- どこから学び始めればいいのか分からない
- 専門用語が多すぎて理解するのが難しい
- 学んでも技術の進化に追いつけないのでは？
- 実践的なスキルと理論的な知識のバランスはどうすべき？

専門家レベルの考慮点：
- AIの倫理的問題をどう理解すべきか
- AIの限界と可能性を実践的に評価する方法
- 業界別のAI応用事例をどう学ぶべきか
- 技術的な深さと広範な概要知識のどちらを優先すべきか

多言語拡張に関する考察：
- AIは国際的な分野であり、英語の情報源が最も豊富なため英語の検索は不可欠
- AIの発展はアメリカと中国が主導しているため、中国語の資料も参考になる
- ドイツはAI倫理に関する議論が進んでいるため、倫理面ではドイツ語の情報も有用
- 母国語（日本語）での検索は理解の深さを確保するために必要
- 異なる言語圏での検索により、文化的背景の異なるAI活用事例を把握できる
</think>
queries: [
  "AI リテラシー 初心者 ロードマップ",
  "人工知能 基礎知識 入門書 おすすめ",
  "AI技術 実践的活用法 具体例",
  "ChatGPT 効果的な使い方 プロンプト設計",
  "AIリテラシー 企業研修 内容",
  "AI用語 わかりやすい解説 初心者向け",
  "AI literacy roadmap for professionals",
  "artificial intelligence concepts explained simply",
  "how to stay updated with AI developments",
  "AI skills future-proof career",
  "balancing technical and ethical AI knowledge",
  "industry-specific AI applications examples",
  "人工智能 入门 学习路径 lang:zh",
  "KI Grundlagen für Berufstätige lang:de",
  "künstliche Intelligenz ethische Fragen Einführung",
  "AI literacy career development practical guide"
]
</example-3>
</examples>`,
user_query:{query}
""")


# planning model，用于拆解用户提问

class PlanningModel:
    def __init__(self):
        self.prompt = ""
        self.evaluation_prompt = ""
        self.model = ChatOpenAI(
            openai_api_base="https://api.siliconflow.cn/v1/",
            openai_api_key="",
            model_name="deepseek-ai/DeepSeek-R1"
        )

    def parse_evaluation_output(self,content: str) -> Dict[str, bool]:
        """解析包含 <output> 标签的结构化数据"""

        # 提取 <output> 标签内的内容
        output_match = re.search(r'<output>\n(.*?)\n</output>', content, re.DOTALL)
        if not output_match:
            raise ValueError("未找到有效的 <output> 标签内容")

        # 清理数据并转换为合法 JSON 格式
        raw_output = output_match.group(1)
        cleaned = (
            raw_output
                .strip()  # 去除首尾空白
                .rstrip(',')  # 去除末尾逗号
                .replace("'", '"')  # 统一引号格式
                .replace('\\', '\\\\')  # 处理转义字符
        )

        # 转换为字典
        try:
            parsed = json.loads(f'{{{cleaned}}}')  # 包裹大括号构成完整 JSON 对象
            return {k: bool(v) if isinstance(v, str) else v for k, v in parsed.items()}
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {str(e)}")

    def parse_further_search_output(self,content: str) -> list:
        """从内容中提取所有查询语句"""

        # 匹配 queries 数组部分
        pattern = r'queries:\s*\[\s*((?:"[^"]+",?\s*)+)\s*\]'
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            return []

        # 提取数组内容并分割条目
        queries_str = match.group(1)
        queries = re.findall(r'"([^"]+)"', queries_str)

        return queries

    def invoke(self,query):
        try:
            self.evaluation_prompt = evaluationPrompt.format(query=query)
            self.prompt = prompt.format(query=query)

            result = self.model.invoke(self.evaluation_prompt)
            res = self.parse_evaluation_output(result.content)
            print("1.",res,"\n")
            result = self.model.invoke(self.prompt)
            r = self.parse_further_search_output(result.content)
            print(r)
        except Exception as e:
            print(f"未知错误: {str(e)}")


if __name__ == "__main__":
    # action 定义
    # plan , search，answer，reflect，
    model = PlanningModel()
    model.invoke("deep research的基本架构是什么")
    # choose action