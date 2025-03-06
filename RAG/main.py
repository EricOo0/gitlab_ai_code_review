from query_rewrite import QureyRewrite
from milvus import VectorStore
from summary import Summarizer

if __name__=="__main__":
    # 1.指令改写
    query = "ai code review 怎么实现"
    qr = QureyRewrite()
    res = qr.rewrite(query)
    print(res)
    
    # 向量召回
    vs = VectorStore()
    vs.create_collection("demo_collection")
    doc = []
    for i,r in enumerate(res): 
        new_query = r.get("query")
        print("num:",i,new_query)
        ret_doc = vs.query("demo_collection", new_query)
        doc.append(ret_doc)
    print(doc)
    # todo 3. rerank
    
    # 4. 总结 summary
    
    sumarizer = Summarizer()
    summary = sumarizer.summarize(query,res,doc)
    print(summary)
    # 5. 提问
        