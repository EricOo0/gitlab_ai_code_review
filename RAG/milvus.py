# 向量数据库使用milvus pip install -U pymilvus

from pymilvus import MilvusClient,DataType,FieldSchema, CollectionSchema
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

class VectorStore:
    def __init__(self):
        self.client = MilvusClient("milvus_demo.db")
        self.embedding_model = SentenceTransformer("all-MiniLM-L12-v2") # 384维 ；使用小模型进行embedding，可更换其他 效果更好
        self.dim = 384

    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        return self.client.has_collection(collection_name=collection_name)
    
    def drop_collection(self, collection_name: str):
        """删除集合"""
        if self.collection_exists(collection_name):
            self.client.drop_collection(collection_name=collection_name)

    # 向量数据库中collection 类比 db 中的表
    def create_collection(self, collection_name):
        if self.collection_exists(collection_name):
            self.drop_collection(collection_name)
            
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
        data_field = FieldSchema(name="doc", dtype=DataType.VARCHAR, description="doc",max_length=65535)  # 增加最大长度
        embedding_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim, description="vector")
        schema = CollectionSchema(fields=[id_field,data_field, embedding_field], auto_id=True, enable_dynamic_field=True, description="desc of a collection")
        self.client.create_collection(
                collection_name=collection_name,
                dimension=self.dim,
                schema=schema
            )
        
        index_params = self.client.prepare_index_params()
        index_params.add_index("vector", "", "", metric_type="IP")
        self.client.create_index(collection_name, index_params)
        
    def query(self,collection, query):
        # 使用小模型进行embedding，可更换其他 效果更好
        embedding = self.embedding_model.encode(query)
        res = self.client.search(
            collection_name=collection,     # 目标集合
            data=[embedding],                # 查询向量
            limit=3,                           # 返回的实体数量
            anns_field="vector",
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["doc"]
        )
        docs = []
        for hits in res:  # 每个查询对应的结果列表
                for hit in hits:
                    entity = hit.get("entity")
                    doc = entity.get("doc") if entity else None
                    if doc:
                        docs.append({"content": doc})  # 修改返回格式以匹配重排序需求
        return docs
        
    def insert(self, collection: str, documents: List[Dict[str, Any]]):
        """
        批量插入文档
        Args:
            collection: 集合名称
            documents: 文档列表，每个文档是一个字典，包含 content 字段
        """
        try:
            # 提取所有文档内容
            contents = [doc.get('content', '') for doc in documents]
            
            # 批量生成向量
            embeddings = self.embedding_model.encode(contents)
            
            # 准备插入数据
            data = []
            for i, content in enumerate(contents):
                data.append({
                    "vector": embeddings[i],  # 保持为 numpy 数组
                    "doc": content
                })
            
            # 执行插入
            res = self.client.insert(
                collection_name=collection,
                data=data
            )
            print(f"成功插入 {len(contents)} 个文档")
            return res
            
        except Exception as e:
            print(f"插入文档时出错: {str(e)}")
            raise

if __name__=="__main__":
    vs = VectorStore()
    vs.create_collection("demo_collection")
    vs.insert("demo_collection", [{"content": "如何评估机器学习的准确率和效率？"}])
    res = vs.query("demo_collection", "如何评估机器学习的准确率和效率？")
    print(res)
    # data=[
    # {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "pink_8682"},
    # {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "color": "red_7025"},
    # {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], "color": "orange_6781"},
    # {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345], "color": "pink_9298"},
    # {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106], "color": "red_4794"},
    # {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955], "color": "yellow_4222"},
    # {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987], "color": "red_9392"},
    # {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052], "color": "grey_8510"},
    # {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336], "color": "white_9381"},
    # {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608], "color": "purple_4976"}
    # ]
    
    # # 4.2. Insert data
    # res = vs.client.insert(
    #     collection_name="demo_collection",
    #     data=data
    # )
    
    # print(res)
    # query_vectors = [
    # [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648]
    # ]
 
    # # 6.2. 开始搜索
    # res = vs.query("demo_collection", query_vectors)
    
    # print(res)

