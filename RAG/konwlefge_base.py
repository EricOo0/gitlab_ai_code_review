import chromadb
from chromadb.utils import embedding_functions
import os
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # 滑动窗口，保留上下文
    return chunks


# ----------------------
# 2. 读取并处理文件
# ----------------------
def process_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 跳过子目录和非文本文件
        if not os.path.isfile(file_path) :
            continue

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # ----------------------
        # 3. 文档切割（按句子）
        # ----------------------
        chunks = sent_tokenize(text)  # 按句子分割

        # 如果句子太短，合并相邻句子（可选）
        merged_chunks = []
        current_chunk = ""
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < 500:  # 合并至多500字符
                current_chunk += " " + chunk
            else:
                merged_chunks.append(current_chunk.strip())
                current_chunk = chunk
        if current_chunk:
            merged_chunks.append(current_chunk.strip())

        # ----------------------
        # 4. 存入向量数据库
        # ----------------------
        documents = merged_chunks
        metadatas = [{
            "source_file": filename,
            "chunk_index": i,
            "total_chunks": len(merged_chunks)
        } for i in range(len(merged_chunks))]

        ids = [f"{filename}_chunk_{i}" for i in range(len(merged_chunks))]

        # 批量添加数据
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Processed {filename} -> {len(merged_chunks)} chunks")

if __name__=="__main__":
    chroma_client = chromadb.PersistentClient(path="./database")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_or_create_collection(
        name="local_knowledge",
        metadata={"hnsw:space": "cosine"},
        embedding_function=sentence_transformer_ef
    )
    #
    # ret = collection.add(
    #     documents=["this is document about 100102039", "this is wzf info doc"],
    #     metadatas=[{"style": "style1"}, {"style": "style2"}],
    #     ids=["uri9", "uri10"],
    # )
    # print(ret)
    # ret = collection.add(
    #     documents=["this is document about 100102039", "this is wzf info doc"],
    #     metadatas=[{"style": "style1"}, {"style": "style2"}],
    #     ids=["uri11", "uri12"],
    # )
    # print(ret)
    # documents = chunks
    # metadatas = [{"source": "doc1", "page": i} for i in range(len(chunks))]  # 添加元数据
    # ids = [f"doc1_{i}" for i in range(len(chunks))]
    process_files("data")
    # te = collection.get()
    # print(te)
    res = collection.query(
        query_texts=["text"],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    print("\nTop 3 results for query:")
    for doc, meta,dis in zip(res["documents"][0], res["metadatas"][0],res["distances"][0]):
        print(f"From {meta['source_file']} (Chunk {meta['chunk_index']})")
        print(f"Content: {doc[:200]}...\n")
        if dis > 0.5 :
            print("get")
        print(f"distance {dis}")

