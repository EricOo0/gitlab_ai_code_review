# rerank 一般策略
# 1. 大模型rerank
# 2. cohere 模型 交叉熵重排；bge reanker模型


class Reranker:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def rerank(self, query, docs):
        inputs = self.tokenizer(query, docs, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return