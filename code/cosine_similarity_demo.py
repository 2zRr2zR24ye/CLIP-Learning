# 导入依赖库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 准备文本数据（比如用户查询和文档库）
corpus = [
    "I like eating apple",  # 文档1
    "apple is healthy",  # 文档2
    "apple is sweeter than banana",  # 文档3
    "today is nice"  # 文档4（无关文档）
]

# 2. 用TF-IDF将文本转成向量（捕捉词的重要性）
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())  # 按空格分词
tfidf_matrix = vectorizer.fit_transform(corpus)

# 3. 计算查询文本（比如“苹果好吃”）与所有文档的余弦相似度
query = ["apple"]
query_vector = vectorizer.transform(query)
similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

# 4. 输出结果（按相似度从高到低排序）
for i, score in enumerate(similarity_scores[0]):
    print(f"文档{i+1}：相似度={score:.4f}")


"""结果：文档1：相似度=0.3458
文档2：相似度=0.4738
文档3：相似度=0.3268
文档4：相似度=0.0000"""