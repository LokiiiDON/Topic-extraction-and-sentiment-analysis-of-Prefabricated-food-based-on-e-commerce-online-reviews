import pandas as pd
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt

def main():
    # 读取 Excel 文件
    file_path = r"E:\pycharm\test\.venv\情感倾向统计表2.xlsx"
    df = pd.read_excel(file_path)

    # 清洗数据
    df["分词结果"] = df["分词结果"].astype(str)  # 将列转换为字符串类型
    df["分词结果"] = df["分词结果"].replace("nan", "")  # 将 "nan" 替换为空字符串

    # 将分词结果转换为列表
    corpus = df["分词结果"].tolist()

    # 过滤掉空字符串
    corpus = [doc for doc in corpus if doc.strip()]

    # 将分词数据转换为 gensim 需要的格式
    texts = [doc.split() for doc in corpus]  # 将每条评论分词
    dictionary = Dictionary(texts)
    corpus_gensim = [dictionary.doc2bow(text) for text in texts]

    # 计算不同主题数下的一致性分数
    coherence_scores = []
    topic_range = range(2, 10)  # 主题数范围
    for n_topics in topic_range:
        lda = LdaModel(corpus_gensim, num_topics=n_topics, id2word=dictionary, random_state=42)
        coherence_model = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(coherence_model.get_coherence())

    # 打印一致性分数
    for n_topics, score in zip(topic_range, coherence_scores):
        print(f"Number of Topics: {n_topics}, Coherence Score: {score}")

    # 绘制一致性分数曲线
    plt.plot(topic_range, coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Coherence Score vs Number of Topics')
    plt.show()

if __name__ == "__main__":
    main()
