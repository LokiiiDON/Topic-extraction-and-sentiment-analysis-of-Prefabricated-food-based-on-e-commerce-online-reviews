import pandas as pd
import jieba
import gensim
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models import CoherenceModel

# 配置路径
EXCEL_PATH = r"E:\pycharm\test\.venv\情感倾向统计表2.xlsx"
STOPWORDS_PATH = "C:/Users/lorrianedon/Desktop/stopwords.txt"


def load_and_clean():
    """数据加载与清洗"""
    df = pd.read_excel(EXCEL_PATH)
    comments = df['分词结果'].dropna().astype(str).tolist()

    # 加载停用词
    try:
        with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
    except:
        stopwords = set()

    # 清洗规则
    texts = []
    for comment in comments:
        words = [
            word for word in jieba.lcut(comment)
            if (word not in stopwords
                and len(word) > 1
                and not word.isdigit()
                and word.isprintable())
        ]
        if len(words) >= 3:  # 丢弃词数<3的文档
            texts.append(words)

    # 合并短文本（如果平均长度<5）
    if np.mean([len(doc) for doc in texts]) < 5:
        texts = [' '.join(texts[i:i + 3]) for i in range(0, len(texts), 3)]
        texts = [doc.split() for doc in texts]

    return texts


def train_and_evaluate(data_set):
    # 构建词典
    dictionary = corpora.Dictionary(data_set)
    dictionary.filter_extremes(no_below=5, no_above=0.7)  # 严格过滤

    # 转换为词袋
    corpus = [dictionary.doc2bow(text) for text in data_set]
    print(f"有效文档: {len(data_set)}, 词汇量: {len(dictionary)}")

    # 训练不同主题数的模型
    results = []
    for n_topics in range(2, 9):
        lda = models.LdaModel(
            corpus=corpus,
            num_topics=n_topics,
            id2word=dictionary,
            passes=30,
            alpha='asymmetric',  # 更适合短文本
            eta=0.01,  # 防止主题过于集中
            random_state=42,
            per_word_topics=True  # 更精确的困惑度计算
        )

        # 计算评估指标
        perplexity = np.exp2(-lda.log_perplexity(corpus))
        coherence = CoherenceModel(
            model=lda, texts=data_set,
            dictionary=dictionary, coherence='c_v'
        ).get_coherence()

        results.append((n_topics, perplexity, coherence))
        print(f"主题数: {n_topics:2d} | 困惑度: {perplexity:7.1f} | 一致性: {coherence:.3f}")

    return results, dictionary, corpus


def find_elbow_point(perplexities):
    """自动确定最佳主题数（肘部法则）"""
    diffs = np.diff(perplexities)
    elbow_point = np.argmin(diffs) + 2  # +2因为从2个主题开始
    return elbow_point


# 修改plot_results函数（其他部分保持不变）
def plot_results(results):
    """可视化优化版（统一坐标轴方向）"""
    plt.figure(figsize=(12, 5))

    x = [r[0] for r in results]
    perplexities = [r[1] for r in results]
    coherences = [r[2] for r in results]

    # 困惑度曲线（左图）
    plt.subplot(1, 2, 1)
    plt.plot(x, perplexities, 'bo-')
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")  # 移除↓符号

    # 标记肘部点（自动选择最低点）
    elbow = np.argmin(perplexities) + 2
    plt.scatter(elbow, perplexities[elbow - 2], c='red', s=100)
    plt.annotate(f'Best Fit: {elbow} topics',
                 xy=(elbow, perplexities[elbow - 2]),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))

    # 一致性曲线（右图）
    plt.subplot(1, 2, 2)
    plt.plot(x, coherences, 'ro-')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence (c_v)")

    # 标记最高点
    best_coherence = np.argmax(coherences) + 2
    plt.scatter(best_coherence, coherences[best_coherence - 2], c='green', s=100)
    plt.annotate(f'Best Coherence: {best_coherence} topics',
                 xy=(best_coherence, coherences[best_coherence - 2]),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))

    plt.tight_layout()
    plt.show()

    return elbow, best_coherence

def final_model(data_set, dictionary, corpus, n_topics):
    """训练最终模型"""
    lda = models.LdaModel(
        corpus=corpus,
        num_topics=n_topics,
        id2word=dictionary,
        passes=20,
        alpha='auto',
        eta=0.05,
        random_state=42,
        per_word_topics=True
    )

    # 打印主题词
    print("\n=== 最终模型主题 ===")
    for topic in lda.print_topics(num_words=10):
        print(topic)

    # 合并相似主题
    similarity_matrix = lda[lda.id2word.doc2bow(list(lda.id2word.token2id.keys()))]
    merged_topics = set()
    for i in range(lda.num_topics):
        if i not in merged_topics:
            for j in range(i + 1, lda.num_topics):
                if similarity_matrix[i][j] > 0.35:  # 相似度阈值
                    merged_topics.add(j)

    valid_topics = [i for i in range(lda.num_topics) if i not in merged_topics]
    print(f"\n实际有效主题数: {len(valid_topics)}")

    return lda


if __name__ == "__main__":
    # 1. 数据准备
    data_set = load_and_clean()

    # 2. 训练评估
    results, dictionary, corpus = train_and_evaluate(data_set)

    # 3. 可视化分析
    elbow, best_coherence = plot_results(results)
    recommended_topics = min(elbow, best_coherence)
    print(f"\n推荐主题数: {recommended_topics} (肘部点: {elbow}, 最佳一致性: {best_coherence})")

    # 4. 训练最终模型
    final_lda = final_model(data_set, dictionary, corpus, recommended_topics)
