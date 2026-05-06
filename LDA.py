import pandas as pd
import jieba
import math
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel, CoherenceModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimSun']  # Windows系统使用
plt.rcParams['axes.unicode_minus'] = False


def chinese_word_cut(text):
    return [word for word in jieba.cut(text) if len(word.strip()) > 1]


def calculate_metrics(ldamodel, testset, dictionary, num_topics, texts):
    perplexity = docperplexity(ldamodel, testset, dictionary, num_topics)

    coherence_model = CoherenceModel(
        model=ldamodel,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v',
        processes=1  # 强制使用单进程
    )
    coherence = coherence_model.get_coherence()

    return perplexity, coherence


def docperplexity(ldamodel, testset, dictionary, num_topics):
    """优化后的困惑度计算函数"""
    # 准备主题-词分布
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_words = ldamodel.get_topic_terms(topic_id, topn=len(dictionary))
        word_probs = {dictionary[id]: prob for id, prob in topic_words}
        topic_word_list.append(word_probs)

    # 准备文档-主题分布
    doc_topics_list = [ldamodel.get_document_topics(doc, minimum_probability=0) for doc in testset]

    # 计算困惑度
    prob_doc_sum = 0.0
    testset_word_num = 0

    # 带进度条的文档处理
    for doc_idx in tqdm(range(len(testset)), desc=f'主题数 {num_topics}', leave=False):
        doc = testset[doc_idx]
        doc_prob = 0.0
        doc_word_count = 0

        # 获取文档主题分布向量
        topic_dist = dict(doc_topics_list[doc_idx])

        for word_id, count in doc:
            word = dictionary[word_id]
            prob_word = 0.0
            for topic_id in range(num_topics):
                prob_topic = topic_dist.get(topic_id, 0.0)
                prob_word += prob_topic * topic_word_list[topic_id].get(word, 0.0)

            # 添加平滑处理避免零概率
            prob_word = max(prob_word, 1e-10)
            doc_prob += math.log(prob_word) * count
            doc_word_count += count

        prob_doc_sum += doc_prob
        testset_word_num += doc_word_count

    return math.exp(-prob_doc_sum / testset_word_num) if testset_word_num else float('inf')


if __name__ == '__main__':
    # Windows多进程必须的修复
    multiprocessing.freeze_support()

    # 数据预处理流程
    df = pd.read_excel(r'E:\pycharm\test\.venv\情感倾向统计表2.xlsx')
    df['content_cutted'] = df['评论内容'].apply(chinese_word_cut)

    # 创建词典和语料库
    documents = df['content_cutted'].tolist()
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=5, no_above=0.5)  # 调整过滤阈值
    corpus_bow = [dictionary.doc2bow(doc) for doc in documents]

    # 应用TF-IDF转换
    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    # 划分训练测试集
    train_corpus, test_corpus = train_test_split(corpus_tfidf, test_size=0.2, random_state=42)

    # 实验参数设置
    topic_range = range(2, 9)
    results = []

    # 带指标计算的训练循环
    with tqdm(total=len(topic_range), desc='模型训练进度') as pbar:
        for num_topics in topic_range:
            # 训练LDA模型
            lda_model = LdaModel(
                corpus=train_corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=5,
                iterations=50,
                random_state=42,
                chunksize=1000,  # 增加块大小减少进程创建
                alpha='auto'  # 自动调整alpha参数
            )

            # 计算指标
            perplexity, coherence = calculate_metrics(lda_model, test_corpus, dictionary, num_topics, documents)
            results.append({
                'topics': num_topics,
                'perplexity': perplexity,
                'coherence': coherence
            })

            pbar.update(1)

    # 创建可视化图表
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 绘制困惑度曲线
    color = 'tab:blue'
    ax1.set_xlabel('主题个数', fontsize=14)
    ax1.set_ylabel('困惑度', color=color, fontsize=14)
    perplexity_line = ax1.plot(
        [r['topics'] for r in results],
        [r['perplexity'] for r in results],
        color=color,
        marker='o',
        linewidth=2,
        label='困惑度'
    )
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(topic_range)

    # 创建第二坐标轴
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('一致性 (c_v)', color=color, fontsize=14)
    coherence_line = ax2.plot(
        [r['topics'] for r in results],
        [r['coherence'] for r in results],
        color=color,
        linestyle='--',
        marker='s',
        linewidth=2,
        label='一致性分数'
    )
    ax2.tick_params(axis='y', labelcolor=color)

    # 合并图例
    lines = perplexity_line + coherence_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=12)


    plt.title('LDA模型困惑度与一致性分数随主题个数变化图', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig('combined_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

# 输出格式化结果
result_df = pd.DataFrame(results)
print("\n评估结果汇总：")
print(result_df.round({'perplexity': 1, 'coherence': 3}).to_string(index=False))
