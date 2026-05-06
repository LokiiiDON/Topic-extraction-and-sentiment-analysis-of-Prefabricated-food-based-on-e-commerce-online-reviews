import pandas as pd
from snownlp import SnowNLP

# 1. 读取 Excel 文件
df = pd.read_excel(r"C:\Users\lorrianedon\Desktop\comments_segmented.xlsx")  # 替换为你的文件名

# 2. 提取分词结果
words_list = df["分词结果"].tolist()  # 替换为实际的列名

# 3. 情感分析
sentiment_scores = []
for words in words_list:
    if isinstance(words, str) and words.strip():  # 确保是字符串且非空
        s = SnowNLP(words)
        sentiment_scores.append(s.sentiments)  # 情感得分范围 0-1
    else:
        sentiment_scores.append(None)  # 空值或无效值标记为 None

# 4. 将情感得分添加到 DataFrame 中
df["情感得分"] = sentiment_scores

# 5. 保存结果到新的 Excel 文件
df.to_excel("情感分析结果2.xlsx", index=False)
