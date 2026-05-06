import pandas as pd
import re

# 读取Excel文件
input_file = r'C:\Users\lorrianedon\Desktop\负面结果评论.xlsx'  # 示例：r'C:\Users\你的用户名\Desktop\comments2.xlsx'
output_file = 'cleaned_comments2.xlsx'
def segment_text(text):
    # 检查是否为字符串类型
    if not isinstance(text, str) or pd.isnull(text):
        return []

    # 精确模式分词
    words = jieba.lcut(text)

    # 去停用词 & 过滤单字
    return [word for word in words
            if word not in stopwords
            and len(word) > 1
            and not word.isspace()]
# 自定义清洗函数
def clean_comment(text):
    # 处理空值
    if pd.isnull(text):
        return ''

    # 压缩重复字符（如：好好好好 -> 好）
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # 至少重复3次的情况

    # 可选：去除特殊字符和空白（根据需求开启）
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)  # 去除非中文字符、数字、字母
    text = re.sub(r'\s+', ' ', text).strip()         # 压缩多个空格

    return text


# 读取数据
df = pd.read_excel(input_file)

# 数据清洗流程
print("原始数据量：", len(df))

# 1. 去重处理（保留第一条重复数据）
df.drop_duplicates(subset=['评论内容'], keep='first', inplace=True)
print("去重后数据量：", len(df))

# 2. 应用清洗函数
df['评论内容'] = df['评论内容'].apply(clean_comment)

# 3. 再次去重（处理清洗后可能产生的新重复）
df.drop_duplicates(subset=['评论内容'], keep='first', inplace=True)
print("最终数据量：", len(df))

# 保存结果
df.to_excel(output_file, index=False)
print(f"数据已清洗并保存到 {output_file}")
