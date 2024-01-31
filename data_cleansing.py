import os
import re

# 定义数据文件夹路径
data_folder = 'data'
contents = []  # 存储对应的文本

# 去除特殊字符和标点符号
def clean_text(text):
    # 删除http 的网址,直到遇到空格
    text = re.sub(r'http\S+', '', text)
    # 删除形如 RT @username: 的内容
    text = re.sub(r'RT @\w+:', '', text)

    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()


# 遍历文件夹中的文本文件
for filename in os.listdir(data_folder):
    if filename.endswith('.txt'):
        try:
            file_path = os.path.join(data_folder, filename)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                print(filename + "open with utf-8")
                original_text = file.read()

            # 数据处理
            cleaned_text = clean_text(original_text)
            contents.append(cleaned_text)

            # 保存处理后的文本回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
        except:
            file_path = os.path.join(data_folder, filename)

            # 读取文件内容
            with open(file_path, 'r', encoding='GBK') as file:
                print(filename + "open with gbk")
                original_text = file.read()

            # 数据处理
            cleaned_text = clean_text(original_text)
            contents.append(cleaned_text)

            # 保存处理后的文本回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)

all_data_path = "all_data.txt"
with open(all_data_path, 'a', encoding='utf-8') as f:
    for data in contents:
        f.write(data)

print("文本处理完成并保存回文件。")
