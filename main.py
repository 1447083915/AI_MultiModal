import os
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_text_length = 64
# 创建多模态模型实例
num_classes = 3  # 你的标签类别数
vocab_size = 15258
embedding_dim = 64
hidden_size = 128

# 设置数据集路径和训练文件路径
data_folder = 'data'
train_txt_file = 'train.txt'
test_txt_file = 'test_without_label.txt'
dict_file = 'dict.txt'

class CustomDataset(Dataset):
    def __init__(self, data_folder, txt_file, dict_file, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.max_text_length = max_text_length

        # 读取加载字典
        with open(dict_file, 'r', encoding='utf-8') as dict_file:
            self.word_index_mapping = eval(dict_file.read())

        with open(txt_file, 'r') as file:
            lines = file.readlines()[1:]
            self.data = [line.strip().split(',') for line in lines]
            # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid, tag = self.data[idx]

        image_path = os.path.join(self.data_folder, f"{guid}.jpg")
        text_path = os.path.join(self.data_folder, f"{guid}.txt")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Load text data
        with open(text_path, 'r', encoding='GBK', errors='replace') as text_file:
            text_data = text_file.read().strip()

        # 文本的映射
        unmapped_words = []
        # text_data_indices = [self.word_index_mapping.get(char, self.word_index_mapping['<unk>']) for char in text_data]
        text_data_indices = [self.word_index_mapping.get(char, self.word_index_mapping['<unk>']) for char in text_data
                             if char != ' ']

        # # 检查映射过程中未成功映射的词
        # for char, index in zip(text_data, text_data_indices):
        #     if index == self.word_index_mapping['<unk>']:
        #         unmapped_words.append(char)
        #
        # # 输出未成功映射的词
        # print("Unmapped Words:", unmapped_words)

        # 截断或填充文本数据
        if len(text_data_indices) < self.max_text_length:
            # 填充
            text_data_indices += [0] * (self.max_text_length - len(text_data_indices))
        elif len(text_data_indices) > self.max_text_length:
            # 截断
            text_data_indices = text_data_indices[:self.max_text_length]

        # 转化为tensor
        text_data_tensor = torch.tensor(text_data_indices, dtype=torch.long)

        # 三种tag的映射
        tag_mapping = {'null': -1, 'negative': 0, 'neutral': 1, 'positive': 2}
        tag = tag_mapping[tag]

        sample = {'image': image, 'text': text_data_tensor, 'tag': tag}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


# 图像模态处理
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # 移除原始的全连接层

    def forward(self, x):
        return self.resnet(x)


# 文本模态处理
class TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        # return output[:, -1, :]  # 使用最后一个时间步的输出作为文本特征
        return output[:, -1, :hidden_size] + output[:, 0, hidden_size:]


# 多模态模型
class MultiModalModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, use_image=True, use_text=True):
        super(MultiModalModel, self).__init__()
        self.use_image = use_image
        self.use_text = use_text

        # 对是否使用文本和图像进行不同的操作
        if self.use_image:
            self.image_model = ImageModel()

        if self.use_text:
            self.text_model = TextModel(vocab_size, embedding_dim, hidden_size)

        if self.use_image and self.use_text:
            self.fc = nn.Linear(512 + hidden_size, num_classes)  # 512是ResNet18的输出特征维度
        elif self.use_image:
            self.fc = nn.Linear(512, num_classes)
        elif self.use_text:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, image, text):
        # 对是否使用文本和图像进行不同的操作
        if self.use_image:
            image_features = self.image_model(image)
        else:
            image_features = torch.tensor([]).to(image.device)

        if self.use_text:
            text_features = self.text_model(text)
        else:
            text_features = torch.tensor([]).to(text.device)

        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.fc(combined_features)
        return output


# 定义数据转换（可根据需要进行修改）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集实例
train_dataset = CustomDataset(data_folder, train_txt_file, dict_file, transform=transform)
val_dataset = CustomDataset(data_folder, test_txt_file, dict_file, transform=transform)


# # 划分数据集
# train_dataset, val_dataset = torch.utils.data.random_split(dataset,
#                                                            [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 打印数据集大小和划分情况
print(f"train samples: {len(train_dataset)}")
print(f"val samples: {len(val_dataset)}")


model = MultiModalModel(vocab_size, embedding_dim, hidden_size, num_classes).to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 10

# 初始化列表记录每个epoch的耗时,loss,以及正确率
train_cost = []
loss_list = []
val_loss_list = []
accuracy_list = []

start = time.time()
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        labels = batch['tag'].to(device)

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(epoch)


# 模型评估
model.eval()
predictions = []
with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        texts = batch['text'].to(device)

        outputs = model(images, texts)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())


# 将预测结果写入 test_without_label.txt
with open(test_txt_file, 'r') as file:
    lines = file.readlines()
    with open('test_with_prediction.txt', 'w') as output_file:
        output_file.write(lines[0].strip().replace(',tag', '') + ',tag\n')  # 写入标题行
        for line, prediction in zip(lines[1:], predictions):
            tag = 'negative' if prediction == 0 else 'neutral' if prediction == 1 else 'positive'# 映射预测结果
            output_file.write(line.strip().replace(',null', '') + f',{tag}\n')

print("Predictions written to test_with_prediction.txt")