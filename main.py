import re
import random
import time
from statistics import mode
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
import unicodedata
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')



# class_mapping.csvの読み込み
class_mapping_path = 'https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/raw/main/data/annotations/class_mapping.csv'
class_mapping_df = pd.read_csv(class_mapping_path)

# class_mapping辞書の作成
answer2idx = {row['answer']: row['class_id'] for _, row in class_mapping_df.iterrows()}
idx2answer = {row['class_id']: row['answer'] for _, row in class_mapping_df.iterrows()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
        'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
        'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
        'million': '1000000', 'billion': '1000000000', 'trillion': '1000000000000',
        'quadrillion': '1000000000000000', 'quintillion': '1000000000000000000',
        'sextillion': '1000000000000000000000', 'septillion': '1000000000000000000000000',
        'octillion': '1000000000000000000000000000',
        'nonillion': '1000000000000000000000000000000',
        'decillion': '1000000000000000000000000000000000',
        'undecillion': '1000000000000000000000000000000000000',
        'duodecillion': '1000000000000000000000000000000000000000',
        'tredecillion': '1000000000000000000000000000000000000000000',
        'quattuordecillion': '1000000000000000000000000000000000000000000000',
        'sexdecillion': '1000000000000000000000000000000000000000000000000',
        'septendecillion': '1000000000000000000000000000000000000000000000000000',
        'octodecillion': '1000000000000000000000000000000000000000000000000000000',
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't", "didnt": "didn't",
        "doesnt": "doesn't", "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
        "isnt": "isn't", "shouldnt": "shouldn't", "wasnt": "wasn't", "werent": "weren't",
        "wont": "won't", "wouldve": "would have", "couldve": "could have", "didve": "did have",
        "doesve": "does have", "hadve": "had have", "hasve": "has have", "havent": "haven't",
        "isve": "is have", "shouldve": "should have", "wasve": "was have", "wereve": "were have",
        "werent": "weren't",
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # カンマの前の空白を削除（誤って2回記述されていた処理を修正）
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_mode_answer(answers, ignore_unanswerable=True):
    processed_answers = [process_text(answer["answer"]) for answer in answers]

    if ignore_unanswerable:
        processed_answers = [ans for ans in processed_answers if ans.lower() != "unanswerable"]

    if not processed_answers:  # すべての回答が "unanswerable" だった場合
        return "unanswerable"

    answer_counts = Counter(processed_answers)
    return max(answer_counts, key=answer_counts.get)

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, unanswerable_skip_rate=0.3):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pd.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.unanswerable_skip_rate = unanswerable_skip_rate
        self.skip_counter = 0

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = answer2idx
        self.idx2question = {}
        self.idx2answer = idx2answer

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text(question)
            tokens = self.tokenizer.tokenize(question)
            for token in tokens:
                if token not in self.question2idx:
                    self.question2idx[token] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)


    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = process_text(self.df["question"][idx])
        question_tokens = self.tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=20)

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
            mode_answer = self.idx2answer[mode_answer_idx]

            if mode_answer.lower() == "unanswerable":
                self.skip_counter += 1
                if self.skip_counter % 3 != 0:  # 3回に2回スキップ
                    return self.__getitem__((idx + 1) % len(self))  # 次のサンプルを取得

            return image, question_tokens['input_ids'].squeeze(), torch.Tensor(answers), int(mode_answer_idx)

        else:
            return image, question_tokens['input_ids'].squeeze()

    def __len__(self):
        return len(self.df)

# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

# 3. モデルの実装
class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(input_ids=question).last_hidden_state[:, 0, :]  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x

# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer)

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

# deviceの設定
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# dataloader / model
transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

train_dataset = VQADataset(df_path="/content/drive/MyDrive/3.Develop/Practice/DL基礎/2_最終課題/VQA_competition_without_git_clone/data/train.json", image_dir="/content/data/train", transform=transform, answer=True, unanswerable_skip_rate=0.3)
test_dataset = VQADataset(df_path="/content/drive/MyDrive/3.Develop/Practice/DL基礎/2_最終課題/VQA_competition_without_git_clone/data/valid.json", image_dir="/content/data/valid", transform=val_transform, answer=False)
test_dataset.update_dict(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

# optimizer / criterion
num_epoch = 5
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# train model
for epoch in range(num_epoch):
    train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
    print(f"【{epoch + 1}/{num_epoch}】\n"
          f"train time: {train_time:.2f} [s]\n"
          f"train loss: {train_loss:.4f}\n"
          f"train acc: {train_acc:.4f}\n"
          f"train simple acc: {train_simple_acc:.4f}")

# 提出用ファイルの作成
model.eval()
submission = []
for image, question in test_loader:
    image, question = image.to(device), question.to(device)
    pred = model(image, question)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)

submission = [train_dataset.idx2answer[id] for id in submission]
submission = np.array(submission)
torch.save(model.state_dict(), f"/content/drive/MyDrive/3.Develop/Practice/DL基礎/2_最終課題/VQA_competition_without_git_clone/output/正解率:{train_acc:.4f}_学習回数:{num_epoch}_学習率:{lr}_model.pth")
np.save(f"/content/drive/MyDrive/3.Develop/Practice/DL基礎/2_最終課題/VQA_competition_without_git_clone/output/正解率:{train_acc:.4f}_学習回数:{15}_学習率:{lr}_submission.npy", submission)