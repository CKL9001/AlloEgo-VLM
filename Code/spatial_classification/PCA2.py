import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from tqdm import tqdm
from transformers import AutoTokenizer
import os

# --- Dataset 定義 ---
class SpatialTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# --- 模型定義 (跟你原本一樣) ---
class BERTSpatialClassifier(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout_rate=0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)

        # 冻结 BERT
        for p in self.bert.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, return_features=False):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS 向量
        dropped = self.dropout(pooled)

        # 取中間倒數第二層 (128-d) 向量，作為可視化的乾淨特徵
        x = self.classifier[0](dropped)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)   # 這裡 shape = (batch, 128)

        if return_features:
            return x  # (B, 128) 特徵
        else:
            return self.classifier[5](x)  # logits


# --- 分析類 ---
class analyze_pca_only:
    def __init__(self, model_path, data_path, text_col='text', label_col='label', batch_size=16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 載入模型
        checkpoint = torch.load(os.path.join(model_path, "model.pth"), map_location=device)
        config = checkpoint["model_config"]
        self.model = BERTSpatialClassifier(model_name=config["model_name"], num_classes=config["num_classes"]).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 載入資料
        df = pd.read_csv(data_path)
        dataset = SpatialTextDataset(df[text_col].values, df[label_col].values, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_feats, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting linear head features"):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].cpu().numpy()

                feats = self.model(ids, mask, return_features=True)  # (B, 128)
                all_feats.append(feats.cpu().numpy())
                all_labels.append(labels)

        X = np.concatenate(all_feats, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # 做 PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_

        # import plotly.express as px

        # df_plot = pd.DataFrame(X_pca, columns=['PC1','PC2','PC3'])
        # df_plot['label'] = y

        # fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3',
        #                     color='label', symbol='label', opacity=0.8)
        # fig.show()

        # 畫 3D 圖
        angles = [
        (20,50),
    ]

        for elev, azim in angles:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap="coolwarm", alpha=0.8)

            ax.set_title(f"3D PCA of Linear Head Features\nExplained Var: PC1={evr[0]:.2f}, PC2={evr[1]:.2f}, PC3={evr[2]:.2f}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            legend1 = ax.legend(*scatter.legend_elements(), title="Labels", loc="best")
            ax.add_artist(legend1)
            ax.view_init(elev=elev, azim=azim)

            plt.tight_layout()
            plt.show()
