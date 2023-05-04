import pandas as pd
from torch.utils.data import Dataset

class ArticleDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = "生成文章: " + self.data.loc[idx, "title"]
        target_text = self.data.loc[idx, "content"]
        return {"source": source_text, "target": target_text}
