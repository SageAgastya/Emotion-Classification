import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

XLfile1 = "D:\ChatBotEmotionDetection\MELD.Raw/train_sent_emo.csv"
XLfile2 = "CHATBOT_DATASET.csv"


class dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv("CHATBOT_DATASET.csv")
        emotions = self.df["Emotion"].to_list()
        self.encoder = LabelEncoder()
        self.encoded = self.encoder.fit_transform(emotions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):         #returns (utterance, speaker, emotion)
        tuple_at_index= self.df.iloc[item]
        utterance = tuple_at_index["Utterance"]
        speaker = tuple_at_index["Speaker"]
        encoded_emotion = self.encoded[item]
        return utterance, speaker, encoded_emotion

    def get_class(self):
        return self.encoder.classes_