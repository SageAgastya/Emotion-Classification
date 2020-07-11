import pandas as pd
from sklearn.preprocessing import LabelEncoder

def convert(XLfile1, XLfile2):
    dataframe = pd.read_csv(XLfile1)
    utterances = dataframe["Utterance"]
    speakers = dataframe["Speaker"].to_list()
    emotions = dataframe["Emotion"].to_list()
    dataset = []
    cleaned_utterances = []
    labels = LabelEncoder()
    encoded_emotions = labels.fit_transform(emotions)

    for utter in utterances:  # cleaning the utterances
        utter = utter.encode("utf8").decode("ascii", "ignore")
        cleaned_utterances.append(utter)

    dataset.append(speakers)
    dataset.append(cleaned_utterances)
    dataset.append(emotions)
    dataset.append(encoded_emotions)
    format = pd.DataFrame(dataset).transpose()
    cols = ["Speaker", "Utterance", "Emotion", "Encoded_Emotions"]
    format.columns = cols
    format.to_csv(XLfile2)
    return