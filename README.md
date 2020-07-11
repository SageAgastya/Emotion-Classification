# Emotion-Classification
The repo uses self-attention and cross-attention over MELD dataset for classifying emotions in a conversation

Cleaning_Meld.py = the script will clean the MELD dataset and will generate a new CSV file that will be used for training.
ConversationalAI_Dataset.py = the script will use Dataset-class to convert the MELD dataset into torch's Dataset class format.
ConversationalAI.py = the script will do the training.

After training, a file with ".pt" extension will be generated, that will save all the trained weights after training, use it for inferencing.

# APPROACH:
Using the pretrained-Bert representation the input is encoded first. Then, 
I am using self-attention for encoding the current utterance and cross-attention for attending the important concepts from previous context as the emotion of current speaker will also depend upon the context of utterance by previous speakers. So, I have tried to preserve the knowledge from previous 3-utterances to get the context. Hence, in the context, I am preserving the self-attention representations from previous 3- attentions. At last, the most discriminative feature is pooled out using Average(for self-attention) and Max pooling(for cross-attention) strategies.

# Future scope:
We can add concept net embeddings to enrich the concepts in our BERT embeddings.
NRC-VAD scores can be used to generate scores of the concepts based on the category to which it belongs to : VALENCE, AROUSAL, DOMINANT


# PAPER I referred:
1. DialogueRNN
2. Knowledge Enriched Transformers (I adopted this one and innovated it by few amount)

# Note:
I am not uploading MELD data, expecting that it is already available with you.
