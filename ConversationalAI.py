import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .ConversationalAI_Dataset import dataset
from transformers import BertTokenizer, BertModel
from torch.autograd.variable import Variable
import math
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------------------------------------------
class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)

    def forward(self, sent):
        tokens = self.tokenizer.tokenize(sent)
        input_ids = torch.tensor(self.tokenizer.encode(tokens, add_special_tokens=False, add_space_before_punct_symbol=True)).unsqueeze(0).to(device)  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_states, mems = outputs[:2]  # The last hidden-state is the first element of the output tuple
        return last_hidden_states

class PositionalEncoder(nn.Module):  # passed successfully
    def __init__(self, dim=768, max_seq_len=95):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_seq_len, dim)
        for pos in range(max_seq_len):
            for i in range(0, dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / dim)))


        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head, seqlen):
        super().__init__()
        self.Q_lin = nn.Linear(dim, dim)
        self.K_lin = nn.Linear(dim, dim)
        self.V_lin = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)
        self.head = head
        self.dim = dim
        self.seqlen = seqlen


    def forward(self, Q,K,V):
        Q = self.Q_lin(Q).view(1, self.seqlen, self.head, -1)
        K = self.K_lin(K).view(1, self.seqlen, self.head, -1)
        V = self.V_lin(V).view(1, self.seqlen, self.head, -1)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        d_k = Q.shape[-1]

        MH_score = Attention(Q, K, V, d_k)
        MH = MH_score.transpose(1, 2).contiguous().view(1, -1, self.dim)
        return MH, self.linear(MH)


def Attention(Q,K,V, d_k):
    score = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k), dim=-1)
    MH_score = torch.matmul(score, V)
    return MH_score

class PointWiseFeedForward(nn.Module):
    def __init__(self, dim, d_ff, keep_prob = 0.1):
        super().__init__()
        self.l1 = nn.Linear(dim,d_ff)
        self.l2 = nn.Linear(d_ff,dim)
        self.dropout = nn.Dropout(keep_prob)

    def forward(self, X):
        res = self.dropout(F.relu(self.l1(X)))
        res = self.l2(res)
        return res

class Norm(nn.Module):  # passed succesfully
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.size = dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class TransEncoderLayer(nn.Module):
    def __init__(self, dim, head, seqlen, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(dim)
        self.norm_2 = Norm(dim)
        self.attn = MultiHeadAttention(dim,head, seqlen)
        self.ff = PointWiseFeedForward(dim, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, utter):
        x2 = self.norm_1(utter)
        x = utter + self.dropout_1(self.attn(x2, x2, x2)[1])
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


def cloning(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransEncoder(nn.Module):

    def __init__(self, N, head, dim, seqlen, d_ff):
        super().__init__()
        self.N = N
        self.layers = cloning(TransEncoderLayer(dim,head,seqlen, d_ff),N)
        self.norm = Norm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.linear(self.norm(x))


class Process(nn.Module):
    def __init__(self, head=8, N=5, dim=768, d_ff=1536, M=3, pad=290, seqlen=95, classes=7):
        super().__init__()
        self.dim = dim
        self.positional_encoder = PositionalEncoder().to(device)
        self.Bert = BERT().to(device)
        self.MH = TransEncoder(N, head, dim, seqlen, d_ff)
        self.crossAttn = MultiHeadAttention(dim, head, pad)
        self.maxpool = nn.MaxPool2d(kernel_size=(pad,1))
        self.avgpool = nn.AvgPool2d(kernel_size=(pad,1))
        self.stack = []
        self.M = M
        self.pad = pad
        self.seqlen = seqlen
        self.classes = classes
        self.dim = dim
        self.FF = nn.Linear(dim,classes)

    def forward(self, X):
        X = self.Bert(X)
        # print(X.shape)
        padded_utter = torch.zeros(1,self.pad,self.dim).cuda()
        padded_positioning = torch.zeros(1,self.seqlen,self.dim).cuda()
        padded_context = torch.zeros(1,self.pad,self.dim).cuda()

        retained_seqlen = X.shape[1]
        padded_positioning[0,:retained_seqlen,:] = X[0,:,:]

        X = self.positional_encoder(padded_positioning)   # 1x95x768
        pad_encodings = torch.zeros(1,self.seqlen,self.dim).cuda()
        pad_encodings[0,:retained_seqlen,:] = X[0,:retained_seqlen,:]  # 1x95x768

        X = pad_encodings
        self_attn_utter = self.MH(X)    # 1x95x768

        padded_utter[0,:retained_seqlen,:] = self_attn_utter[0,:retained_seqlen,:]     # 1x290x768
        size = len(self.stack)

        if(size>0):
            concat = torch.cat(self.stack,dim=-2)
            context_info = self.MH(concat)
            # print(context_info.shape)
            padded_context[0, :context_info.shape[1], :] = context_info[0, :, :]
            # print(padded_context.shape)
            res = self.crossAttn(padded_utter,padded_context,padded_context)[0]
            mpool = self.maxpool(res).squeeze(0)
            out = self.FF(mpool)                     # Have removed softmax
            # print("paddedutter:"+str(index_of_example+1),padded_utter.shape)


        else:
            apool = self.avgpool(padded_utter).squeeze(0)
            out = self.FF(apool)
            # print("paddedutter:"+str(index_of_example+1), padded_utter.shape)

        if (len(self.stack) >= self.M):
            self.stack.pop(0)
            self.stack.append(self_attn_utter)
        else:
            self.stack.append(self_attn_utter)

        return out

def train():
    net = Process().to(device)
    optimizer = optim.Adam(params=net.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=0)
    criterion = nn.CrossEntropyLoss().to(device)

    epochs = 2
    data = dataset()
    length = len(data)
    loss_curve = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, x in enumerate(data):
            input_utterance, speaker, target = x             # encoded_emotion = target
            target = torch.tensor([target]).cuda()
            optimizer.zero_grad()
            output = net(input_utterance)
            loss = criterion(output, target)
            print("i: ",i, "epoch: ",epoch,"loss: ",loss)
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()

        loss_curve.append(running_loss / length)
    print("training fininshed")

    PATH = "trained_chatbot_weights2.pt"
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_curve,
    }, PATH)

    plt.plot(loss_curve, "b")
    plt.show()

def test():
    PATH = "trained_chatbot_weights2.pt"
    model = Process().to(device)
    checkpnt = torch.load(PATH)
    model.load_state_dict(checkpnt["model_state_dict"])
    model = model.eval()

    test_dataset = ["surprised surprised surprised !!", "this is going to be awesome", "My duties?  All right."]

    train_dataset = dataset()
    classes = train_dataset.get_class()

    for i in range(len(test_dataset)):
        output = model(test_dataset[i])
        max_index = torch.argmax(output, dim=-1)
        print(test_dataset[i]+":", classes[max_index])

# train()
# test()