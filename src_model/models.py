import torch
import torch.nn as nn

# from functions import ReverseLayerF
from transformers import BertModel
# from transformers import BertPreTrainedModel
# from transformers import RobertaModel

# from main import args


def max_pool(x):
    return x.max(2)[0]


def mean_pool(x, sl):
    return torch.sum(x, 1) / sl.unsqueeze(1).float()


class LR(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding = nn.Embedding(args.n_emb, args.emb_dim)
        self.fc = nn.Linear(args.emb_dim, args.n_cls)

    def forward(self, x, sl):
        emb = self.embedding(x)  # [B, L, H_e]
        rep = mean_pool(emb, sl)  # [B, H_e]
        logits = self.fc(rep)
        return logits


class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = args.device

        self.embedding = nn.Embedding(args.n_emb, args.emb_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(args.emb_dim, args.n_chan, k, padding=k)
                for k in args.ksizes
            ]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(
            args.n_chan * len(args.ksizes), args.n_cls
        )

    def conv_and_pool(self, x, conv):
        return max_pool(self.relu(conv(x)))

    def forward(self, x, sl=None):
        emb = self.embedding(x).transpose(1, 2)
        rep = [self.conv_and_pool(emb, conv) for conv in self.convs]
        rep = torch.cat(rep, dim=1).to(self.device)
        logits = self.fc(self.dropout(rep))
        return logits


class GRU(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding = nn.Embedding(args.n_emb, args.emb_dim)
        self.gru = nn.GRU(args.emb_dim, args.n_gru_hid)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.n_gru_hid, args.n_cls)

    def forward(self, x, sl):
        emb = self.embedding(x)  # [B, L, H_e]
        out, _ = self.gru(emb)  # [B, L, H_l]
        rep = mean_pool(out, sl)  # [B, H_l]
        logits = self.fc(self.dropout(rep))  # [B, C]
        return logits


class BertMLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bert = BertModel.from_pretrained(args.bert_model)
        self.fc = nn.Linear(args.n_bert_hid, args.n_cls)

    def forward(self, x, sl=None):
        mask = (x != 0).float()
        emb, _ = self.bert(x, attention_mask=mask)  # [B, L, H_b]
        rep = emb[:, 0, :]  # [B, H_b]
        logits = self.fc(rep)  # [B, C]
        return logits


class RobertaMLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(args.bert_model)
        self.fc = nn.Linear(args.n_bert_hid, args.n_cls)

    def forward(self, x, sl=None):
        mask = (x != 1).float()
        emb, _ = self.roberta(x, attention_mask=mask)  # [B, L, H_b]
        rep = emb[:, 0, :]  # [B, H_b]
        logits = self.fc(rep)  # [B, C]
        return logits


# class BertDANN(BertPreTrainedModel):
class BertDANN(nn.Module):
    def __init__(self, args, cache_dir=None):
        super().__init__()
        self.args = args
        self.num_labels = args.n_cls
        self.bert_path = args.bert_path
        self.bert_model = args.bert_model
        self.cache_dir = cache_dir
        self.initialize_bert()
        self.dropout = nn.Dropout(args.dp)
        # self.classifier_cls = nn.Linear(args.hidden_size, args.n_cls)
        self.classifier_cls = nn.Sequential(
            nn.Linear(args.hidden_size, 256), nn.Dropout(args.ddp), nn.ReLU(), nn.Linear(256, args.n_cls))

        # self.init_weights()
        # self.fc = nn.Linear(args.n_bert_hid, args.n_cls)
        # self.bert = BertModel(config)

        # self.classifier_domain = nn.Linear(args.hidden_size, 2)
        self.classifier_domain = nn.Sequential(
            nn.Linear(args.hidden_size, 256), nn.Dropout(args.ddp), nn.ReLU(), nn.Linear(256, 2))
        # nn.Linear(args.hidden_size, 256), nn.Softmax(dim=-1))

    def initialize_bert(self):
        if self.bert_path != None:
            self.bert = BertModel.from_pretrained(self.bert_path)
        else:
            self.bert = BertModel.from_pretrained(
                self.bert_model, cache_dir=self.cache_dir)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.reset_parameters()

    def initialize_domain(self):
        self.classifier_domain.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        bool_domain=False,
        bool_class=True,
    ):
        return_dict = return_dict if return_dict is not None else False
        # alpha = alpha if alpha is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        # if alpha is not None:
        #     reverse_pooled_output = ReverseLayerF.apply(pooled_output, alpha)

        logits_cls = self.classifier_cls(pooled_output) if bool_class else None
        logits_domain = self.classifier_domain(
            pooled_output) if bool_domain else None

        return logits_cls, logits_domain
