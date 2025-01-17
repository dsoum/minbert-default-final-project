import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask


TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        #raise NotImplementedError
        
        # sentiment analysis
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)

        # paraphrase detection
        self.para_linear1 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.para_linear2 = nn.Linear(BERT_HIDDEN_SIZE,BERT_HIDDEN_SIZE)
        self.para_dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.para_dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.para_proj_score = nn.Linear(3*BERT_HIDDEN_SIZE, 1)


        # semantic textual similarity
        self.sts_linear1 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.sts_linear2 = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.sts_dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.sts_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        #raise NotImplementedError
        out = self.bert(input_ids, attention_mask)
        return out['pooler_output']


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        #raise NotImplementedError
        out = self.sentiment_dropout(self.forward(input_ids, attention_mask))
        return self.sentiment_classifier(out)


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        #raise NotImplementedError
        out1 = self.para_linear1(self.para_dropout1(self.forward(input_ids_1, attention_mask_1)))
        out2 = self.para_linear2(self.para_dropout2(self.forward(input_ids_2, attention_mask_2)))
        abs = torch.abs(out1-out2)

        return self.para_proj_score(torch.cat((out1, out2, abs), -1))

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        #raise NotImplementedError
        out1 = self.sts_linear1(self.sts_dropout1(self.forward(input_ids_1, attention_mask_1)))
        out2 = self.sts_linear2(self.sts_dropout2(self.forward(input_ids_2, attention_mask_2)))
        return F.cosine_similarity(out1, out2)





def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # sentiment analysis dataset
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # paraphrase dataset
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    # sts dataset
    sts_train_data = SentencePairDataset(para_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)



    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    
    # Run for the specified number of epochs
    # for epoch in range(args.epochs):
    #     model.train()
    #     train_loss = 0
    #     num_batches = 0

    #     # sst training
    #     for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
    #         b_ids, b_mask, b_labels = (batch['token_ids'],
    #                                    batch['attention_mask'], batch['labels'])

    #         b_ids = b_ids.to(device)
    #         b_mask = b_mask.to(device)
    #         b_labels = b_labels.to(device)

    #         optimizer.zero_grad()
    #         logits = model.predict_sentiment(b_ids, b_mask)
    #         loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

    #         loss.backward()
    #         optimizer.step()

    #         train_loss += loss.item()
    #         num_batches += 1

    #     train_loss = train_loss / (num_batches)

    #     train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
    #     dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

    #     if dev_acc > best_dev_acc:
    #         best_dev_acc = dev_acc
    #         save_model(model, optimizer, args, config, args.filepath)

    #     print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

    ######################################################
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        # sentiment classification training (ideal learning rate with 1e-3)
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_sentiment(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        print('Epoch: ', epoch, 'Sentiment Classification Loss: ', train_loss)
        #wandb.log({"Sentiment Classification Epoch": epoch, "train loss": train_loss})

        # paraphrase training -- classification problem
        train_loss = 0
        num_batches = 0

        # Paraphrase not need to train over 10 epochs
        if epoch <= 10:
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_id_1, b_mask_1, b_id_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                           batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])

                b_id_1 = b_id_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_id_2 = b_id_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_id_1, b_mask_1, b_id_2, b_mask_2)
                logits = torch.sigmoid(logits) # normalize

                # L1 loss
                loss = F.l1_loss(logits.view(-1), b_labels) / args.batch_size

                # Cross Entropy Loss Try
                #loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)
            print('Epoch: ', epoch, 'Paraphrase Loss: ', train_loss)
            #wandb.log({"Paraphrase Epoch": epoch, "train loss": train_loss})

        # SemEval training -- regression problem
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_id_1, b_mask_1, b_id_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])

            b_id_1 = b_id_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            b_id_2 = b_id_2.to(device)
            b_mask_2 = b_mask_2.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_id_1, b_mask_1, b_id_2, b_mask_2)
            logits = torch.mul(logits.add(1), 2.5)
            # b_labels_norm = torch.Tensor([1 if x > 2.5 else -1 for x in b_labels])

            # hinge loss
            # loss = F.hinge_embedding_loss(logits, b_labels_norm) / args.batch_size

            # L1 loss
            # loss = F.l1_loss(logits.view(-1), b_labels) / args.batch_size # L1 loss

            # MSE loss
            loss = F.mse_loss(logits, b_labels.to(logits.dtype)) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        print('Epoch: ', epoch, 'SemEval Loss: ', train_loss)
        #wandb.log({"SemEval Epoch": epoch, "train loss": train_loss})

        train_acc_para, _, _, train_acc_sent, _, _, train_acc_sts, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_acc_para, _, _, dev_acc_sent, _, _, dev_acc_sts, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        train_acc = np.average([train_acc_sent, train_acc_para, train_acc_sts])
        dev_acc = np.average([dev_acc_sent, dev_acc_para, dev_acc_sts])

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        # ---------- logging ------------------------ #
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        #wandb.log({"Epoch": epoch, "train loss": train_loss, "train acc": train_acc, "dev acc": dev_acc})


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
