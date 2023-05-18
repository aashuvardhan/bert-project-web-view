import torch
from bert_using_pytorch import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd

print('Preparing the Model...')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()
print('BERT Model Created...')

print('Preparing the dataset...')
df = pd.read_csv('C:/Users/aashu/Desktop/sem project/datasets/t_asv.csv')


def text_processing(cont):
    sentence = list(filter(lambda x: x != '', cont.split(' ')))
    total_words = len(sentence)
    from random import randrange
    hidden_ind = randrange(total_words)
    count = hidden_ind

    tgt_word = sentence[count - 1]
    sentence[count - 1] = '---'

    single_text = ' '.join(sentence)
    return (single_text, tgt_word)


def prediction_generator(inputted_text):
    ind = inputted_text.find('---')
    if (ind + 3 < len(inputted_text) and inputted_text[ind + 3] != ' '):
        inputted_text = inputted_text[0:ind + 3] + ' ' + inputted_text[ind + 3:]

    text = '[CLS] ' + inputted_text.replace('---', '[MASK]') + ' [SEP]'

    tokenized_text = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    predictions = model(tokens_tensor, segments_tensors)

    masked_index = tokenized_text.index("[MASK]")

    sorted_order, indices = predictions[0][masked_index].sort(descending=True)

    result = []

    for i in range(5):
        result.append(tokenizer.convert_ids_to_tokens([indices[i].item()])[0])

    return result


clean_txts = []
targets = []
for i in range(len(df)):
    line = df.iloc[i]['t']
    txt, tgt = text_processing(line)
    clean_txts.append(txt)
    targets.append(tgt)

df['Clean_text'] = clean_txts
df['target_wrd'] = targets
print('Datasets Prepared...')


def sentence_parser(sentence, tgt_word):
    predictions = prediction_generator(sentence)
    for ele in predictions:
        if (ele.lower() == tgt_word.lower()):
            return True
    return False


print('Prediction Started...')
counter_true = 0
for i in range(len(df)):
    sent = df.iloc[i]['Clean_text']
    target = df.iloc[i]['target_wrd']
    flagger = sentence_parser(sent, target)
    print(i, flagger)
    if (flagger == True):
        counter_true += 1
print('Predictions ended...')
print(counter_true / len(df))