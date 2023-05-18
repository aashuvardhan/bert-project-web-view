import torch
from bert_using_pytorch import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd

print('Preparing the BERT Model...')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()
print('BERT Model Created...')

print('Preparing the dataset...')
df=pd.read_csv('C:/Users/aashu/Desktop/sem project/datasets/articles1.csv')

def text_processing(cont):
    sentences = cont.split('.')
    first_five = sentences[:5]
    n=len(first_five)
    total_words = 0
    container_sent=[]
    for j in range(n):
        newsent = list(filter(lambda x: x != '', first_five[j].split(' ')))
        # print(newsent)
        if(len(newsent)>0):
            container_sent.append((newsent, len(newsent)))
            total_words += len(newsent)
    # print(first_five)
    if(total_words==0 or total_words==1 or total_words>500):
        return ('---','.')

    else:
        from random import randrange
        hidden_ind = randrange(total_words)
        # print(hidden_ind)
        count = hidden_ind
        j = 0
        n=len(container_sent)
        while (j < n):
            if (container_sent[j][1] < count):
                count -= container_sent[j][1]
                j += 1
            else:
                break
        tgt_word = container_sent[j][0][count - 1]
        # print(tgt_word)
        container_sent[j][0][count - 1] = '---'

        for j in range(n):
            container_sent[j] = ' '.join(container_sent[j][0])
        # print(first_five)
        single_sent = '. '.join(container_sent)
        # print(single_sent)
        return (single_sent,tgt_word)



clean_txts=[]
targets=[]
for i in range(len(df)):
    cont = df.iloc[i]['content']
    txt,tgt=text_processing(cont)
    clean_txts.append(txt)
    targets.append(tgt)
df['Clean_text']=clean_txts
df['target_wrd']=targets
print('Dataset Prepared...')


def prediction_generator(text):
    ind=text.find('---')
    if(ind+3<len(text) and text[ind+3]!=' '):
        text=text[0:ind+3]+' '+text[ind+3:]


    text='[CLS] '+text.replace('---','[MASK]')+' [SEP]'

    tokenized_text = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0]*len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    predictions = model(tokens_tensor, segments_tensors)

    masked_index = tokenized_text.index("[MASK]")

    #predicted_index = torch.argmax(predictions[0, masked_index]).item()

    sorted_order,indices=predictions[0][masked_index].sort(descending =True)

    result=[]

    for i in range(5):
        result.append(tokenizer.convert_ids_to_tokens([indices[i].item()])[0])

    return result

def sentence_parser(sentence,tgt_word):
    predictions = prediction_generator(sentence)
    for ele in predictions:
        if(ele.lower()==tgt_word.lower()):
            return True
    return False


print('Predictions Started...')
counter_true=0
for i in range(len(df)):
    sent=df.iloc[i]['Clean_text']
    target=df.iloc[i]['target_wrd']
    flagger=sentence_parser(sent,target)
    print(i,flagger)
    if(flagger==True):
        counter_true+=1
print('Predictions ended')
print(counter_true/len(df))





