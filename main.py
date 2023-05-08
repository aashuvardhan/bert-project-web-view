# Tokenized input
'''
The dog barks loudly when it spots a wolf trying to breech the security of white house at dusk.
text = "[CLS] Where is the [MASK] that i brought yesterday. [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
'''
#inputted_text=input("Please type '---' wherever you want to mask \n")
# adding space in the text after '---'


def input_predictor(inputted_text):

    try:
        import torch
        from bert_using_pytorch import BertTokenizer, BertModel, BertForMaskedLM
        import numpy

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        ind=inputted_text.find('---')
        if(ind+3<len(inputted_text) and inputted_text[ind+3]!=' '):
            inputted_text=inputted_text[0:ind+3]+' '+inputted_text[ind+3:]


        text='[CLS] '+inputted_text.replace('---','[MASK]')+' [SEP]'

        tokenized_text = tokenizer.tokenize(text)
        #print(tokenized_text)

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        #print(indexed_tokens)
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = [0]*len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        
        

        # Load pre-trained model (weights)
        model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        model.eval()

        # Predict all tokens
        predictions = model(tokens_tensor, segments_tensors)

        masked_index = tokenized_text.index("[MASK]")
        # confirm we were able to predict 'henson'
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        #print(predicted_index)

        # predicting top 5 tokens
        sorted_order,indices=predictions[0][masked_index].sort(descending =True)

        result=[]

        for i in range(5):
            result.append(tokenizer.convert_ids_to_tokens([indices[i].item()])[0])

        # the top most token
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        #print(predicted_token)
        # print(predicted_token == 'henson')
        #print(input_predictor('The dog barks loudly when it spots a wolf trying to breech the security of white house at ---.'))
        return "   ||   ".join(result)
    except Exception as e:
        print(e)
        return "There is some error in your sentence. Please Check and re-enter"
