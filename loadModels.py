from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch, transformerClasses
from transformers import XLNetTokenizer


    
def loadModel(name):
    xlnet = False
    print(name)
    if name == "SciBERT" :
        # model_name = "allenai/scibert_scivocab_uncased"
        model_name = "saved\pretrained\sci"
        tokenizer_path = "saved\\scibert_scivocab\\tokenizer"
        model_path = "saved\\scibert_scivocab\\scibert_state_dict.pth"

    elif name == "SiEBERT":
        # model_name = "siebert/sentiment-roberta-large-english"
        model_name = "saved\pretrained\sie"
        tokenizer_path = "saved\\siebert\\tokenizer"
        model_path = "saved\\siebert\\siebert_dict.pth"
    
    elif name == "BERT Large":
        # model_name = "bert-large-uncased"
        model_name = "saved\pretrained\\bertL"
        tokenizer_path = "saved\\BERT\\Large\\tokenizer"
        model_path = "saved\\BERT\\Large\\bert_large_state_dict.pth"
        
    elif name == "BERT Base":
        # model_name = "bert-base-uncased"
        model_name = "saved\pretrained\\bertB"
        tokenizer_path = "saved\\BERT\\Base\tokenizer"
        model_path = "saved\\BERT\\Base\tokenizer\\bert_base_state_dict.pth"
        
    elif name == "XLNet Large":
        model_name = "xlnet-large-cased"
        xlnet = True
        tokenizer_path = "saved\\XLNet\\Large\\tokenizer"
        model_path = "saved\\XLNet\\Large\\xlnet_large_state_dict.pth"
        
    elif name == "XLNet Base":
        model_name = "xlnet-base-cased"
        xlnet = True
        tokenizer_path = "saved\\XLNet\\Base\\tokenizer"
        model_path = "saved\\XLNet\\Base\\xlnet_base_drdc_state_dict.pth"
    
    if xlnet:
        tokenizer = XLNetTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)

    if not xlnet:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    trained_model = transformerClasses.Citation_Classifier(xlnet, transformerClasses.getConfig(model_name))
    trained_model.load_state_dict(torch.load(model_path, map_location='cpu'))

    return tokenizer, trained_model


def predict(text, model, tokenizer, name):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )
    _, test_prediction = model(encoding["input_ids"], encoding["attention_mask"])
    if not (name == "SiEBERT"):
        test_prediction = torch.nn.Softmax(dim=1)(test_prediction)
    test_prediction = test_prediction.flatten().detach().numpy()
    return test_prediction


def firstTimeLoad():
    print('.')
    model_name = "saved\pretrained\sci"
    model_path = "saved\\scibert_scivocab\\scibert_dict.pth"
    trained_model = transformerClasses.Citation_Classifier(False, transformerClasses.getConfig(model_name))
    print('..')
    model_name = "saved\pretrained\sie"
    model_path = "saved\\siebert\\siebert_state_dict.pth"
    trained_model = transformerClasses.Citation_Classifier(False, transformerClasses.getConfig(model_name))
    print('...')

    model_name = "saved\pretrained\bertL"
    model_path = "saved\\BERT\\Large\\bert_large_state_dict.pth"
    trained_model = transformerClasses.Citation_Classifier(False, transformerClasses.getConfig(model_name))
    print('....')

    model_name = "saved\pretrained\bertB"
    model_path = "saved\\BERT\\Base\tokenizer\\bert_base_state_dict.pth"
    trained_model = transformerClasses.Citation_Classifier(False, transformerClasses.getConfig(model_name))
    print('.....')

    model_name = "xlnet-large-cased"
    model_path = "saved\\XLNet\\Large\\xlnet_large_dict.pth"
    trained_model = transformerClasses.Citation_Classifier(True, transformerClasses.getConfig(model_name))
    print('......')

    model_name = "xlnet-base-cased"
    model_path = "saved\\XLNet\\Base\\xlnet_base_state_dict.pth"
    trained_model = transformerClasses.Citation_Classifier(True, transformerClasses.getConfig(model_name))
    print('.......')


if __name__ == '__main__':
    available_models = ["SciBERT", "SiEBERT", "BERT Large", "BERT Base", "XLNet Large", "XLNet Base"]
    name = available_models[0]
    tokenizer, model = loadModel(name)

    textP = "4.1 Complete ambiguity classes Ambiguity classes capture the relevant property we are interested in: words with the same category possibilities are grouped together.4 And ambiguity classes have been shown to be successfully employed, in a variety of ways, to improve POS tagging (e.g., Cutting et al., 1992; Daelemans et al., 1996; Dickinson, 2007; Goldberg et al., 2008; Tseng et al., 2005)."
    textN ="Many approaches for POS tagging have been developed in the past, including rule-based tagging (Brill, 1995), HMM taggers (Brants, 2000; Cutting and others, 1992), maximum-entropy models (Rathnaparki, 1996), cyclic dependency networks (Toutanova et al. , 2003), memory-based learning (Daelemans et al. , 1996), etc. All of these approaches require either a large amount of annotated training data (for supervised tagging) or a lexicon listing all possible tags for each word (for unsupervised tagging)."

    # firstTimeLoad()

    predictions = predict(textP, model, tokenizer, name)
    print(predictions)
    print(type(predictions))
