num workers = 4
target = torch.tensor([0.4/3, 0.1/3, 2.5/3])
"bert-large-uncased"

config = {
    'model_name': model_name,
    'n_labels': len(attributes),
    'batch_size': 32,
    'lr': 1.5e-5,
    'warmup': 0.2, 
    'train_size': len(Data_module.train_dataloader()),
    'weight_decay': 0.00,
    'n_epochs': 4,
    'weights' : target
}
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True, num_labels =  self.config['n_labels'])
    self.loss_func = nn.CrossEntropyLoss(weight = config['weights'])
    self.dropout = nn.Dropout()
    

# bert
    output = self.pretrained_model(input_ids=input_ids, attention_mask = attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = self.dropout(pooled_output)
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output) 
    
	
saving model to 
'/kaggle/working/models/bert_Cross_dhdr'

test_prediction = nn.Softmax(dim=1)(test_prediction)
test_prediction = test_prediction.flatten().numpy()
  
  prediction = nn.Softmax(dim=1)(prediction)#
  predictions.append(prediction.flatten())
  
tensor(0.8318)

AUROC per tag
Positive: 0.8358280658721924
Objective: 0.8645967245101929
Negative: 0.9514877200126648

              precision    recall  f1-score   support

    Positive       0.27      0.49      0.35        75
   Objective       0.97      0.72      0.83       778
    Negative       0.20      0.71      0.32        21

   micro avg       0.77      0.70      0.73       874
   macro avg       0.48      0.64      0.50       874
weighted avg       0.89      0.70      0.77       874
 samples avg       0.70      0.70      0.70       874



[[ 52  17   6]
 [166 559  53]
 [  6   0  15]]
 
[[697 102]
 [ 38  37]]
[[ 79  17]
 [219 559]]
[[794  59]
 [  6  15]]