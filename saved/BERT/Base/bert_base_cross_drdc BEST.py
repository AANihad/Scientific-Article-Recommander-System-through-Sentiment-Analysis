num workers = 4
target = torch.tensor([0.4/3, 0.1/3, 2.5/3])
"bert-base-uncased"

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
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output) 
    
	
saving model to 
'/content/drive/MyDrive/Studies/bert_base_cross_dhdrdc/checkpoints/best-checkpoint-drdc.ckpt'
test_prediction = nn.Softmax(dim=1)(test_prediction)
test_prediction = test_prediction.flatten().numpy()
  
  prediction = nn.Softmax(dim=1)(prediction)#
  predictions.append(prediction.flatten())
  
tensor(0.8890)

AUROC per tag
Positive: 0.9323320984840393
Objective: 0.9276322722434998
Negative: 0.9840896725654602

              precision    recall  f1-score   support

    Positive       0.58      0.65      0.62        75
   Objective       0.99      0.83      0.90       778
    Negative       0.19      0.95      0.32        21

   micro avg       0.85      0.81      0.83       874
   macro avg       0.59      0.81      0.61       874
weighted avg       0.93      0.81      0.86       874
 samples avg       0.81      0.81      0.81       874



[[ 59   7   9]
 [ 60 642  76]
 [  0   1  20]]
 
 
[[764  35]
 [ 26  49]]
[[ 88   8]
 [136 642]]
[[768  85]
 [  1  20]]