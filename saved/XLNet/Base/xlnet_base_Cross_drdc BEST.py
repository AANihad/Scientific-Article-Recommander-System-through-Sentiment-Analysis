num workers = 4
target = torch.tensor([0.4/3, 0.1/3, 2.5/3])
"xlnet-base-cased"

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
'/kaggle/working/xlnet_base_Cross_drdc_state_dict.pth'
test_prediction = test_prediction.flatten().numpy()
  
  prediction = nn.Softmax(dim=1)(prediction)#
  predictions.append(prediction.flatten())
  
tensor(0.8177)

AUROC per tag
Positive: 0.8531164526939392
Objective: 0.8631641268730164
Negative: 0.9623737335205078

              precision    recall  f1-score   support

    Positive       0.31      0.67      0.43        75
   Objective       0.97      0.68      0.80       778
    Negative       0.19      0.90      0.31        21

   micro avg       0.75      0.69      0.72       874
   macro avg       0.49      0.75      0.51       874
weighted avg       0.90      0.69      0.76       874
 samples avg       0.69      0.69      0.69       874



[[ 54  14   7]
 [172 532  74]
 [  2   0  19]]
 
 
[[689 110]
 [ 25  50]]
[[ 82  14]
 [246 532]]
[[772  81]
 [  2  19]]