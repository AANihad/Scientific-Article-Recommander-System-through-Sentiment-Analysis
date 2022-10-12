num workers = 4
target = torch.tensor([0.4/3, 0.1/3, 2.5/3])
'siebert/sentiment-roberta-large-english'
config = {
    'model_name': model_name,
    'n_labels': len(attributes),
    'batch_size': 32,
    'lr': 1.5e-5,
    'warmup': 0.2, 
    'train_size': len(Data_module.train_dataloader()),
    'weight_decay': 0.00,
    'n_epochs': 3,
    'weights' : target
}
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True, num_labels =  self.config['n_labels'])
    self.loss_func = nn.CrossEntropyLoss(weight = config['weights'])
    self.dropout = nn.Dropout(p=0.2)
    

 # sciebert
    output = self.pretrained_model(input_ids=input_ids, attention_mask = attention_mask)

    pooled_output = torch.mean(output.last_hidden_state, 1)
    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.classifier(pooled_output) 
    logits = self.soft(pooled_output) 
    
	
saving model to '/content/drive/MyDrive/Studies/siebert_cross_d_h_d_c/checkpoints/best-checkpoint-v1.ckpt' 


test_prediction = test_prediction.flatten().numpy()

  predictions.append(prediction.flatten())
  
tensor(0.8261)

AUROC per tag
Positive: 0.8674342632293701
Objective: 0.8508194088935852
Negative: 0.9268129467964172

              precision    recall  f1-score   support

    Positive       0.29      0.73      0.42        75
   Objective       0.98      0.73      0.84       778
    Negative       0.19      0.90      0.32        21

   micro avg       0.74      0.74      0.74       874
   macro avg       0.49      0.79      0.52       874
weighted avg       0.90      0.74      0.79       874
 samples avg       0.74      0.74      0.74       874



[[ 55  12   8]
 [135 571  72]
 [  1   1  19]]
 
 
 [[665  134]
 [ 20  55]]
[[ 83   13]
 [207 571]]
[[773  80]
 [  2  19]]