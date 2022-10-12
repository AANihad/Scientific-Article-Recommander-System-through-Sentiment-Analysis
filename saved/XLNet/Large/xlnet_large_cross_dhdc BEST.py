num workers = 4
target = torch.tensor([0.4/3, 0.1/3, 2.5/3])
model_name = "xlnet-large-cased"

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
    self.pretrained_model = XLNetModel.from_pretrained(config['model_name'], return_dict = True, num_labels =  self.config['n_labels'])
    self.loss_func = nn.CrossEntropyLoss(weight = config['weights'])
    self.dropout = nn.Dropout()
    

 # xlnet
    # xlnet
    output = self.pretrained_model(input_ids=input_ids, attention_mask = attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output) 
    #logits =  self.soft(pooled_output)

	
saving model to '/content/drive/MyDrive/Studies/PFE Master/XLNet_cross_soft/checkpoints/best-checkpoint.ckpt'Epoch 0, global step 874: 'val_loss' reached 0.09610 
Epoch 0, global step 874: 'val_loss' reached 0.07367 
Epoch 1, global step 1748: 'val_loss' reached 0.07099
Epoch 2, global step 2622: 'val_loss' reached 0.06921
Epoch 3, global step 3496: 'val_loss' reached 0.05543
Epoch 4, global step 4370: 'val_loss' reached 0.05298

test_prediction = nn.Softmax(dim=1)(test_prediction)
test_prediction = test_prediction.flatten().numpy()

  prediction = nn.Softmax(dim=1)(prediction)#
  predictions.append(prediction.flatten
  
tensor(0.7201)

AUROC per tag
Positive: 0.7433125376701355
Objective: 0.8268663883209229
Negative: 0.964941680431366

              precision    recall  f1-score   support

    Positive       0.20      0.57      0.30        75
   Objective       0.99      0.48      0.65       778
    Negative       0.15      1.00      0.25        21

   micro avg       0.59      0.50      0.55       874
   macro avg       0.44      0.69      0.40       874
weighted avg       0.90      0.50      0.61       874
 samples avg       0.50      0.50      0.50       874




[[ 52   5  18]
 [296 377 105]
 [  0   0  21]]
 
 [[626 173]
 [ 32  43]]
[[ 91   5]
 [401 377]]
[[730 123]
 [  0  21]]