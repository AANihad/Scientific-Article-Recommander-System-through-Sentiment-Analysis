num workers = 4
target = torch.tensor([0.4/3, 0.1/3, 2.5/3])

config = {
    'model_name': 'allenai/scibert_scivocab_uncased',
    'n_labels': len(attributes),
    'batch_size': 32,
    'lr': 1.5e-5,
    'warmup': 0.2, 
    'train_size': len(Data_module.train_dataloader()),
    'weight_decay': 0.00,
    'n_epochs': 10,
    'weights' : target
}

    self.loss_func = nn.CrossEntropyLoss(weight = config['weights'])
    self.dropout = nn.Dropout
    
    
    # scibert
    output = self.pretrained_model(input_ids=input_ids, attention_mask = attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = self.dropout(pooled_output)
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output) 
    
    
	saving model to '/content/drive/MyDrive/Studies/scibert_CrossRelu/checkpoints/best-checkpoint.ckpt' as top 1
Epoch 0   global step 874: 'val_loss' reached 0.06882 
Epoch 1, global step 1748: 'val_loss' reached 0.05579 
Epoch 2, global step 2622: 'val_loss' reached 0.04283 
Epoch 3, global step 3496: 'val_loss' was not in top 1
Epoch 4, global step 4370: 'val_loss' reached 0.02564 
Epoch 5, global step 5244: 'val_loss' reached 0.02160 
Epoch 6, global step 6118: 'val_loss' reached 0.02016 
Epoch 7, global step 6992: 'val_loss' reached 0.01788 
Epoch 8, global step 7866: 'val_loss' was not in top 1
Epoch 9, global step 8740: 'val_loss' reached 0.01624 

test_prediction = nn.Softmax(dim=1)(test_prediction)
test_prediction = test_prediction.flatten().numpy()

  prediction = nn.Softmax(dim=1)(prediction)#
  predictions.append(prediction.flatten())

tensor(0.9325)

AUROC per tag
Positive: 0.979607880115509
Objective: 0.9589358568191528
Negative: 0.997264564037323

              precision    recall  f1-score   support

    Positive       0.59      0.88      0.71        75
   Objective       0.94      0.96      0.95       778
    Negative       0.30      1.00      0.47        21

   micro avg       0.86      0.96      0.90       874
   macro avg       0.61      0.95      0.71       874
weighted avg       0.90      0.96      0.92       874
 samples avg       0.90      0.96      0.92       874

[[ 67   7   1]
 [ 48 704  26]
 [  0   0  21]]
 
 [[754  45]
 [ 9  66]]
[[ 51   8]
 [30 748]]
[[805  48]
 [  0  21]]