# [Zero to GANs - Human Protein Classification](https://www.kaggle.com/c/jovian-pytorch-z2g)

## 1<sup>st</sup> place solution

<img src="protein_image.png" width="400" height="400">


## Note

During training I used many manual steps. 

- I stopped the training when the model started to overfit and reduce the learning rate to train more.

- I used checkpoint the save the best weight and use it as an initialize weight for the next training.

- I hand-picked different training and validation set without using cross-validation for training the model.


## Files

`image_classification.ipynb` is for training the classification.

`ensemble.ipynb` is for merging multiple predicted probabilities as the final submission. 

## Solution

Below are the solution that I used for the final submission.

### Model: EfficientNet-B1 and EfficientNet-B2
    - EfficientNet-b0 vs EfficientNet-b1 vs EfficientNet-b2 vs EfficientNet-b4 vs ResetNet101 vs DenseNet121
        - EfficientNets perform better than resnets and densenet.
    - EfficientNet-B1 vs EfficientNet-B2: I used both of them for ensemble.
    - I didn't use EfficientNet-B4 because the score drops when I resized the image.
    
### Optimizer: AdamW, amsgrad=True, weight_decay=0.01
    AdamW vs Adam:         
        AdamW optimizer converges faster than Adam. 
   See more details in [Why AdamW matters](https://towardsdatascience.com/why-adamw-matters-736223f31b5d) and [AdamW and Super-convergence is now the fastest way to train neural nets](https://www.fast.ai/2018/07/02/adam-weight-decay/)
        

### Learning rate scheduler: OneCyclic Learning Rate Scheduler
    OneCyclic vs CosineAnnealingWarmRestarts:  
        - CosineAnnealingWarmRestarts converges faster but the better f1 scores (>0.82) are from using Onecyclic with initial learning rate 0.0001 and 26 #epochs.
        - I used CosineAnnealingWarmRestarts for finding the best model, then I used OneCyclic for generating a single submission result before ensemble.
   This [article](https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee) explains why choosing the right number of epochs and learning rate matters for OneCyclic.

### Image augmentation:
    - I use below augmentation during training time and turn it off during validation and test time.
    - RandomHorizontalFlip
    - RandomVerticalFlip
    - RandomRotation
    - ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2)
    - Resize() (used only when searching for hyperparameters, I get higher scores without using Resize())
    
### Loss: Binary cross-entropy
    - binary cross-entropy vs focal loss: 
        when focal loss decreases, f1 score doesn't increase much. I get better f1-scores from using binary cross-entropy.
    
### Ensemble
    - I ensemble the 6 best checkpoints of my model.
 
### Convert probability that exceeds a certain threshold to labels: 0.5 > 0.46 > 0.445 > argmax
    - I set 0.5 as the threshold value as the predicted classes and then fill missing classes with threshold over 0.46 and 0.445 respectively. 
    - I Use argmax() for predicting the rest of the missing classes.
    - At first I filled the missing classes with mode class (class 4) and then I changed to fill it with the argmax probabilities that the model generated.

### Find my best single checkpoint here:
`best_checkpoint/weight.pth`
