# The first place solution of [Zero to GANs - Human Protein Classification](https://www.kaggle.com/c/jovian-pytorch-z2g)


### Model:
    - EfficientNet-B1 vs EfficientNet-B2: I used both of them for ensemble.
    - I didn't use EfficientNet-B4 because the score drops when I resized the image.
    
### Optimizer:
    AdamW vs Adam:         
        AdamW optimizer converges faster than Adam. See more details in [Why AdamW matters](https://towardsdatascience.com/why-adamw-matters-736223f31b5d)

### Learning rate scheduler: 
    OneCyclic vs CosineAnnealingWarmRestarts:      
        CosineAnnealingWarmRestarts converges faster but the better f1 scores (>0.82) are from using Onecyclic with initial learning rate 0.0001 and 26 #epochs.
        This [article](https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee) explains why OneCyclic performs better with the right number of epochs and learning rate.

### Image augmentation:
    - I use below augmentation during training time and turn it off during validation and test time.
    - RandomHorizontalFlip
    - RandomVerticalFlip
    - RandomRotation
    - ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2)
    - Resize() (used only when searching for hyperparameters, I get higher scores without using Resize())
    
### Loss:
    - binary crossentropy vs focal loss: when focal loss decreases, f1 score doesn't increase much and I get better f1-score from using binary cross-entropy.
    
### Ensemble: 
    - I ensemble the 6 best checkpoints of my model.
 
### Convert probability that exceeds a certain threshold to labels
    - I set 0.5 as the threshold value as the predicted classes and then fill missing classes with threshold over 0.46 and 0.445 respectively. 
    - I Use argmax() for predicting the rest of the missing classes.
