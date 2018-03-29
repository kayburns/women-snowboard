# Women also Snowboard: Overcoming Bias in Captioning Models 

This is meant to be internal documentation so that we can easily remember how things work in May (rebuttal time!).  
When we release the code, we can think about how to succinctly present this for others.


## Constructing tfrecords with blocked images:

TODO: Kaylee

## Flags added to training code:

input_file_pattern2: Path for an additional path to tfrecord files.  Use to train balanced model.

blocked_input_file_pattern: Path to indicate path to blocked tfrecord files. 

loss_weight_value:  Indicates the loss weight to place on man/woman words.  This is used for all UpWeight models in the paper.  Can only indicate one weight for man words and for woman words.

blocked_image:  Whether or not to train the appearance confusion loss.

two_input_queues:  Whether or not to trian data from two input queues.  This was used to train the balanced data in the paper.

blocked_weight_selective:  What a confusing name!  This is the flag for sum/no-sum for the appearance confusion loss.  By default this is True.  This means that by default models will be trained as indicated in equations from the paper; mainly the difference in probabilities is only considered for gendered words.

blocked_loss_weight:  How much to weight the appearance confusion loss.

blocked_image_ce: Flag to include cross entropy loss on blocked images

blocked_image_ce_weight:  Weight for blocked image ce loss

confusion_word_non_blocked_weight:  Weight for confident loss.

confusion_word_non_blocked_type: Type for confident loss (we use type quotient for our ECCV experiments). 


## Running GradCam

TODO for Anja

## Running Saliency Code

TODO for Lisa/Kaylee

## Running Data Analysis

TODO for Lisa/Kaylee

## Auxiliary Scripts

Creating blocked images: TODO for Lisa

Creating gt segmentation masks: TODO for Anja

Creation of different subsets: TODO for Lisa/Anja
