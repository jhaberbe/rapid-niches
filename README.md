# Quickly perform niche analysis on your spatial dataset

Tutorial in the notebook folder.

#### Additions to be done
[] Update loss to see if KL div and adversarial batch correction would be helpful.


# As it turns out...
[Novae](https://github.com/MICS-Lab/novae) was released before my package as a pre-print and is now public. It does a lot of the things that this package does. Basically, it does the same type of edge-attribute usage on spatial graphs, but the objective function is different between the two packages. Novae uses a SwAV objective, which is very [cool](https://github.com/facebookresearch/swav?tab=readme-ov-file), I used likelihood-based objective from a normalizing flow. I haven't tested to see which model does better, but I suspect that Novae would do better in terms of clustering performance. I'm probably going to toy around with SwAV objective in the future, but I'm going to keep this model how it is. 

This blurb is mostly just to say I didn't see this package prior to 12DEC2025.