# Sketchback: Convolutional Sketch Inversion using Keras
Implementation of sketch inversion using deep convolution neural networks (_synthesising photo-realistic images from pencil sketches_) following the work of [Convolutional Sketch Inversion][conv_paper] and [Scribbler][scribbler]. 

We focused on sketches of human faces and architectural drawings of buildings. However according to [Scribbler][scribbler] and our experimentation with their proposed framework, we believe that given a large dataset and ample training time, this network could generalize to other categories as well. 

We trained the model using a dataset generated from a large database of face images, and then fine-tuned the network to fit the purpose of architectural sketches. You can find our reserach poster [here][poster].

## Results
![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)
![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)
![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)
![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)

## Datasets
We used the following datasets to train, validate and test our model:
- [*Large-scale CelebFaces Attributes (CelebA) dataset*][celeba]. CelebA dataset contains 202,599 celebrity face images of 10,177 identities. It covers large pose variation and background clutter. We used this dataset to train the network.
- [*ZuBuD Image Database*][zubud]. The ZuBuD dataset is provided by the computer vision lab of ETH Zurich. It contains 1005 images of 201 buildings in Zurich; 5 images per building from different angles.
- [*CUHK Face Sketch (CUFS) database*][cuhk]. This dataset contains 188 hand-drawn face sketches and their corresponding photographs. We used the CUHK student database for testing our model.
- We finally used various building sketches from *Google Images* for testing

# Sketching
The datasets were simulated, i.e. the sketches were generated, using the following methods (with the exception of the CUHK dataset, which contains sketches and the corresponding images)
- [Pencil Sketchify][pencil]: 
- [XDoG (Extended Difference of Gaussians)][xdog] 
- [Neural Style Transfer][style_transfer]

Furthermore, due to the low number of images of buildings available, we applied various augmentations on the ZuBuD dataset to produce more images using the following [image augmentation tool][tool]

## Network Architecture 

<p align="center">![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)<p>
We used the network architecture used in [Scribbler][scribbler]. The generator follows an encoder-decoder design, with down-sampling steps, followed by residual layers, followed by up-sampling steps.

## Loss Functions 
The Loss function was computed as the weighted average of three loss functions; namely: pixel loss, total variation loss, and feature loss. 

The pixel loss was computed as:
<p align="center">![alt text][pixel_loss]</p>
Where t is the true image, p is the predicted image, and n,m,c are the height, width, and number of color channels respectively.


The feature loss was computed as:
<p align="center">![alt text][feature_loss]</p>

The total variation loss was used to encourage smoothness of the output, and was computed as
<p align="center">![alt text][total_loss]<p>

Where *phi(x)* is the output of the fourth layer in a pre-trained model (VGG16 relu\_2\_2) to feature transform the targets and predictions.

The total loss is then computed as
<p align="center">
*L<sub>t</sub> = w<sub>p</sub>L<sub>p</sub> + w<sub>f</sub>L<sub>f</sub> + w<sub>v</sub>L<sub>v</sub>*
</p>

For the present application, *w<sub>f</sub> = 0.001, w<sub>p</sub> = 1, w<sub>v</sub> = 0.00001*
## Weights
For your convience you can find the following weights here:
- [Weights][w1] after training the network on the CelebA dataset using the Pencil Sketchify method
- [Weights][w2] after fine-tuning the network for the building sketches using the augmented ZuBuD dataset with the Pencil Sketchify method

## Todo
- Training with a **larger building dataset** using a variety of sketch styles to improve the generality of the network
- Adding adversarial loss to the network.
- Using sketch anti-roughing to unify the styles of
the training and input sketches.
- Passing the sketch results to a super-resolution network to improve image clarity.
- Increasing the image size of the training data. 


License
----

MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[celeba]: <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
[zubud]: <http://www.vision.ee.ethz.ch/showroom/zubud/>
[cuhk]: <http://www.ee.cuhk.edu.hk/~jshao/CUHKcrowd_files/cuhk_crowd_dataset.htm>
[poster]: https://github.com/BKHMSI/architecture_sketch_inversion
[xdog]: <>
[pencil]: <>
[style_transfer]: <>
[tool]: <https://codebox.net/pages/image-augmentation-with-python>
[conv_paper]: <https://arxiv.org/abs/1606.03073>
[scribbler]: <https://arxiv.org/abs/1612.00835>
[w1]: <https://drive.google.com/file/d/0B-HY3igdAAMNemRpZHc2SkVIV2s/view?usp=sharing>
[w2]: <https://drive.google.com/file/d/0B-HY3igdAAMNajljMGVObDJwOTA/view?usp=sharing>
[pixel_loss]: <http://www.sciweavers.org/tex2img.php?eq=%09L_%7Bp%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bm%7D%5Csum_%7Bk%3D1%7D%5Ec%28t_%7Bi%2Cj%2Ck%7D%20-%20p_%7Bi%2Cj%2Ck%7D%29%5E%7B2%7D%29%7D%7Bnmc%7D%7D>
[feature_loss]: <>
[total_loss]: <>
