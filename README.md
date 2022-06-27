# VISIOPE_GANime_corvitto_faiella
Repository for the project of Vision and Perception about a GAN that generates anime faces.

## Dataset
Since our goal is to generate drawn and colored faces in an updated anime style, we searched and found a dataset that met our exception on Kaggle.com (https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset). This dataset contains 92,219 images of size 256x256 and take examples from different works and different authors. This examples are mainly composed by female anime faces, so our architecture is biased to generate female anime faces.

## Baseline: DCGAN
We started our architecture from a simple **D**eep **C**onvlutional **G**enerative **A**dversarial **N**etwork.
Our goal was to explore different variation of this architecture, trying to enhance performances reaching good-looking results.

## RESdcgan
The first modification was on the models. We changed the generator (mainly) and the discriminator adding Residual blocks.
We built different version of them and tried different combination to find the best one.
Some combination did not give good results, on the contrary they diminished the performance leading to divergence (noisy results).
Others, instead, enhanced greatly the performance of the network, at the cost of some computational time.

## Checkpoints
At some point of our work we thought that if we want to generate some really good images, we needed to train our architecture for long time, because we had limits on our computational power, in fact all the architectures were trained on a single GPU (NVIDIA RTX 3060 laptop).
So we added checkpoints to store our progresses and keep the training from the point we stopped the last time.
Thanks to this we were able to train the architecture for 200 epochs.

## FID metric for Anime Images
We also wanted an objective metric to evaluate our progresses, so we implemented FID.
However, the original FID was based on the Inception V3 model, that does not work very well on illustration, paintings et similia.
So we used the Illustration2Vec pre-trained model (loaded on the releases, also available at https://github.com/rezoo/illustration2vec) as feature extractor and then computed the FID.
