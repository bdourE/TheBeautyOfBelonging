


# WGAN-Saudi Heritage Archicture Art



## Project Overview

 This project inspired from Heritage architecture from 13 regioen of Saudi arabie .it's part of [Artathon Competion](https://www.theglobalaisummit.com). the deep learning model  we used is  modified version of [Erik Linder-Norén PyTorch implementation](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp) of  Wasserstein GAN [WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf) . our model can generate rectangle images with size ( 480x270 )    .Our dataset is collection of digital illustrations created by [Nourah Alashwali](https://www.instagram.com/noura_alashwali/) . 
 
![Sample of dataset](./images/sample_dog_output.png)


 since aur dataset wes small ( 114 paintings  ) we use many augmentation techniques to increase the diversity of data available for training models . one of them is   [The Color Transfer Algorithm](https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/) 
 
 ![Sample of color swap](./images/color_swap.png)
 
  the pre-trained model is included you can download it from Model folder in this repo . Below are some of examples of what it can generate . Don't sell the outputs of the pre-trained models .
  
 ![Sample of result](./images/result.png)
 
 We will  also train image resolution model to increase output size from (480 *270 ) to (1920*1080) pixel 
 
 

## Project Instructions

### Instructions

1. Install dependencies in requirmnet.txt


2. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/bdourE/TheBeautyOfBelonging
	```
3. Download the [Saudi Heritage Archicture dataset](https://drive.google.com/open?id=1VNvM5xlqKkJr1-T7mI1jaF3m6f_sO5jR).  Unzip the folder and place it in the repo, at folder Dataset .  

4. Make sure you have already installed the necessary Python packages according to the requirment.txt in repository.

5. Open a terminal window and navigate to the project folder. use the following command to get generative artwork . generated atwork save in results folder.
	
	```
		python generateArtwork.py
	```
    
6. if you like to create video of latent spece interpolotion , use the following command .
    
    ```
    python latentSpace.py
    ```
    
7. if you want to make exta training .

    ```
    python train.py
    ```


## (Optionally) Accelerating the Training Process 

If your code is taking too long to run, you will need to switch to running your code on a GPU.  If you'd like to use a GPU, you can use [spell](https://web.spell.run/refer/bdour)


##  Contact me 

*  email: b.e_91@yahoo.com

*  LinkedIn: [Bdour Ebrahim](https://www.linkedin.com/in/bdour-ebrahim-682b17145/)
