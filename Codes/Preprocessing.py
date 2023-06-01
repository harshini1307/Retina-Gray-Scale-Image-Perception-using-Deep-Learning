
import matplotlib.pyplot as plt
import glob
import SimpleITK as sitk
import numpy as np
import os
import skimage
from skimage.io import imread_collection
import numpy as np


## For Training:

ground_dir= "C:/Users/LENOVO/Documents/Harsha/14.12.22/retina-segmentation-CNN-/training/1st_manual/*.gif"
img_dir="C:/Users/LENOVO/Documents/Harsha/14.12.22/retina-segmentation-CNN-/training/images/*.tif"
mask_dir= "C:/Users/LENOVO/Documents/Harsha/14.12.22/retina-segmentation-CNN-/training/mask/*.gif"

train_image_set= imread_collection(img_dir)


ground_truth_set= imread_collection(ground_dir)
mask_set=imread_collection(mask_dir)

train_image_list=list(train_image_set)
ground_truth_list=list(ground_truth_set)
mask_list= list(mask_set)


## Preprocessing steps- normalising and taking only the green channel
train_list=[]
for i in range(len(train_image_set)):
    image= sitk.GetImageFromArray(train_image_list[i])
    normalised_image= sitk.NormalizeImageFilter()
    normalised_image=normalised_image.Execute(image)
    normalised_image=sitk.GetArrayFromImage(normalised_image)
    green_image= normalised_image[:,:,1]
    train_list.append(green_image)



print ('train_data')
np.savez('data_train_final.npz',mask_set=mask_set,ground_truth_list=ground_truth_list,train_list=train_list)
print ('saved')

## For test Images:

ground_test_dir= "C:/Users/LENOVO/Documents/Harsha/14.12.22/retina-segmentation-CNN-/test/1st_manual/*.gif"
img_test_dir="C:/Users/LENOVO/Documents/Harsha/14.12.22/retina-segmentation-CNN-/test/images/*.tif"
mask_test_dir= "C:/Users/LENOVO/Documents/Harsha/14.12.22/retina-segmentation-CNN-/test/mask/*.gif"

test_image_set= imread_collection(img_test_dir)
ground_truth_set= imread_collection(ground_test_dir)
test_mask_set=imread_collection(mask_test_dir)

test_image_list=list(test_image_set)
test_ground_truth_list=list(ground_truth_set)
test_mask_list= list(test_mask_set)
test_mask_list=np.asarray(test_mask_list)

test_list=[]
for i in range(len(test_image_set)):
    image= sitk.GetImageFromArray(test_image_list[i])
    normalised_image= sitk.NormalizeImageFilter()
    normalised_image=normalised_image.Execute(image)
    normalised_image=sitk.GetArrayFromImage(normalised_image)
    green_image= normalised_image[:,:,1]
    test_list.append(green_image)

test_list=np.asarray(test_list)
test_ground_truth_list=np.asarray(test_ground_truth_list)
test_mask_list=np.asarray(test_mask_list)

print ('test_data')
np.savez('data_test_final.npz',test_list=test_list,test_ground_truth_list=test_ground_truth_list,test_mask_list=test_mask_list)

print ('saved')
