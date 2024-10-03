import os

valPath = "C:/Users/F. Grasso/pytorch_img_class/custom_image_classifier_pytorch/dataset/market/val/"
trainPath = "C:/Users/F. Grasso/pytorch_img_class/custom_image_classifier_pytorch/dataset/market/train/"

val_listFolder = os.listdir(valPath)
train_listFolder = os.listdir(trainPath)
val_Image = []

for folder in val_listFolder:
    val_Image.append(os.listdir(valPath + folder))

for folder in train_listFolder:
    for image in os.listdir(trainPath + folder):
        for lst in val_Image:
            if image in lst:
                print(image)

