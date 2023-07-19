import os
import cv2
import numpy as np

def extractFeatures(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  imgNormalized = cv2.normalize(gray, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  grayValues = imgNormalized.flatten()

  (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
  colorStats = np.concatenate([means, stds]).flatten()

  return np.hstack([colorStats, grayValues])

def getLabelsAndData(folderPath, label):
  labels = []
  data = []

  images = os.listdir(folderPath)

  for image_name in images:
    image = cv2.imread(folderPath + '/' + image_name)
    resizedImage = cv2.resize(image, (150, 150))
    features = extractFeatures(resizedImage)
    labels.append(label)
    data.append(features)
  
  return [labels, data]