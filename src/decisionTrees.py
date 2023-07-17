import mahotas
import os
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def extractFeatures(image):
  (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
  colorStats = np.concatenate([means, stds]).flatten()
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  haralick = mahotas.features.haralick(gray).mean(axis = 0)
  return np.hstack([colorStats, haralick])

def getLabelsAndData(folderPath, label):
  labels = []
  data = []

  images = os.listdir(folderPath)

  for image_name in images:
    image = cv2.imread(folderPath + '/' + image_name)
    features = extractFeatures(image)
    labels.append(label)
    data.append(features)
  
  return [labels, data]

def run():
  (trainHealthyLabels, trainHealthyData) = getLabelsAndData('dataset/Training/healthy_corals', 'healthy')
  (trainBleachedLabels, trainBleachedData) = getLabelsAndData('dataset/Training/bleached_corals', 'bleached')
  trainData = trainHealthyData + trainBleachedData
  trainLabels = trainHealthyLabels + trainBleachedLabels

  (testHealthyLabels, testHealthyData) = getLabelsAndData('dataset/Testing/healthy_corals', 'healthy')
  (testBleachedLabels, testBleachedData) = getLabelsAndData('dataset/Testing/bleached_corals', 'bleached')
  testData = testHealthyData + testBleachedData
  testLabels = testHealthyLabels + testBleachedLabels

  model = DecisionTreeClassifier(random_state = 84)
  print("[INFO] Treinando modelo...")
  model.fit(trainData, trainLabels)

  print("[INFO] Avaliando modelo...")
  predictions = model.predict(testData)
  print(classification_report(testLabels, predictions))
