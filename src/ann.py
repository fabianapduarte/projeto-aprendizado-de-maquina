from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import utils as utils

def run():
  (trainHealthyLabels, trainHealthyData) = utils.getLabelsAndData('dataset/Training/healthy_corals', 'healthy')
  (trainBleachedLabels, trainBleachedData) = utils.getLabelsAndData('dataset/Training/bleached_corals', 'bleached')
  trainData = trainHealthyData + trainBleachedData
  trainLabels = trainHealthyLabels + trainBleachedLabels

  (testHealthyLabels, testHealthyData) = utils.getLabelsAndData('dataset/Testing/healthy_corals', 'healthy')
  (testBleachedLabels, testBleachedData) = utils.getLabelsAndData('dataset/Testing/bleached_corals', 'bleached')
  testData = testHealthyData + testBleachedData
  testLabels = testHealthyLabels + testBleachedLabels

  model = MLPClassifier(activation='logistic', max_iter=100, random_state=84, verbose=True, early_stopping=True, validation_fraction=0.2, solver="adam")
  print("[INFO] Treinando modelo...")
  model.fit(trainData, trainLabels)

  print("[INFO] Avaliando modelo...")
  predicted = model.predict(testData)
  print(classification_report(testLabels, predicted))