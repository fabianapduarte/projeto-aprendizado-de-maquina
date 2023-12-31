from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
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

  model = DecisionTreeClassifier(random_state = 84)
  print("[INFO] Treinando modelo...")
  model.fit(trainData, trainLabels)

  print("[INFO] Avaliando modelo...")
  predictions = model.predict(testData)
  print(classification_report(testLabels, predictions))
