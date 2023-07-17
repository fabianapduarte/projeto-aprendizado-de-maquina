import argparse
import decisionTrees as dt

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-a", "--alg", required=True, help="Classifier algorithm")
  args = vars(ap.parse_args())

  if (args["alg"] == "dt"):
    dt.run()
  else:
    print("Algorithm not supported.")

if __name__ == '__main__':
  main()