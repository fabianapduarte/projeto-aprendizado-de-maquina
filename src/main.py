import argparse
import decisionTrees as dt
import ann as ann

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("-a", "--alg", required=True, help="Algoritmo de classificação")
  args = vars(ap.parse_args())

  if (args["alg"] == "dt"):
    dt.run()
  elif (args["alg" == "ann"]):
    ann.run()
  else:
    print("Algoritmo não reconhecido.")

if __name__ == '__main__':
  main()