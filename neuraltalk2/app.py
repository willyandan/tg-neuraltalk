from flask import Flask
from eval import Neuraltalk2
nt2 = Neuraltalk2()
app = Flask(__name__)


@app.route("/predict")
def predict():
  print(nt2.eval_image("png/kids.png"))

if __name__ == "__main__":
  app.run()