import pandas as pd

def add_result(model, class_, acc, loss, f1):
    results = pd.read_csv("../results.csv")
    results = pd.concat([results, pd.DataFrame({
        "model": model,
        "class": class_,
        "loss": loss,
        "f1score": f1,
        "accuracy": acc 
    }, index=[0])])
    results.to_csv("../results.csv")