import joblib

model = joblib.load("esg_text_model.joblib")
labels = sorted(set(model.classes_))
print("ESG Text Classifier loaded. Labels:", labels)
print("Type a sentence (or 'quit'):")

while True:
    txt = input("> ")
    if txt.strip().lower() == "quit":
        break
    pred = model.predict([txt])[0]
    print("→", pred)
