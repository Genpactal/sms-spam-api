from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def main():
    folder = Path(__file__).resolve().parent
    data_file = folder / "data" / "SMSSpamCollection"
    if not data_file.exists():
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        with urlopen(url, timeout=60) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                data_file.parent.mkdir(exist_ok=True)
                data_file.write_bytes(archive.read("SMSSpamCollection"))

    # Remove duplicate messages before splitting to keep them out of both sets.
    messages = {}
    for line in data_file.read_text(encoding="utf-8").splitlines():
        label, message = line.split("\t", 1)
        messages[message.strip()] = "spam" if label == "spam" else "not_spam"

    x_train, x_test, y_train, y_test = train_test_split(
        list(messages), list(messages.values()),
        test_size=0.2, random_state=42, stratify=list(messages.values()),
    )
    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
    )
    model.fit(x_train, y_train)
    print(f"Training messages: {len(x_train)}; test messages: {len(x_test)}")
    print(classification_report(y_test, model.predict(x_test), digits=3))
    joblib.dump(model, folder / "model.joblib")
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
