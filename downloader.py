import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ARABIC_ID = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
ENGLISH_ID = "mrm8488/deberta-v3-small-finetuned-sst2"
MULTI_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

def download(model_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    tok.save_pretrained(out_dir.as_posix())
    mdl.save_pretrained(out_dir.as_posix())

def main() -> None:
    base = Path("./models").resolve()
    download(ARABIC_ID, base / "arabic")
    download(ENGLISH_ID, base / "english")
    download(MULTI_ID, base / "multi")
    print((base / "arabic").as_posix())
    print((base / "english").as_posix())
    print((base / "multi").as_posix())

if __name__ == "__main__":
    main()
