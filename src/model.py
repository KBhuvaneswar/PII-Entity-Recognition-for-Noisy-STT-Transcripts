from transformers import AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str, dropout: float = 0.1):
    # DistilBERT uses 'dropout' instead of 'hidden_dropout_prob'
    if 'distilbert' in model_name.lower():
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            dropout=dropout,
            attention_dropout=dropout,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
    return model
