# Gerekli kütüphanelerin yüklenmesi
!pip install transformers datasets torch scikit-learn

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
import numpy as np
from sklearn.metrics import accuracy_score

# 1. VERİ YÜKLEME VE ÖN İŞLEME
# IMDb veri kümesini yükleme
dataset = load_dataset("imdb")

# Daha dengeli ve temsili bir örneklem seçmek için:
train_data = dataset["train"].shuffle(seed=42).select(range(5000))  # 5,000 örnek
test_data = dataset["test"].shuffle(seed=42).select(range(1000))    # 1,000 örnek

# 2. MODEL VE TOKENIZER YÜKLEME
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. VERİ TOKENIZASYONU
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512  # Maksimum uzunluk belirtildi
    )

tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)

# 4. METRIK HESAPLAMA FONKSIYONU
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 5. EĞITIM AYARLARI
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,  # Daha büyük batch boyutu
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-5,  # Optimize edilmiş learning rate
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=100
)

# 6. TRAINER OLUŞTURMA VE EĞITIM
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics  # Accuracy metriği eklendi
)

trainer.train()

# 7. DEĞERLENDIRME
eval_results = trainer.evaluate()
print(f"Test Accuracy: {eval_results['eval_accuracy']}")  # Doğru metrik ismi

# 8. MODELI KULLANMA
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1  # GPU kullanımı
)

# Test örneği
result = classifier("This movie was absolutely fantastic!")
print(result)  # Örnek çıktı: [{'label': 'LABEL_1', 'score': 0.98}]