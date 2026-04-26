import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset # 혹은 직접 만든 데이터셋 로딩

# 1. 모델과 토크나이저가 저장된 폴더 경로
model_path = "/Users/ewonjin/Desktop/4-1/캡스톤디자인2/파이썬 파일/phishing_bert_model" 

# 2. 저장된 모델 및 토크나이저 불러오기
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. 데이터셋 준비 (학습 때 사용한 코드를 그대로 활용하세요)
# 예시: csv 파일로 되어 있는 경우
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 학습 때 사용했던 데이터를 다시 불러온다고 가정 (예: test.csv 또는 train.csv)
raw_datasets = load_dataset('csv', data_files={'test': '상세분류추가-1로.csv'})
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 4. 평가를 위한 설정 (로그를 남기지 않으려면 최소한의 설정만)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
)

# 5. Trainer 객체 생성 및 평가 실행
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["test"], # 이 부분이 데이터셋을 넣는 곳입니다.
)

# 최종 성능 확인
metrics = trainer.evaluate()
print(metrics)