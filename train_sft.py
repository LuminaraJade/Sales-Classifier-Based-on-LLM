###初始化设定和随机种子
import os
os.environ["CUDAVISIBLE_DEVICES"] = "0, 1"

import torch
import numpy as np
import pandas as pd
import random
import json
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def get_dataset_from_json(json_path, cols):
    with open(json_path, "r") as file:
        data = json.load(file)
        df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df[cols], split='train')
    return dataset


# 加载数据集
cols = ['content', 'label', '标注类别']
train_ds = get_dataset_from_json('./data/goods_train.json', cols)
logger.info(f"TrainData num: {len(train_ds)}")
valid_ds = get_dataset_from_json('./data/goods_valid.json', cols)
logger.info(f"ValidData num: {len(valid_ds)}")
test_ds = get_dataset_from_json('./data/goods_test.json', cols)
logger.info(f"TestData num: {len(test_ds)}")


# 定义正负例和标签的map
id2label = {0: "负例", 1: "正例"}
label2id = {v:k for k,v in id2label.items()}


from modelscope import AutoTokenizer, DataCollatorwithPadding

model_name_or_path = "/mnt/workspace/.cache/modelscope/hub/qwen/Qwen2___5-3B-Instruct"
model_name = model_name_or_path.split("/")[-1]
print(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

MAX_LEN = 1024
txt_colname = 'content'

def preprocess_function(examples):
    # padding后处理效率不高,需要动态batch padding
    return tokenizer(examples[txt_colname], max_length=MAX_LEN, padding=True, truncation=True)

tokenized_train = train_ds.map(preprocess_function, num_proc=64, batched=True)
tokenized_valid = valid_ds.map(preprocess_function, num_proc=64, batched=True)


def evals(test_ds, model):
    k_list = [x[txt_colname] for x in test_ds]
    model.eval()

    k_result = []
    for idx, txt in tqdm(enumerate(k_list)):
        model_inputs = tokenizer([txt], max_length=MAX_LEN, truncation=True, return_tensors="pt").to(model.device)
        logits = model(**model_inputs).logits
        res = int(torch.argmax(logits, axis=1).cpu())
        k_result.append(id2label.get(res))

    y_true = np.array(test_ds['label'])
    y_pred = np.array([label2id.get(x) for x in k_result])
    return y_true, y_pred

def compute_metrics(eval_pred):
    predictions, label = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"f1": f1_score(y_true=label, y_pred=predictions, average='weighted')}

def compute_valid_metrics(eval_pred):
    predictions, label = eval_pred
    y_true, y_pred = label, predictions
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy}')
    metric_types = ['micro', 'macro', 'weighted']
    for metric_type in metric_types:
        precision = precision_score(y_true, y_pred, average=metric_type) 
        recall = recall_score(y_true, y_pred, average=metric_type)
        f1 = f1_score(y_true, y_pred, average=metric_type)
        print(f'{metric_type} Precision: {precision}')
        print(f'{metric_type} Recall: {recall}')
        print(f'{metric_type} F1 Score: {f1}')


### 训练
rank = 64
alpha = rank*2
training_args = TrainingArguments(
    output_dir=f"./output/{model_name}/seqence_classify/",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 去掉就是SFT
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=False,
#     r=rank,
#     lora_alpha=alpha,
#     lora_dropout=0.1
# )


model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash attention2"
)

model.config.pad_token_id = tokenizer.pad_token_id

# 去掉下面这行就是SFT
# model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)



logger.info(f"start Trainingrank: {rank}")
trainer.train()

logger.info(f"Valid Set, rank: {rank}")
y_true, y_pred = evals(valid_ds, model)
metrics = compute_valid_metrics((y_pred, y_true))
logger.info(metrics)

logger.info(f"Test Set, rank: {rank}")
y_true, y_pred = evals(test_ds, model)
metrics = compute_valid_metrics((y_pred, y_true))
logger.info(metrics)

saved_model = model.merge_and_unload()
saved_model.save_pretrained('./model/qwen2-3b/goodscls')
