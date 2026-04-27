## TeDi: Discrete Style Modeling Framework for Text Generation

### ⚙️ Setup:
Install the necessary dependencies to run the project.
```
sh setup.sh
```

### 📌 Model Training:

- for Amazon dataset
```
python main.py \
  --mode train \
  --dataset Amazon \
  --epochs 20 \
  --batch_size 64 \
  --codebook 512 \
  --codebook_dim 256 \
  --embeds uniform \
  --lr_d 0.0005 \
  --lr_m 0.0003 \
  --cycle True \
  --gpu <DEVICE_ID>
```
- for Yelp dataset
```
python main.py \
  --mode train \
  --dataset Yelp \
  --epochs 20 \
  --batch_size 64 \
  --codebook 512 \
  --codebook_dim 256 \
  --embeds uniform \
  --lr_d 0.0006 \
  --lr_m 0.0004 \
  --cycle True \
  --gpu <DEVICE_ID>
```

***

### ⚙️ Pre-trained Models:
You can download pre-trained weights to perform inference without training the model from scratch.
1. full model (instant inference)
```
sh download.sh
```
2. preliminary model
```
sh download_train.sh
```
3. evaluation file (required for PPL metrics)
```
sh download_eval.sh
```


### 📌 Model Inference:
Perform inference to generate text with target styles using the trained model.

▼ Usage example (style 1 → style 5):
```
python main.py \
  --mode test \
  --dataset <Amazon|Yelp> \
  --batch_size 1 \
  --t_date <YYMMDD> \
  --t_time <HHMM> \
  --codebook 512 \
  --codebook_dim 256 \
  --embeds uniform \
  --usage test1 \
  --option test5
```

--usage: source style

--option: target style

### 📌 Model Evaluation:
Evaluate the performance of the generated text using various metrics (BLEU, BertScore, PPL).
```
python evaluator.py \
  --name <FILE_NAME> \
  --dataset <Amazon|Yelp> \
  --save <SAVE_DIR> \
  --data_path <INPUT_DATA_DIR> \
  --ref_path <REFERENCE_DATA_DIR> \
  --metric s_self bert ppl
```