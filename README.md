# TeDi: Discrete Style Modeling Framework for Text Generation

### ⚙️ Setup:

```
sh setup.sh
```

### 📌 Model Training:

```
python main.py --mode train --dataset <Amazon|Yelp> --epochs 20 --batch_size 64 --codebook 512 --codebook_dim 256 --embeds uniform --lr_d 0.0005 --lr_m 0.0003 --cycle True --gpu 4
```

***

### 📌 Model Inference:

```
python main.py --mode test --dataset <Amazon|Yelp> --batch_size 1 --t_date [YYMMDD] --t_time [HHMM] --codebook 512 --codebook_dim 256 --embeds uniform --usage test1 --option [file name]
```

### 📌 Model Evaluation:

```
python evaluator.py \
  --name <MODEL_NAME> \
  --dataset <Amazon|Yelp> \
  --save <SAVE_DIR> \
  --data_path <INPUT_DATA_DIR> \
  --ref_path <REFERENCE_DATA_DIR> \
  --metric self s_self bert ppl
```