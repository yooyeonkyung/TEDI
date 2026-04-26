# TeDi: Discrete Style Modeling Framework for Text Generation

### ⚙️ To install python dependencies before train:

```
sh setup.sh
```

### 📌 To train model:

```
python main.py --mode train --dataset [Amazon|Yelp] --epochs 20 --batch_size 64 --codebook 512 --codebook_dim 256 --embeds uniform --lr_d 0.0005 --lr_m 0.0003 --cycle True --gpu 4
```

### 📌 To inference model:

```
python main.py --mode test --dataset [Amazon|Yelp] --batch_size 1 --t_date [YYMMDD] --t_time [HHMM] --codebook 512 --codebook_dim 256 --embeds uniform --usage test1 --option [file name]
```