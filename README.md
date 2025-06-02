# [WWW '25] Leveraging Refined Negative Feedback with LLM for Recommender Systems

[![View Paper](https://img.shields.io/badge/View%20Paper-PDF-red?logo=adobeacrobatreader)](https://dl.acm.org/doi/10.1145/3701716.3715538)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14808051.svg)](https://doi.org/10.5281/zenodo.14808051)

# Requirements
python 3.9.20, cuda 11.8, and the following installations:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install pandas
pip install scikit-learn
pip install transformers
pip install ptyprocess ipykernel pyzmq -U --force-reinstall
```

# Run
##### ML-100K
```
python main.py --dataset ML-100K --r 100
```
##### ML-1M
```
python main.py --dataset ML-1M --r 250
```
##### Netflix-1M
```
python main.py --dataset Netflix-1M --r 300
```

# Settings
We implement the model with PyTorch Geometric.
The batch size is set to 1024, embedding dimension to 64, the number of layers is 4, learning rate to 1e-3, and the training runs for 1000 epochs.
The model is evaluated on the validation set at each epoch, and early stopping is applied if no improvement in recall@20 is observed over 50 consecutive epochs.
The experiments are conducted using a single NVIDIA RTX A6000 GPU.

# Compare with our results
The **results4comparison** folder contains the results of our experiment.
Each file includes the loss and performance metrics for every epoch, as well as the hyperparameters, dataset statistics, and training time.
You can compare our results with your own reproduced results.

# Datasets
Download **ml-100k.inter** and **ml-100k.item** from [here](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-100k.zip).

Download **ml-1m.inter** and **ml-1m.item** from [here](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip).

Download **netflix.inter** from [here](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Netflix/netflix.zip).

Download **movie_titles.csv** from [here](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data?select=movie_titles.csv).

Download **netflix_genres.csv** from [here](https://github.com/tommasocarraro/netflix-prize-with-genres).

**If you would like to access the dataset refined by LLM according to our methodology, you can download it [here](https://github.com/Chanwoo-Jeong-2000/ReFINe_plus/tree/main/dataset).**

# Citation
If you find ReFINe useful for your research or development, please cite the following our papers:
```
@inproceedings{jeong2025leveraging,
  title={Leveraging Refined Negative Feedback with LLM for Recommender Systems},
  author={Jeong, Chanwoo and Kang, Yujin and Cho, Yoon-Sik},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={1028--1032},
  year={2025}
}
```

# Acknowledgments
This research was supported by the MSIT(Ministry of Science and ICT), Korea, under the ITRC(Information Technology Research Center) support program(IITP-2025-RS-2024-00438056) and the Artificial Intelligence Graduate School Program of Chung-Ang Univ.(No. 2021-0-01341) both supervised by the IITP(Institute for Information & Communications Technology Planning & Evaluation).
