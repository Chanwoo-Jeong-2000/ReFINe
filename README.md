# [WWW '25] Leveraging Refined Negative Feedback with LLM for Recommender Systems

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

# Settings
We implement the model with PyTorch Geometric.
The batch size is set to 1024, embedding dimension to 64, the number of layers is 4, learning rate to 1e-3, and the training runs for 1000 epochs.
The model is evaluated on the validation set at each epoch, and early stopping is applied if no improvement in recall@20 is observed over 50 consecutive epochs.
The experiments are conducted using a single NVIDIA RTX A6000 GPU.

# Run
