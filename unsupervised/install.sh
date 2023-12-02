conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install pytorch=1.11.0 cudatoolkit=11.3
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html 
pip install rdkit -y
conda install tqdm -y
conda install tensorboardx -y
pip install networkx -y