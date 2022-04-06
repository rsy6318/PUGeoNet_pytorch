# PUGeoNet_pytorch  
## Data Preparation  
run [prepare_data/mesh2ply.py](prepare_data/mesh2ply.py) and [prepare_data/ply2patch.py](prepare_data/ply2patch.py) to prepare the training and testing data.  
You can download our prepared data [here](https://tjueducn-my.sharepoint.com/:f:/g/personal/rsy6318_tju_edu_cn/EqwtghjBbURFvM6eymC8C3cBdA4aE1EaaPuitk02YwGP8w?e=cLbdsm).
## Training
run [main.py](main.py) to training model with different upsample ratios.  
Our trained models are saved in [logx4](logx4), [logx8](logx8), [logx12](logx12) and [logx16](logx16).  
## Testing
### Upsampling point cloud
run [evaluation/eval.py](evaluation/eval.py)  
### Evaluation metric

First you need to comple the [P2F evaluation](https://github.com/yulequan/PU-Net).  
```  
cd evaluation/p2f  
cmake .  
make  
```  
Then run [evaluation/cal_p2f.py](evaluation/cal_p2f.py) and [evaluation/cal_metric.py](evaluation/cal_metric.py) to compute the metric(CD, HD, P2F).  
