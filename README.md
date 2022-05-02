# PUGeoNet_pytorch  
[Tensorflow version](https://github.com/ninaqy/PUGeo)
## Data Preparation  
run [prepare_data/mesh2ply.py](prepare_data/mesh2ply.py) and [prepare_data/ply2patch.py](prepare_data/ply2patch.py) to prepare the training and testing data.  

## Training
run [main.py](main.py) to training model with different upsample ratios.  

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
