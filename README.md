# PUGeoNet_pytorch  
## Data Preparation  
run prepare_data/mesh2ply.py and prepare_data/ply2patch.py to prepare the training and testing data.
## Training
run [main.py](https://github.com/rsy6318/PUGeoNet_pytorch/blob/main/main.py) to training model with different upsample ratios.
## Testing
First you need to comple the [P2F evaluation](https://github.com/yulequan/PU-Net).  
```  
cd evaluation/p2f  
cmake .  
make  
```  
Secondly run /eval.py to upsample the test point cloud.  
Then run /evaluation/cal_p2f.py to compute the P2F between point cloud and mesh.  
Finally, run /evaluation/cal_metric.py to compute the metric(CD, HD, P2F).  
