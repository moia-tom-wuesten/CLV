# CLV


The goal of this repo is to provide a new framework for predicting CLV and comparing results between the DL frameworks Tensorflow and Pytorch.

## Comparison between Tensorflow and Pytorch

### Example Retail Bank Data

Retail bank transaction data taken from https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions (trans.csv file only). This is a small dataset (fast training), with a strong aggregate monthly pattern.
![Alt text for image](src/CLV/plots_valendin/calibration_holdout.png)

#### Prediction with Pytorch 

![Alt text for image](src/CLV/plots_pytorch/bank_actuals_full_prediction.png)

![Alt text for image](src/CLV/plots_pytorch/bank_actuals_insample_prediction.png)

![Alt text for image](src/CLV/plots_pytorch/bank_actuals_outsample_prediction.png)

#### Prediction with Tensorflow

![Alt text for image](src/CLV/plots_valendin/actuals_prediction.png)

![Alt text for image](src/CLV/plots_valendin/insample_prediction.png)

