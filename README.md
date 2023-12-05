# Go-strength-imitation-and-play-style-recognition-competition
## Environment Details 
* Operating System: Ubuntu 20.04.6 LTS
* programming language: Python 3.7.16
* Anaconda 1.12.0
  
## Requirements: 
* PyTorch 1.13.1
* pytorch-cuda 11.6
* tqdm 
* scikit-learn 
* numpy 
* einops 0.3.0
* tensorboard 2.11.2

## Environment setup
According to environment setup.txt create and activate the conda environment:

    conda env create -f environment.yaml
    conda activate team4762


如果沒辦法以上面指令建立虛擬環境，可以查看手動建立環境.txt檔的說明，按照裡面的步驟建立環境。

## Evaluation of our results
team4762 final models資料夾中儲存的是訓練好的最終模型，執行team4762 evaluation.ipynb可評估模型準確率以及輸出submission.csv檔。
## Trainning
如果要使用模型進行訓練，可依照所要訓練的資料集，執行my Dan Trainning.ipynb、my kyu Trainning.ipynb、my PlayStyle Trainning.ipynb來進行訓練。
訓練好的模型會分別儲存在dan_models, kyu_models, playstyle_models資料夾裡。
## Evaluation of trainning
訓練完成後如要進行評估，執行trainning evaluation.ipynb，程式會從dan_models, kyu_models, playstyle_models資料夾裡取得模型，以評估模型準確率。
如要輸出submission.csv檔，可透過my Create Private Upload CSV.ipynb及my Create Public Upload CSV.ipynb來輸出各別的submission.csv檔。
