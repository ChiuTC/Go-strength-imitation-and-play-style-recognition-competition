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

    conda create --name team4762 python=3.7.16
    conda activate team4762
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install tqdm
    pip install -U scikit-learn
    pip install numpy
    pip install einops==0.3.0
    pip install ipykernel
    pip install tensorboard
## Required download 
* 請到google drive下載我們訓練好的模型，將team4762 final models資料夾放到本專案資料夾中以用於評估，資料夾中應有三個模型，  
  下載後請將資料夾後的時間戳刪除，例如: team4762 final models-20231205T080404Z-001 改為 team4762 final models，以利程式讀取。
  https://drive.google.com/drive/folders/1y6WSu6H1D5sF26FGwu3aaL5-dOll5vH4?usp=sharing
  
      └── Go-strength-imitation-and-play-style-recognition-competition-main
        └── team4762 final models
            ├── model2000.pth
            ├── model2300.pth
            └── best_PS_model.pth
* 訓練集測試資料集，也請放到本專案資料夾中，與官方所給的資料夾名稱及存放路徑應該相同。
  
      └── Go-strength-imitation-and-play-style-recognition-competition-main
        └── 29_Private Testing Dataset_Public and Private Submission Template_v2
          └── 29_Private Testing Dataset_v2
            ├── dan_test_private.csv
            ├── kyu_test_private.csv
            └── play_style_test_private.csv
        └── 29_Public Testing Dataset_Public Submission Template_v2
          └── 29_Public Testing Dataset_v2
            ├── dan_test_public.csv
            ├── kyu_test_public.csv
            └── play_style_test_public.csv
        └── 29_Training Dataset
          └── Training Dataset
            ├── dan_train.csv
            ├── kyu_train.csv
            └── play_style_train.csv
## Documentation and Code Explanation
* team4762 final models資料夾中儲存的是我們訓練好的最終模型。
* team4762 evaluation.ipynb 為評估我們訓練好的最終模型模型之準確率以及輸出submission.csv檔。
* my Dan Trainning.ipynb、my kyu Trainning.ipynb、my PlayStyle Trainning.ipynb 依序為對三個資料集進行訓練的程式碼。
* trainning evaluation.ipynb 為用來評估訓練結果之程式碼。
* my Create Private Upload CSV.ipynb及my Create Public Upload CSV.ipynb 用來輸出訓練完成後各別的submission.csv檔。
* my_data.py 為資料處理程式碼。
* my_model.py 為模型實現的程式碼。
* logs資料夾裡儲存的是使用tensorboard之訓練過程數據。

## Evaluation of our results
team4762 final models資料夾中儲存的是我們訓練好的最終模型，執行team4762 evaluation.ipynb 可評估模型準確率以及輸出submission.csv檔。
## Trainning
如果要進行訓練，可依照所要訓練的資料集，執行my Dan Trainning.ipynb、my kyu Trainning.ipynb、my PlayStyle Trainning.ipynb來進行訓練，   
訓練好的模型會分別儲存在dan_models, kyu_models, playstyle_models資料夾裡。  
在終端執行下面指令可利用tensorboard查看訓練過程中，訓練集和驗證集的loss及準確率。  
`tensorboard --logdir=./logs`
## Evaluation of trainning
訓練完成後如要進行評估，執行trainning evaluation.ipynb，程式會從dan_models, kyu_models, playstyle_models資料夾裡取得模型，以評估模型準確率。  
如果要輸出submission.csv檔，可透過my Create Private Upload CSV.ipynb及my Create Public Upload CSV.ipynb來輸出各別的submission.csv檔。
