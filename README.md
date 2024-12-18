### For the convenience of review, we have directly saved the proposed dataset MutationTeleFraud in Shannxi_Graph\raw\. (Except for normal users' VOC records, it cannot be uploaded due to its large size.)

### Run on our dataset：

Data Preparation: Firstly, place the files from https://pan.baidu.com/s/1UiNvhFL9ybp2CrASSrQLyQ (提取码: Shan) into Shannxi_Graph\raw\

Use the .sh file for direct training and testing: nohup sh run_models.sh &

The optimal results on the validation set and testing set will be automatically saved in the results.csv file.

### Run on open dataset

Data Preparation: Firstly, place the files from https://pan.baidu.com/s/1syw-9lN5rU7q8iioaj66WA (提取码: Sich) into Shannxi_Graph\raw\

(The results will be saved in the WandB project, and you need to log in to the website to view them. We trained for 20 epochs on the public dataset and directly tested the model after training (following the previous work)：

python Ours_open.py --mode 06

### Run GAT-COBO (IEEE Transactions on Big Data)

We tested it on two datasets based on open-source code (https://github.com/xxhu94/GAT-COBO). Need a new environment to run (The original codebase did not provide explicit dependency files. After testing, we found that the following versions are compatible and can run successfully.)：torch=2.0.0, torchdata=0.6.0, python=3.11.3,  pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html

cd GAT-COBO

python main.py

