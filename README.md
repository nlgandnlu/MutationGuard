### For the convenience of review, we have directly attached the datasets and codes in the 'Supplementary Material'.

### Run on our dataset：

1. Data Preparation: Firstly, place the files from https://pan.baidu.com/s/1UiNvhFL9ybp2CrASSrQLyQ (提取码: Shan) into Mutation_Experiment\Shannxi_Graph\raw\

2. Use the .sh file for direct training and testing: nohup sh run_models.sh &

The optimal results on the validation set and testing set will be automatically saved in the results.csv file.

### Run on open dataset
1. Data Preparation: Firstly, place the files from https://pan.baidu.com/s/1syw-9lN5rU7q8iioaj66WA (提取码: Sich) into Mutation_Experiment\Shannxi_Graph\raw\

(The results will be saved in the WandB project, and you need to log in to the website to view them. We trained for 20 epochs on the public dataset and directly tested the model after training (following the previous work)：

python Ours_open.py --mode 06
