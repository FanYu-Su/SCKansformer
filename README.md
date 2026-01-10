# SCKansformer Ofiicial Repository Readme

We propose a novel fine-grained classification model, SCKansformer, for bone marrow blood cells. The SCKansformer model primarily comprises three parts: Kansformer Encoder, SCConv Encoder and Global-Local Attention Encoder. The overall architecture of our proposed SCKansformer model: 
![image](img/OverallFramework.png)

## 1. Environment

- Please clone this repository and navigate to it in your terminal.
- Then prepare an environment with python=3.8, and then use the command `pip install -r requirements.txt` for the dependencies.

## 2. Train/Test

- Put the BMCD-FGCD dataset(PBC/ALL-IDB dataset) into data/BM_data(PBC_data/ALL_data), then split folders by category and modify the `class_indices.json` file.
- Run `train_SCkansformer_cell.py` to Train/Test in data/BM_data.
- The batch size we used is 40 for V100. If you do not have enough GPU memory, the bacth size can be reduced to 30 for GeForce RTX 4090 or 6 to save memory.

## 3. BMCD-FGCD dataset

In collaboration with the Department of Hematology at Zhejiang Hospital in Hangzhou, Zhejiang Province, our team has established the Bone Marrow Cell Dataset for Fine-Grained Classification (BMCD-FGCD),
containing over 10,000 data points across nearly forty classifications. We have made our private BMCD-FGCD dataset available to other researchers, contributing to the field's advancement.
If you want to use our private dataset, please cite our article.

Download link is available at [https://drive.google.com/file/d/1hOmQ9s8eE__nqIe3lpwGYoydR4_UNRrU/view?usp=drive_link](https://drive.google.com/file/d/1NrbK-OZCgTiFhWqeboaq8OVFXPNONegv/view?usp=sharing).

Details of our BMCD-FGCD dataset:
![image](img/Detail.png)

## 4. Establishment and Usage of our BMCD-FGCD dataset

**Workflow of the establishment of our BMCD-FGCD dataset:**

![image](img/Process.png)

**Below, we delineate the specific utility of our BMCD-FGCD dataset in various application contexts:**

- Training of Deep Learning Models and Automatic Blood Cell Identification.
- Integrated Diagnosis with Clinical Data.
- Identification of Rare and Atypical Blood Cells.

## 5. Citation

```
@article{chen2024sckansformer,
  title={Sckansformer: Fine-grained classification of bone marrow cells via kansformer backbone and hierarchical attention mechanisms},
  author={Chen, Yifei and Zhu, Zhu and Zhu, Shenghao and Qiu, Linwei and Zou, Binfeng and Jia, Fan and Zhu, Yunpeng and Zhang, Chenyan and Fang, Zhaojie and Qin, Feiwei and others},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```


# New Using Instructions
## official link: https://github.com/JustlfC03/SCKansformer 
### You can see more details about the model in the official repository, here I just give some simple instructions about how to use the code for training and testing, at the same time, I have optimized some parts of the code to make it easier to use.
### This code is based on the official repository.

## 1. 训练前必看
- 1.因为官方的使用过多模型，我这里进行了简单的优化，只需要将数据集放在data/BM_data下即可，也可以自己调整路径，由于BMCD-FGCD数据已经直接分类了test和train文件夹，所以需要将test文件夹改名为val文件夹，然后再放置进去
- 2.其中num_class官方代码默认为5，我们由于训练的class有32个（官方有40个分类，但是有8个分类无数据，故而实际为32个分类）。
- 3.这个代码优化了，默认支持断点续传的。
- 4.特别的，建议手动设定输出的save_path，尽量不要用默认的default，保证后续能够找得到断点续训的文件，续训文件被保存在对应的result/weights/save_path下的checkpoint.pth文件中
- 4.如果想使用断点续训的话，直接在命令行后面加上--resume ./results/weights/自己命名的文件夹/checkpoint.pth即可
- 5.如果不想使用混合精度训练的话，直接在命令行后面加上--use_amp即可，默认是开启的
- 6.我在notebook中给出了confusion_matrix的可视化代码，可以直接使用调用，具体使用方法可以再notebook中的文件中再去阅览，注意：需要安装jupyter notebook才能够使用。
- 7.对于原定的requirements.txt文件，我添加了一些新的包，可以直接使用pip install -r requirements.txt进行安装，推荐使用中科大的镜像源安装，部分未列出的安装包，请自行安装
- 8.对于需要续训的epoch数，需要在命令行中自行调整epochs的数量，断点续训会从上次训练的epoch数继续往下训练，例如：原来的epoch为200，如果想要继续训练100个epoch的话，需要将epochs设置为300
- 9.代码已经优化支持了多GPU并行训练，并且可以自动识别和使用多个GPU进行训练，但是需要加上--multi_gpu True参数才能开启，训练的时候默认是开启cuda训练的；
- 10.如果想要固定随机种子的话，可以加上--seed参数，默认是不固定的，训练还是推荐固定一下，代码固定seed默认为7
- 11.我在这个模型中添加了消融实验的模块，使用testing.ipynb的时候，请记住要根据模型进行调整
- 12.在使用testing.ipynb进行性能检测的时候，parallel的参数也需要进行同步调整，否则会报错，请注意
- 13.特别的，testing.ipynb已经做好了recall、precision、f1-score的计算，可以直接使用。最后也给出了train下的dataset处理过程，也可以通过这个看到图片是如何处理的。
- 14.消融实验说明：
   - kansformer_s: 代表丢弃了SCConv模块
   - kansformer_g: 代表丢弃了Global-Local Attention模块
   - kansformer_k: 代表丢弃了Kansformer模块
   - kansformer_ss: 代表只丢弃了SRU模块
   - kansformer_sc: 代表只丢弃了CRU模块
   - kansformer_sg: 代表丢弃了GLAE和SCConv模块
   - kansformer_msa: 代表只使用了Global-Local Attention模块中的Global模块（既MSA模块）+Kansformer模块
- 15.如果想要使用消融实验的模型进行训练的话，可以将--model_name后面的参数进行替换即可，例如：--model_name kansformer_sg代表使用丢弃了GLAE和SCConv模块的模型进行训练
- 16.由于官方代码并没有给出明确的参数，复现出来的结果并不理想，所以我这里仅是完善一下缺失的代码，并没有完全复现出官方论文中的结果，所以经过修改的代码也是仅供参考。

## 2.开始训练
- 直接通过python train.py --tensorboard --num_classes 32 --epochs 100 --data_path data/BM_data --device cuda:1 --seed --save_path time3tensor  --batch_size 32  可以开始训练（以上是我训练的一个例子，参数可以自行依照实际进行调整）

## 3.测试
- 建议训练完成后，最后使用notebook里面的testing.ipynb进行测试，可以直接得到confusion_matrix的可视化结果，后续也会添加每个图的热力图显示。
- data_name_dic = ['Haemocytoblast', 'Myeloblast', 'Promyelocyte', 'Neutrophilic myelocyte', 'Neutrophilic metamyelocyte', 'Neutrophilic granulocyte band form', 'Neutrophilic granulocyte segmented form', 'Acidophil in young', 'Acidophil late young', 'Acidophillic rod-shaped nucleus', 'Eosinophillic phloem granulocyte', 'Basophillic in young', 'Basophillic late young', 'Basophillic rod-shaped nucleus',
                 'Basophllic lobule nucleus', 'Pronormoblast', 'Prorubricyte', 'Polychromatic erythroblast', 'Metarubricyte', 'Prolymphocyte', 'Mature lymphocyte', 'Hetertypic lymphocyte', 'Primitive monocyte', 'Promonocyte', 'Mature monocyte', 'Plasmablast', 'infantile plasmocyte', 'Matrue plasmocyte', 'Bistiocyte', 'Juvenile cell', 'Granulocyte megakaryocyte', 'Naked megakaryocyte']

## 4.写在最后
- 请注意：以上代码仅供学习参考之用，请勿用于商业用途，如有侵权请联系删除，谢谢！