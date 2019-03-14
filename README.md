## wordembedding
自己的wordembedding玩耍，把原先做过的都上传上来

## 环境
* python 3.6
* jieba_fast分词或者THC分词，jieba比较慢
* wikiextractor 抽取中文（https://github.com/attardi/wikiextractor.git）

## play_pretriain_embedding
```
# 查找相似的单词
python pre_train_embedding.py --word 机器学习
python pre_train_embedding.py --word 百度
python pre_train_embedding.py --word 机器学习 --top_k  10
# 比较两个单词的相似度
python pre_train_embedding.py --word 微积分/概率论
python pre_train_embedding.py --word 微积分/物理
python pre_train_embedding.py --word 微积分/地理
# 词之间的推断
python pre_train_embedding.py --word 卷积+深度学习
python pre_train_embedding.py --word 摩托车-单车
python pre_train_embedding.py --word 冬季-夏季+炎热
python pre_train_embedding.py --word 微积分/地理
# 增加新词
python pre_train_embedding.py --add_word '残差网络=0.3*卷积神经网络+0.3*残差+0.3*图像识别+0.1*人工智能'
python pre_train_embedding.py --add_word '残差网络=0.3*卷积神经网络+0.3*残差+0.4*图像识别' --add_vocabulary True
```

## wiki embedding
主要步骤
* 下载维基中文语料
* 抽取中文文本
* 转换繁体为简体
* 训练词嵌入
### 下载维基数据
wget -c https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

### 在wikiextractor目录下抽取维基文字，http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
cd wikiextractor && python WikiExtractor.py -b 500M -o ../ ../zhwiki-latest-pages-articles1.xml-p1p162886.bz2

### 使用opencc 将繁体转换为简体
opencc -i input_filename -o output_filename -c t2s.json

### 下载字典
wget -c http://horatio-jsy.oss-cn-beijing.aliyuncs.com/seg_dict.txt





