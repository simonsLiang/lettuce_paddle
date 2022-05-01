# lettuce_paddle


## 目录


- [1. 简介]()
- [2. 复现精度]()
- [3. 准备环境]()
- [4. 开始使用]()
- [5. LICENSE]()
- [6. 参考链接与文献]()


## 1. 简介

感谢百度飞桨提供的算力支持

## 2. 复现精度

可以从[BaiduYun](https://pan.baidu.com/s/1p8N9yoqkVypHZDUUsgFhAA?pwd=cexk)处下载wide_resnet52_2的预训练参数init.pdparams以及模型对齐过程,由于本项目模型不需要训练，故不提供训练日志
                                        
                                        
## 3. 准备环境与数据

### 3.1 准备环境

* 下载代码

```bash
git clone https://github.com/simonsLiang/orthoad_paddle
cd orthoad_paddle
```

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements

```bash
pip install -r requirements.txt
```


## 4. 开始使用



## 5. LICENSE

[Apache License 2.0](./LICENSE)


## 6. 参考链接与文献
[Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation](https://arxiv.org/pdf/2105.14737v1.pdf)
