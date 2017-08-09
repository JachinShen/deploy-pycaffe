#coding=utf-8
import caffe
import cv2
import time
model = './lenet_deploy.prototxt'#读取模型
weights = './lenet_iter_10000.caffemodel'#读取参数

caffe.set_mode_cpu;#设置成cpu模式，可改成gpu

net = caffe.Net(model, weights, caffe.TEST)#加载网络

start = time.time()#测试时间用
img_raw = cv2.imread('test3.PNG',0)#读图
_,img_raw = cv2.threshold(img_raw, 128, 255, cv2.THRESH_BINARY_INV)#二值化
img_raw = cv2.normalize(img_raw.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)#归一化
results = [[0 for col in range(10)] for row in range(9)]#中间结果矩阵
result = [0,0,0,0,0,0,0,0,0]#最终结果矩阵
for i in range(0,3):
        for j in range(0,3):
            img = img_raw[100+225*i:325+225*i,318*j:318+318*j]#截取测试图中九宫格部分
            img = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_LINEAR )#压缩为28*28大小
            img = img.reshape((1,1,28,28))#转换为1*1*28*28维向量，便于数据输入
            net.blobs['data'].data[...] = img     #将图片载入到blob中
            out = net.forward()#前传
            results[3*i+j] = net.blobs['loss'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值
for i in range(1,10):
    tmp = [0,0,0,0,0,0,0,0,0]
    for j in range(0,9):
        tmp[j] = results[j][i]
    print(tmp)
    result[tmp.index(max(tmp))] = i#取出每个数字最大概率所在的九宫格位置
print(result)
end = time.time()
print(end- start)#显示计算时间

