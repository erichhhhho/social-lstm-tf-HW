from PIL import Image

from pylab import *

# 读取图像到数组中
im = array(Image.open('/home/hesl/PycharmProjects/social-lstm-tf-HW/frame-001.jpg'))

# 显示图像
imshow(im)

print ('Please click 3 points')

#获取点击并将点击坐标保存在[x,y]列表中
x = ginput(3)

#输出保存的数据
print('you clicked:',x)

show()