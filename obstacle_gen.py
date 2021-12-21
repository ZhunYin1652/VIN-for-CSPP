import numpy as np
import matplotlib.pyplot as plt


class obstacles:
    """A class for generating obstacles in a domain"""
    '''
    一个类，在一个区域内产生障碍物
    '''
    def __init__(self,
                 domsize=None, #这个区域的尺寸
                 mask=None, #这个mask和目标坐标goal有关吗？
                 size_max=None,
                 dom=None,
                 obs_types=None, #障碍物类型
                 num_types=None): #类型的数量
        #默认值为什么不在上面的参数列表里面直接给出呢？
        self.domsize = domsize or [] #如果外部提供了这个参数，就用外部参数，否则self.domsize属性为一个空列表，很有意思的写法
        self.mask = mask or [] #如果外部提供了这个参数，就用外部参数，否则self.mask属性为一个空列表。
        self.dom = dom or np.zeros(self.domsize) #如果外部提供了这个参数，就用外部参数，否则self.dom属性就是一个和self.domsize同形状的全零张量
        self.obs_types = obs_types or ["circ", "rect"] #如果外部提供了这个参数，就用外部参数，否则self.obs_types属性默认为["circ", "rect"]，即圆形或者矩形。
        self.num_types = num_types or len(self.obs_types) #如果外部提供了这个参数，就用外部参数，否则self.num_types属性默认为self.obs_types的长度。
        self.size_max = size_max or np.max(self.domsize) / 4 #如果外部提供了这个参数，就用外部参数，否则self.size_max属性为np.max(self.domsize) / 4，为什么是这么一个值？而且如果self.domsize为空列表，怎么办？

    def check_mask(self, dom=None): #这个类在检查mask
        # Ensure goal is in free space
        # 确保目标在自由空间
        # dom的默认值是None
        # ‘==’返回每个位置上的元素是否相等，.all()对这一结果做与运算，.any()对这一结果做或运算
        # 好像就是在判零
        if dom is not None: #如果区域不是不存在的，就用该方法传入的dom
            return np.any(dom[self.mask[0], self.mask[1]]) #难道dom里面的元素是数组吗？
        else: #如果区域是不存在的，就用该类的属性self.dom
            return np.any(self.dom[self.mask[0], self.mask[1]]) #根据上面的情况来看的话，self.dom有可能是np数组

    def insert_rect(self, x, y, height, width): #插入矩形；输入从哪一行哪一列开始，大小为多少行多少列
        # Insert a rectangular obstacle into map
        '''
        在地图中插入一个矩形障碍物
        '''
        im_try = np.copy(self.dom) #为什么要取消关联性？
        im_try[x:x + height, y:y + width] = 1 #把这些特定行和特定列赋值为1
        return im_try

    def add_rand_obs(self, obj_type): #插入随机障碍物；输入障碍物类型
        # Add random (valid) obstacle to map
        '''
        在地图中添加随机（有效）障碍物
        '''
        if obj_type == "circ": #如果障碍物类型是圆形
            print("circ is not yet implemented... sorry") #打印信息，不可用
        elif obj_type == "rect": #如果障碍物类型是矩形
            rand_height = int(np.ceil(np.random.rand() * self.size_max)) #任意高 #本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1，括号里面可以写形状，比如np.random.rand(2)，np.random.rand(3,2)
            rand_width = int(np.ceil(np.random.rand() * self.size_max)) #任意宽
            randx = int(np.ceil(np.random.rand() * (self.domsize[0] - 1)))#int(np.ceil(np.random.rand() * (self.domsize[1] - 1))) #随机行号
            randy = int(np.ceil(np.random.rand() * (self.domsize[1] - 1))) #随机列号
            im_try = self.insert_rect(randx, randy, rand_height, rand_width) #调用insert_rect方法，修改矩阵中的值，从0到1
        if self.check_mask(im_try): #调用check_mask方法，好像是在比较什么 #新增障碍物后，目标是否在障碍物中
            return False
        else:
            self.dom = im_try #更新self.dom属性
            return True

    def add_n_rand_obs(self, n): #插入n个随机障碍物；输入数量n
        # Add random (valid) obstacles to map
        count = 0 #计数
        for i in range(n): #n次调用add_rand_obs方法，随机添加n个矩形障碍物
            obj_type = "rect"
            if self.add_rand_obs(obj_type):
                count += 1
        return count

    def add_border(self):
        # Make full outer border an obstacle
        '''
        为谁制造完全外边界？
        '''
        im_try = np.copy(self.dom)
        im_try[0:self.domsize[0], 0] = 1 #左边
        im_try[0, 0:self.domsize[1]] = 1 #上边
        im_try[0:self.domsize[0], self.domsize[1] - 1] = 1 #右边
        im_try[self.domsize[0] - 1, 0:self.domsize[1]] = 1 #下边
        if self.check_mask(im_try):
            return False
        else:
            self.dom = im_try
            return True

    def get_final(self):
        # Process obstacle map for domain
        '''
        为区域处理障碍物地图？
        '''
        im = np.copy(self.dom)
        im = np.max(im) - im #取余
        im = im / np.max(im) #归一化
        return im #这个im好像是np数组

    def show(self):
        # Utility function to view obstacle map
        # Utility
        # n. 实用；效用；公共设施；功用
        # adj. 实用的；通用的；有多种用途的
        plt.imshow(self.get_final(), cmap='Greys') #调用方法get_final，把它生成的图像显示出来
        plt.show()

    def _print(self): #打印一些信息
        # Utility function to view obstacle map
        #  information
        print("domsize: ", self.domsize)
        print("mask: ", self.mask)
        print("dom: ", self.dom)
        print("obs_types: ", self.obs_types)
        print("num_types: ", self.num_types)
        print("size_max: ", self.size_max)
