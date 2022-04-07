等距投影鱼眼图像与全景图投影（ERP，CubeMap）转换，基于python、numpy、pytorch

鱼眼图像默认是等距投影，即主光轴夹角与图像坐标半径之间是线性变化，如果是真实图像，需要采用正、反两组多项式拟合进行转换。

核心是利用空间直角坐标系作为转换媒介，利用pytorch提供的grid_sample方法实现。
转换算子定义为类，一般流程为：
1. 类初始化，输入相关参数，生成目标投影每个位置在源投影的采样位置坐标（grid）数据类型np.array， float32；
2. 调用trans方法，将源图像与grid均转化为torch.tensor，并使用grid_sample完成投影变换，输出目标投影图。

fisheyeCubemap.py 进行鱼眼与Cubemap之间的相互转换，文件定义了两个类负责转换。默认鱼眼镜头朝向（主光轴方向）正对Cubemap正面（front face）；根据真实情况传入旋转矩阵可以改变相应的方向；

fisheyeERP.py 进行鱼眼与ERP之间的相互转换。默认鱼眼镜头朝向（主光轴方向）正对ERP正面，主光轴位于ERP投影正中心；根据实际情况传入旋转矩阵改变方向。