import numpy as np

def get_downsample(data):
    data_shape = data.shape
    #chuan hoa anh
    data = data * (1/256)
    sampleCount = data_shape[0]
    res = np.empty((sampleCount, 14, 14))
    #Gan gia tri cac phan tu trong mang res deu bang 0.
    res = res*0
    for sampleIndex in range(sampleCount):
        for i in range(data_shape[1]):
            for j in range(data_shape[2]):
                #Tinh trung binh cong 4 o gop thanh 1 o, chuyen ve doan [0,1].
                res[sampleIndex, i//2, j//2] += (data[sampleIndex, i, j])/4
    return res