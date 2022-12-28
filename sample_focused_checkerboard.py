import cv2
import matplotlib.pyplot as plt
import shutil
def get_local_maximas(data, width=50):
    i_list = [int(x[0]) for x in data]
    fm_list = [x[1] for x in data]
    max_i = max(i_list)

    local_maximas = []
    for i in range(max_i//width+1):
        sublist = [d for d in data if (d[0] >= i*width and d[0] < (i+1)*width)]
        if len(sublist) > 0:
            local_maximas.append(max(sublist, key=lambda x:x[1]))
    return local_maximas

if __name__ == '__main__':
    with open('blur_metric.txt','r') as f:
        lines = f.readlines()
        lines = [line.split(' ') for line in lines]
    fmL_list = []
    fmR_list = []
    for line in lines:
        i, fmL, fmR = list(map(float, line))

        fmL_list.append((i, fmL))
        fmR_list.append((i, fmR))
    # xs = [i for i, _ in fmL_list]
    # ys = [fmL for _, fmL in fmL_list]
    # plt.plot(xs, ys)
    # plt.show()
    # exit()
    local_maximas = get_local_maximas(fmL_list, 30)
    local_maximas = [local_maxima for local_maxima in local_maximas if local_maxima[1] > 200]
    print(len(local_maximas))
    min_i, min_blur = min(local_maximas, key=lambda x:x[1])
    print(min_i,min_blur)
    for local_maxima in local_maximas:
        i, blur = local_maxima
        src = 'underwater_calibration/left/{:05d}.png'.format(int(i))
        dst = src.replace('underwater_calibration', 'underwater_calibration_focused')
        shutil.copy(src, dst)
        src = 'underwater_calibration/right/{:05d}.png'.format(int(i))
        dst = src.replace('underwater_calibration', 'underwater_calibration_focused')
        shutil.copy(src, dst)
    # img = cv2.imread('underwater_calibration/left/{:05d}.png'.format(int(min_i)))
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # cv2.imshow('most blurry', img)
    # cv2.waitKey()