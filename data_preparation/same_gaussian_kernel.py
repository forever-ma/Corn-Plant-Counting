import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import pyplot as plt
import PIL.Image as Image
from matplotlib import cm as CM

def generate_density_map_with_fixed_kernel(img,points,kernel_size=15,sigma=4.0):
    '''
    img: input image.
    points: annotated corn-plant's position like [row,col]
    kernel_size: the fixed size of gaussian kernel, must be odd number.
    sigma: the sigma of gaussian kernel.
    return:d_map: density-map we want
    '''
    def guassian_kernel(size,sigma):
        rows=size[0] 
        cols=size[1]
        mean_x=int((rows-1)/2)
        mean_y=int((cols-1)/2)
        f=np.zeros(size)
        for x in range(0,rows):
            for y in range(0,cols):
                mean_x2=(x-mean_x)*(x-mean_x)
                mean_y2=(y-mean_y)*(y-mean_y)
                f[x,y]=(1.0/(2.0*np.pi*sigma*sigma))*np.exp((mean_x2+mean_y2)/(-2.0*sigma*sigma))
        return f

    [rows,cols]=[img.shape[0],img.shape[1]]
    d_map=np.zeros([rows,cols])
    f=guassian_kernel([kernel_size,kernel_size],sigma)
    normed_f=(1.0/f.sum())*f # normalization

    if len(points)==0:
        return d_map
    else:
        for p in points:
            r,c=int(p[0]),int(p[1])
            if r>=rows or c>=cols:
                continue
            for x in range(0,f.shape[0]):
                for y in range(0,f.shape[1]):
                    if x+(r-int((f.shape[0]-1)/2))<0 or x+(r-int((f.shape[0]-1)/2))>rows-1 \
                    or y+(c-int((f.shape[1]-1)/2))<0 or y+(c-int((f.shape[1]-1)/2))>cols-1:
                        continue
                    else:
                        d_map[x+(r-int((f.shape[0]-1)/2)),y+(c-int((f.shape[1]-1)/2))]+=normed_f[x,y]
                        #print(d_map[x+((r+1)-int((f.shape[0]-1)/2)),y+((c+1)-int((f.shape[1]-1)/2))])
    return d_map

'''
# test code
if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    img_path="/home/user/corn/data/test/images/IMG_67.jpg"
    img=plt.imread(img_path)
    plt.imshow(img)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth'))
    pts = mat["file_data"]
    points=[]
    for p in pts:
        points.append([p[1],p[0]]) #convert (col,row) to (row,col)
    density_map=generate_density_map_with_fixed_kernel(img,points)
    plt.figure()
    from matplotlib import cm as CM
    plt.imshow(density_map,cmap=CM.jet)
    plt.show()
    np.save(img_path.replace('.jpg', '.npy').replace('images', 'ground_truth'), density_map)
    print(len(points))  # ground truth count
    print(density_map.sum())  # density_map count
    gt_file = np.load(img_path.replace('.jpg', '.npy').replace('images', 'ground_truth'))
    plt.imshow(gt_file, cmap=CM.jet)
'''


# test code
if __name__ == "__main__":
    root = '/home/user/corn/data'
    train = os.path.join(root, 'train', 'images')
    test = os.path.join(root, 'test', 'images')
    path_sets = [train, test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth'))
        img = plt.imread(img_path) 
        pts = mat["file_data"]
        points = []
        for p in pts:
            points.append([p[1], p[0]])  # convert (col,row) to (row,col)
        density_map = np.zeros((img.shape[0], img.shape[1]))
        density_map = generate_density_map_with_fixed_kernel(img, points)
        plt.imshow(density_map, cmap=CM.jet)
        np.save(img_path.replace('.jpg', '.npy').replace('images', 'ground_truth'), density_map)

'''    
    #see a sample 
    plt.imshow(Image.open(img_paths[0]))
    gt_file = np.load(img_paths[0].replace('.jpg','.npy').replace('images','ground_truth'))
    plt.imshow(gt_file,cmap=CM.jet)
    print(np.sum(gt_file))# don't mind this slight variation
'''