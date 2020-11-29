import numpy as np
import time
import icp
from rigid_registration import rigid_registration

# Constants
N = 55225                                   # number of random points in the dataset
num_tests = 1                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = 2                            # max translation of the test set
rotation = .5                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def test_best_fit():

    # Generate a random dataset
    A = np.random.rand(N, dim)
    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp(dataList):

    # Generate a random dataset
    A = np.mat(dataList);

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        #B += np.random.randn(N, dim) * noise_sigma

        # Shuffle to disrupt correspondence
        #np.random.shuffle(B)

        # Run ICP
        start = time.time()
        T, distances, iterations, mean_error = icp.icp(B, A, tolerance=0.0001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T
        print("mean error is %f"%mean_error);
        assert np.mean(distances) < 3*noise_sigma                   # mean error should be small
        #assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
        #assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses
        
        print("iter: %d"%i);
    print(T);
    print("mean error is %f"%mean_error);
    print('icp time: {:.3}'.format(total_time/num_tests))

    return C,B;

def loadLowData():
    dataList=[];
    with open("hand-low-tri.ply","r") as f:
        for i in range(10):
            f.readline();
        for i in range(55225):
            line =f.readline();
            ptList=line.strip().split(" ");
            dataList.append([float(ptList[0]),float(ptList[1]),float(ptList[2])]);
    return dataList;

def loadHighData():
    dataList=[];
    with open("hand-high-tri.ply","r") as f:
        for i in range(10):
            f.readline();
        for i in range(96942):
            line =f.readline();
            ptList=line.strip().split(" ");
            dataList.append([float(ptList[0]),float(ptList[1]),float(ptList[2])]);
    return dataList;

def loadHighSampleData():
    dataList=[];
    with open("new-high.ply","r") as f:
        for i in range(10):
            f.readline();
        for i in range(96942):
            line =f.readline();
            ptList=line.strip().split(" ");
            dataList.append([float(ptList[0]),float(ptList[1]),float(ptList[2])]);
    return dataList;

def saveData(dataMat,fileName):
    with open(fileName,"w") as fw:
        with open("hand-low-tri.ply","r") as f:
            for i in range(10):
                fileInfo=f.readline();
                fw.write(fileInfo);
            for i in range(dataMat.shape[0]):
                ptline="%f %f %f\n"%(dataMat[i,0],dataMat[i,1],dataMat[i,2]);
                fw.write(ptline);
            for j in range(55225):
                f.readline();
            for i in range(109893):
                faceData=f.readline();
                fw.write(faceData);
            f.close();
        fw.close();

def savehighData(dataMat,fileName):
    with open(fileName,"w") as fw:
        with open("hand-high-tri.ply","r") as f:
            for i in range(10):
                fileInfo=f.readline();
                fw.write(fileInfo);
            for i in range(dataMat.shape[0]):
                ptline="%f %f %f\n"%(dataMat[i,0],dataMat[i,1],dataMat[i,2]);
                fw.write(ptline);
            for j in range(96942):
                f.readline();
            for i in range(170772):
                faceData=f.readline();
                fw.write(faceData);
            f.close();
        fw.close();
def pcd():
    lowData=loadLowData();
    highData=loadHighData();
    # get number of dimensions
    A=np.mat(lowData);
    B=np.mat(highData);
    n,m = A.shape

    # make points homogeneous, copy them to maintain the originals
#      src = np.ones((A.shape[0],m))
#     dst = np.ones((B.shape[0]/2,m))
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    src = A[0:A.shape[0]:16,:];
    dst = B[0:B.shape[0]:32,:]
    src=src-centroid_A;
    dst=dst-centroid_B;
    for i in range(1):
#         sampleIndex=np.random.randint(0,high=n,size=500);
#         samples=src[sampleIndex[:],:];
        #distances, indices = icp.nearest_neighbor(samples, dst);
        a=rigid_registration(src, dst);
        ty,r,t,s=a.register();
#         src = s*np.dot(src,r)+t.T;
      
    print(r);
    print(t);
    print(s);
    src = s*np.dot(B,r)+t.T;
    
    savehighData(src,"high-dst.ply");
    newSrc=np.ones((m+1,src.shape[0]))
    newSrc[:m,:]=icp.sample(src, B.shape[0]);
    T, distances, iterations, mean_error=icp.icp(B, newSrc);
    print(t);
    C = np.dot(T, src.T)
    saveData(C,"new-low.ply");
#     savehighData(dst,"high-dst.ply")
    print("done");
    
def myicp():
    lowdata=loadLowData();
    result1,result2=test_icp(lowdata);
    saveData(result1, "python-icp1.ply");
    saveData(result2, "python-icptransform.ply");
    
if __name__ == "__main__":
    pcd();
    print("done")