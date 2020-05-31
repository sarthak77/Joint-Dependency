import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def load_data():
    """
    Return data in np format
    """

    from scipy.io import loadmat
    f1='drx5day_zone.csv.mat'
    f2='dwsdi_zone.csv.mat'
    return(loadmat(f1)['tn'],loadmat(f2)['tn'])



def plot_hm(X,T):
    """
    Plot heat map
    """

    import seaborn as sns
    
    X=X.transpose()
    X=np.flipud(X)
    
    #creat mask
    # M=np.array([i>0 for i in X])
    M=np.invert(np.array(X,dtype=bool))
    
    # create heat map
    ax=sns.heatmap(X,xticklabels=False,yticklabels=False,mask=M)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,vmin=0,vmax=.1,mask=M)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,vmin=0,vmax=.5,mask=M)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,vmin=-.5,vmax=0,mask=M)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,vmin=0,vmax=1)
    
    plt.title(T)
    plt.show()



def distancecc(X,Y):
    """
    Compute distance CC
    """
    
    from sklearn.metrics import pairwise_distances

    X=X.reshape((-1,1))
    X=pairwise_distances(X)
    Y=Y.reshape((-1,1))
    Y=pairwise_distances(Y)
    
    # double centre the data
    def centred(x):
        return x-x.mean(axis=0)[None,:]-x.mean(axis=1)[:,None]+x.mean()

    # calculate distance covariance
    def dist_cov(x,y):
        cx=centred(x)
        cy=centred(y)
        return np.mean(np.multiply(cx,cy))

    cov=dist_cov(X,Y)
    sd=np.sqrt(dist_cov(X,X)*dist_cov(Y,Y))
    return [cov/sd,0]



def va1():
    """
    Example for distance CC (value addition for 1st part)
    """

    # generate points on circle
    X=[]
    Y=[]
    pi=np.pi
    n=100
    r=1
    for i in range(n):
        X.append(np.cos(2*pi/n*i)*r)
        Y.append(np.sin(2*pi/n*i)*r)

    X=np.array(X)
    Y=np.array(Y)
    
    s1="Pearson CC: "+str(stats.pearsonr(X,Y)[0])
    s2="Distance CC: "+str(distancecc(X,Y)[0])

    plt.scatter(X,Y)
    plt.text(0,0,s1+"\n"+s2)
    plt.title("Sample Dataset")
    plt.show()



def part1():
    """
    Correlation coefficients
    """
    
    Func=[stats.pearsonr,stats.spearmanr,stats.kendalltau,distancecc]
    Title=["Pearson CC","Spearman CC","Kendall CC","Distance CC"]

    for (f,t) in zip(Func,Title):
        temp=np.zeros(121*121).reshape((121,121))
        for i in range(121):
            for j in range(121):
                a=f(P[i][j],T[i][j])
                if np.isnan(a[0]):
                    temp[i][j]=0
                else:
                    temp[i][j]=a[0]
        plot_hm(temp,t)
    va1()



def plot_PDF(X):
    """
    Plot PDF/CDF
    """

    import seaborn as sns

    # set labels
    Lx=["T1","T2","T3","T4","T5"]
    Ly=["P1","P2","P3","P4","P5"]
    ax=sns.heatmap(X,annot=True,xticklabels=Lx,yticklabels=Ly,fmt="f",cmap="YlGnBu",linewidth=.3,cbar=False)
    
    # extend boundary from top and bottom
    b,t=ax.get_ylim()
    ax.set_ylim(b+.5,t-.5)
    
    # ax.set_yticklabels(ax.get_yticklabels(),rotation=45)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    ax.xaxis.tick_top()#x axis on top
    ax.tick_params(length=0)#remove ticks
    plt.yticks(rotation=0)#remove rotation

    # ax.set_xlabel("Joint PDF",fontsize='large')
    ax.set_xlabel("Joint CDF",fontsize='large')
    plt.show()



def get_mean():
    """
    Return mean P and T
    """

    AP=[]
    AT=[]
    for i in range(P.shape[2]):
        t1=np.sum(P[:,:,i])
        t2=np.count_nonzero(P[:,:,i])
        AP.append(t1/t2)
        t1=np.sum(T[:,:,i])
        t2=np.count_nonzero(T[:,:,i])
        AT.append(t1/t2)
    return AP,AT



def va2():
    """
    Mutual Information (valua addition for 2nd part)
    """

    X,Y=get_mean()
    px,t1=np.histogram(X,bins=5)
    py,t1=np.histogram(Y,bins=5)
    pxy,t1,t2=np.histogram2d(X,Y,bins=5)
    
    # find marginal and joint probabilities
    px=np.divide(px,P.shape[2])
    py=np.divide(py,P.shape[2])
    pxy=np.divide(pxy,P.shape[2])
    
    pxy=pxy.reshape((25,1))

    # compute MI
    mi=0
    for i in px:
        mi-=i*np.log2(i)
    for i in py:
        mi-=i*np.log2(i)
    for i in pxy:
        if i:
            mi+=i*np.log(i)
    print(mi)



def copula_based(X,Y):
    """
    Calculate joint PDF/CDF using copula
    """

    import pandas as pd
    from copulas.multivariate import GaussianMultivariate
    
    # fit gaussian copula
    data=pd.DataFrame(list(zip(X,Y)),columns=['P','T'])
    dist=GaussianMultivariate()
    dist.fit(data)

    sampled=dist.sample(1)
    sampled.at[0,'P']=np.mean(X)
    sampled.at[0,'T']=np.mean(Y)
    
    # find pdf/cdf at mean value
    pdf=dist.pdf(sampled)
    cdf=dist.cumulative_distribution(sampled)
    return [pdf,cdf]



def discrete():
    """
    Calculate joint PDF/CDF assuming discrete data
    """
    
    AP,AT=get_mean()
    H,X,Y=np.histogram2d(AP,AT,bins=5)
    H=np.divide(H,P.shape[2])

    print(H)
    print(np.sum(H))
    plot_PDF(H)

    # compute CDF
    for i in range(5):
        for j in range(1,5,1):
            H[i][j]+=H[i][j-1]
    for i in range(1,5,1):
        for j in range(5):
            H[i][j]+=H[i-1][j]

    print(H)
    print(np.sum(H))
    plot_PDF(H)



def part2():
    """
    Joint dependency
    """

    discrete()

    # using copula
    temp1=np.zeros(121*121).reshape((121,121))
    temp2=np.zeros(121*121).reshape((121,121))
    for i in range(121):
        for j in range(121):
            print(i,j)
            temp1[i][j],temp2[i][j]=copula_based(P[i][j],T[i][j])
    
    plot_hm(temp1,"Joint PDF")
    plot_hm(temp2,"Joint CDF")

    va2()



def part3():
    """
    Calculate return period
    """

    temp=np.zeros(121*121).reshape((121,121))
    for i in range(121):
        for j in range(121):
            print(i,j)
            p,c=copula_based(P[i][j],T[i][j])
            if 1-c != 0:
                temp[i][j]=1/(1-c)
    plot_hm(temp,"Return Period")



if __name__ == "__main__":
    
    P,T=load_data()

    # for temporal analysis
    # P=P[:,:,0:10]
    # T=T[:,:,0:10]
    # print(P.shape)
    
    part1()
    part2()
    part3()