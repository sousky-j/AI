import numpy as np 
def feature_normalization(data): # 10 points
    # parameter 
    feature_num = data.shape[1]
    data_point = data.shape[0]
    
    # you should get this parameter correctly
    normal_feature = np.zeros([data_point, feature_num])
    mu = np.zeros([feature_num])
    std = np.zeros([feature_num])

    mu=np.mean(data, axis=0)
    std=np.std(data, axis=0)

    for i in range(feature_num):
        normal_feature[:,i]=data[:,i]-np.min(data[:,i])/np.max(data[:,i])-np.min(data[:,i])
        normal_feature[:,i]=(data[:,i]-mu[i])/std[i]
    return normal_feature
        
def split_data(data, label, split_factor):
    return  data[:split_factor], data[split_factor:], label[:split_factor], label[split_factor:] #training_data, test_data, training_label, test_label

def get_normal_parameter(data, label, label_num): # 20 points
    # parameter
    feature_num = data.shape[1]
    
    # you should get this parameter correctly    
    mu = np.zeros([label_num,feature_num])
    sigma = np.zeros([label_num,feature_num])

    idx = [] #List comprehension for class maping
    for _ in range(label_num): #list generative of class
        idx.append([])
    
    for i in range(len(label)): # maping to 2d list
        idx[label[i]].append(i)

    for i in range(label_num): # use the built-in function in numpy
        for j in range(feature_num):
            mu[i][j] = np.mean(data[idx[i], j])
            sigma[i][j] = np.std(data[idx[i], j])

    return mu, sigma # both 3*4 matrix

def get_prior_probability(label, label_num): # 10 points
    # parameter
    data_point = label.shape[0]
    
    # you should get this parameter correctly
    prior = np.zeros([label_num])
    for i in range(data_point): #find the class number
        prior[label[i]]+=1
    prior/=data_point #calculate the probability

    return prior

def Gaussian_PDF(x, mu, sigma): # 10 points
    pdf = 0
    coef=1 / (np.sqrt(2*np.pi*sigma**2))
    func=np.exp(-(((x-mu)/sigma)**2)/2) #Gaussian_PDF equation
    pdf=coef*func
    return pdf

def Gaussian_Log_PDF(x, mu, sigma): # 10 points
    log_pdf = np.log1p(Gaussian_PDF(x, mu, sigma)) # log(P(x)+1)
    return log_pdf

def Gaussian_NB(mu, sigma, prior, data): # 40 points
    # parameter
    data_point = data.shape[0]
    label_num = mu.shape[0]
    
    # you should get this parameter correctly   
    likelihood = np.zeros([data_point, label_num])
    posterior = np.zeros([data_point, label_num])
    ## evidence can be omitted because it is a constant
    
    for j in range(label_num): #class loop
        for k in range(data.shape[1]): #pdf calculate loop (feature)
            likelihood[:, j]+=Gaussian_Log_PDF(data[:, k], mu[j][k], sigma[j][k])

    for i in range(label_num):
            posterior[:, i]=likelihood[:, i]+np.log1p(prior[i]) #likelihood & prior probability
    return posterior

def classifier(posterior):
    data_point = posterior.shape[0]
    prediction = np.zeros([data_point])
    
    prediction = np.argmax(posterior, axis=1) # predict use each of posterior probability
    return prediction

def accuracy(pred, gnd):
    data_point = len(gnd)
    
    hit_num = np.sum(pred == gnd) #Score the answer

    return (hit_num / data_point) * 100, hit_num
    ## total 100 point you can get