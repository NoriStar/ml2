import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Extracting the data using read_excel, given the file could not be downloaded as a csv on my laptop
Data = pd.read_csv('data_problem2 (1).csv', header=None)
data=np.array(Data)
data=data.T
print(data)

#Extracting the values according to their classes then making them arrays
x0 = data[data[:,1] == 0, 0]
x1 = data[data[:, 1] == 1, 0]
x0=np.array(x0)
x1=np.array(x1)

#plt.figure(figsize=(10, 6))  # Set figure size

#plotting their histograms
plt.hist(x0, bins=20, alpha=0.7, label='Class 0 (C0)', color='blue')
plt.hist(x1, bins=20, alpha=0.7, label='Class 1 (C1)', color='orange')

#plt.xlabel('X values')
#plt.ylabel('Frequency (proportional to probability)')
#plt.title('Histogram of X values for Class 0 and Class 1')

# Adding a legend to distinguish between classes
plt.legend()

# Show the plot
#plt.show()

#splitting the data into training and test
split_index_0 = int(len(x0) * 0.8)
split_index_1 = int(len(x1) * 0.8)
x0_train = x0[:split_index_0]
x0_test = x0[split_index_0:]
x1_train = x1[:split_index_1]
x1_test = x1[split_index_1:]

#-----Starting by the training data
# Calculate mean and standard deviation for x0
mux1 = np.mean(x1_train)
sdx1 = np.std(x1_train)

#gauss function
def gaussian(x,mu,sigma):
    a = 1 / (sigma*np.sqrt(2*np.pi))
    b = np.exp(-0.5 *((x-mu)/sigma) ** 2)
    return a*b

#calling function and defining data to our x axis.
x_values = np.linspace(min(x1_train),max(x1_train))
gaussian = gaussian(x_values, mux1, sdx1)

#plt.figure(figsize=(10, 6))
#plt.plot(x_values, gaussian, label='Gaussian PDF for Class 1', color='red')
#plt.hist(x1_train, bins=20, alpha=0.3, label='Class 1 Train Samples', color='orange', density=True)  # Normalize histogram

#plt.xlabel('X values')
#plt.ylabel('Probability Density')
#plt.title('Gaussian Probability Density Function for Class 1')

n0 = len(x0_train)
al = 2
beta = (1 / (n0*al)) * np.sum(x0_train)

#Gamma function
def gamma(x,beta):
    coeff = 1 / (beta ** 2 * 1)
    pdf_values = coeff * (x ** (2 - 1)) * np.exp(-x / beta)
    return pdf_values

#xvalues of training data
xt = np.linspace(min(x0_train), max(x0_train))

#calling or using gamma in our data
gamma = gamma(xt,beta)

#Plot
plt.plot(xt,gamma, label='Gamma PDF for Class 0')

# Setting limits to zoom out further
plt.xlim(min(x0_train) - 10, max(x0_train) + 10)  # Adjust x limits since it does not fit on a single plot
plt.ylim(0, max(gamma))  # Adjust y limits for a bit of padding

# Adding labels and title
plt.xlabel('X values')
plt.ylabel('Probability Density')
plt.title('Gamma Probability Density Function for Class 0')
plt.legend()
plt.grid()
plt.show()

#____For our test
#Making functions to obtain parameter
def mu_hat(x):
    return np.mean(x)

def sigma_hat(x,mu):
    return np.sqrt(np.sum((x - mu)**2) / len(x))

mu=mu_hat(x1_test)
#calling function
sigma=sigma_hat(x1_test, mu)

# Applying gaussian test
def Gaussian_test(x,sigma1,mu1):
    c=1/(sigma1 * np.sqrt(2 * np.pi))
    d=np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    return c * d

Gauss_test = Gaussian_test(x1_test, sigma, mu)

#for Gamma:
#Making function for parameters to calculate
def beta_hat(x,n):
    return (1/(2*n))*np.sum(x)

# Call beta_hat with test data
beta_value = beta_hat(x0_test, len(x0_test))

# Function to calculate Gamma distribution
def Gamma(x,beta):
    v=1/(beta**2)
    return v*x* np.exp(-x/beta)

#calling functiom
x_values_gauss = np.linspace(min(x0_test), max(x0_test), len(x0_test))
gauss_test_values = Gaussian_test(x_values_gauss, sigma, mu)

x_values_gamma=np.linspace(min(x1_test), max(x1_test), len(x1_test))
gamma_test_values=Gamma(x_values_gamma, beta_value)

# Plotting both PDFs on the same plot
plt.figure(figsize=(10, 6))

# Plot Gaussian PDF for Class 0 (Test data)
plt.plot(x_values_gauss, gauss_test_values, label='Gaussian PDF for Class 0 (Test)', color='blue')

#Plot Gamma PDF for Class 1 (Test data)
plt.plot(x_values_gamma, gamma_test_values, label='Gamma PDF for Class 1 (Test)', color='orange')

#Adding labels, title, legend, and grid
plt.xlabel('X values')
plt.ylabel('Probability Density')
plt.title('PDFs for Class 0 (Gaussian) and Class 1 (Gamma) on Test Data')
plt.legend()
plt.grid()
plt.show()

#calculate priors
P_C0=len(x0_train)/(len(x0_train)+len(x1_train))
P_C1=len(x1_train)/(len(x0_train)+len(x1_train))

predict=[]

# Classify test data using Bayes' theorem
for x in np.concatenate([x0_test, x1_test]):
    # Calculate likelihoods
    l0=(1/(beta**2))*(x**(al-1))*np.exp(-x/beta) if x >= 0 else 0
    l1=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)
    #Calculate posterior probabilities
    p0=l0*P_C0
    p1=l1*P_C1
    #Classify based on the higher posterior
    a=0 if p0>p1 else 1
    predict.append(a)

#True labels of the  test data
true_labels = np.concatenate([np.zeros(len(x0_test)), np.ones(len(x1_test))])

accuracy = np.sum(predict==true_labels)/len(true_labels)
print(f"Test accuracy: {accuracy*100:.2f}%")