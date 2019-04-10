
# 9.6.1 Support Vector Classifier


# We now use the svm() function to fit the support vector classifier for a given value of 
# the cost parameter. Here we demonstrate the use of this function on a two-dimensional 
# example so that we can plot the resulting decision boundary. We begin by generating the 
# observations, which belong to two classes, and checking whether the classes are linearly 
# separable.

set.seed(1)
x=matrix(rnorm (20*2), ncol=2)
y=c(rep(-1,10), rep(1,10))
x[y==1,]=x[y==1,] + 1
plot(x, col=(3-y))


# They are not. Next, we fit the support vector classifier. Note that in order for the 
# svm() function to perform classification (as opposed to SVM-based regression), we must 
# encode the response as a factor variable. We now create a data frame with the response 
# coded as a factor.

dat=data.frame(x=x, y=as.factor(y))
library(e1071)
svmfit=svm(y~., data=dat , kernel="linear",cost=10,scale=FALSE)


# We can now plot the support vector classifier obtained:

plot(svmfit, dat)


# The support vectors are plotted as crosses and the remaining observations are plotted as 
# circles; we see here that there are seven support vectors. We can determine their 
# identities as follows:

svmfit$index

summary(svmfit)


# What if we instead used a smaller value of the cost parameter?

svmfit=svm(y~., data=dat, kernel ="linear", cost=0.1,scale=FALSE)
plot(svmfit, dat)
svmfit$index


# The following command indicates that we want to compare SVMs with a linear kernel, using 
# a range of values of the cost parameter.

set.seed(1)
tune.out=tune(svm,y~.,data=dat,kernel ="linear",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))


summary(tune.out)


# The tune() function stores the best model obtained, which can be accessed as follows:

bestmod=tune.out$best.model
summary(bestmod)


# The predict() function can be used to predict the class label on a set of test 
# observations, at any given value of the cost parameter. We begin by generating a test 
# data set.

xtest=matrix(rnorm (20*2), ncol=2)
ytest=sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1,]=xtest[ytest==1,] + 1
testdat=data.frame(x=xtest, y=as.factor(ytest))

# Now we predict the class labels of these test observations. Here we use the best model 
# obtained through cross-validation in order to make predictions.

ypred=predict(bestmod,testdat)
table(predict=ypred,truth=testdat$y)


# Thus, with this value of cost, 19 of the test observations are correctly classified. What 
# if we had instead used cost=0.01?

svmfit=svm(y~.,data=dat,kernel ="linear",cost =1,scale=FALSE)
ypred=predict(svmfit,testdat)
table(predict=ypred,truth=testdat$y)


# In this case one additional observation is misclassified. Now consider a situation in 
# which the two classes are linearly separable. Then we can find a separating hyperplane 
# using the svm() function. We first further separate the two classes in our simulated 
# data so that they are linearly separable:

x[y==1,]=x[y==1,]+0.5
plot(x, col=(y+5)/2, pch =19)


# Now the observations are just barely linearly separable. We fit the support vector 
# classifier and plot the resulting hyperplane, using a very large value of cost so that 
# no observations are misclassified.

dat=data.frame(x=x,y=as.factor(y))
svmfit=svm(y~.,data=dat,kernel="linear",cost=1)
summary(svmfit)
plot(svmfit,dat)


# It seems likely that this model will perform poorly on test data. We now try a smaller 
# value of cost:

svmfit=svm(y~.,data=dat,kernel="linear", cost=1)
summary(svmfit)
plot(svmfit ,dat)



# 9.6.2 Support Vector Machine

#  We first generate some data with a non-linear class boundary, as follows:

set.seed(1)
x=matrix(rnorm (200*2),ncol=2)
x[1:100,]=x[1:100,]+2
x[101:150 ,]=x[101:150,]-2
y=c(rep(1,150),rep(2,50))
dat=data.frame(x=x,y=as.factor(y))

plot(x,col=y)

# The data is randomly split into training and testing groups. We then fit the training 
# data using the svm() function with a radial kernel and γ = 1:

train=sample(200,100)
svmfit=svm(y~.,data=dat[train,],kernel ="radial",gamma=1,cost=1)
plot(svmfit,dat[train,])

summary(svmfit)

svmfit=svm(y~.,data=dat[train,],kernel="radial",gamma=1,cost=1e5)
plot(svmfit,dat[train,])


# We can perform cross-validation using tune() to select the best choice of γ and cost for
# an SVM with a radial kernel:

set.seed(1)
tune.out=tune(svm,y~.,data=dat[train,],kernel ="radial",
              ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
summary(tune.out)


# Therefore, the best choice of parameters involves cost=1 and gamma=2. We can view the 
# test set predictions for this model by applying the predict() function to the data. 
# Notice that to do this we subset the dataframe dat using -train as an index set.

table(true=dat[-train,"y"],pred=predict(tune.out$best.model,newdata=dat[-train,]))

(74+16)/(74+16+7+3)


# 9.6.3 ROC Curves

library(ROCR)
rocplot =function (pred , truth , ...){
  predob = prediction (pred , truth)
  perf = performance (predob , "tpr", "fpr")
  plot(perf ,...)}

# In order to obtain the fitted values for a given SVM model fit, we use 
# decision.values=TRUE when fitting svm(). Then the predict() function will output the 
# fitted values.

svmfit.opt=svm(y~.,data=dat[train,],kernel="radial",gamma=2,cost=1,decision.values=T)
fitted=attributes(predict(svmfit.opt,dat[train,],decision.values=TRUE))$decision.values


# Now we can produce the ROC plot.
par(mfrow=c(1,2))
rocplot(fitted,dat[train,"y"],main="Training Data")

# SVM appears to be producing accurate predictions. By increasing γ we can produce a more 
# flexible fit and generate further improvements in accuracy.

svmfit.flex=svm(y~.,data=dat[train,],kernel ="radial",gamma=50,cost=1,decision.values=T)
fitted=attributes(predict(svmfit.flex,dat[train,],decision.values=T))$decision.values
rocplot(fitted,dat[train,"y"],add=T,col="red ")


# When we compute the ROC curves on the test data, the model with γ = 2 appears to provide
# the most accurate results.

fitted=attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train ,"y"],main="Test Data")
fitted=attributes(predict(svmfit.flex,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],add=T,col="red")


# 9.6.4 SVM with Multiple Classes

# If the response is a factor containing more than two levels, then the svm() function 
# will perform multi-class classification using the one-versus-one approach. We explore 
# that setting here by generating a third class of observations.

set.seed(1)
x=rbind(x,matrix(rnorm (50*2) , ncol=2))
y=c(y,rep(0,50))
x[y==0,2]=x[y==0 ,2]+2
dat=data.frame(x=x, y=as.factor(y))
par(mfrow=c(1,1))
plot(x,col=(y+1))


# We now fit an SVM to the data:

svmfit=svm(y~.,data=dat,kernel ="radial",cost=10,gamma=1)
plot(svmfit,dat)



# 9.6.5 Application to Gene Expression Data

library(ISLR)
names(Khan)
dim(Khan$xtrain )
dim(Khan$xtest )
length(Khan$ytrain )
length(Khan$ytest )


table(Khan$ytrain)
table(Khan$ytest)


# We will use a support vector approach to predict cancer subtype using gene expression 
# measurements. In this data set, there are a very large number of features relative to 
# the number of observations. This suggests that we should use a linear kernel, because 
# the additional flexibility that will result from using a polynomial or radial kernel is 
# unnecessary.

dat=data.frame(x=Khan$xtrain,y=as.factor(Khan$ytrain))
out=svm(y~.,data=dat, kernel ="linear",cost=10)
summary(out)

table(out$fitted,dat$y)

# test observations

dat.te=data.frame(x=Khan$xtest,y=as.factor(Khan$ytest))
pred.te=predict(out,newdata=dat.te)
table(pred.te,dat.te$y)

# We see that using cost=10 yields two test set errors on this data.












