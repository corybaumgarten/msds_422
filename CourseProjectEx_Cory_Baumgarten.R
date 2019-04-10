
# PREDICT 422 Practical Machine Learning

# Course Project - Example R Script File

# OBJECTIVE: A charitable organization wishes to develop a machine learning
# model to improve the cost-effectiveness of their direct marketing campaigns
# to previous donors.

# 1) Develop a classification model using data from the most recent campaign that
# can effectively capture likely donors so that the expected net profit is maximized.

# 2) Develop a prediction model to predict donation amounts for donors - the data
# for this will consist of the records for donors only.

# load the data
charity <- read.csv("/Users/corybaumgarten/Documents/Northwestern/2018SU_MSDS_422-DL_SEC55/Course Project/charity.csv",header = TRUE) # load the "charity.csv" file
names(charity)
summary(charity)
str(charity)

# load packages
library(corrplot)
library(lessR)
library(MASS)
library(moments)
library(leaps)
library(car)
attach(charity)


# overview of raw data

# check for missing values

table(is.na(charity[charity$part=="train",]))
table(is.na(charity[charity$part=="valid",]))


# overview of all data
CountAll(charity)


# reg1,reg2,reg3,reg4
table(reg1)
table(reg2)
table(reg3)
table(reg4)

# reg prop
sum(reg1=="1")/nrow(charity)
sum(reg2=="1")/nrow(charity)
sum(reg3=="1")/nrow(charity)
sum(reg4=="1")/nrow(charity)

#reg5 = 1
nrow(charity)-sum(sum(reg1=="1")+sum(reg2=="1")+sum(reg3=="1")+sum(reg4=="1"))

#reg5 = 0
nrow(charity)-(nrow(charity)-sum(sum(reg1=="1")+sum(reg2=="1")+sum(reg3=="1")+sum(reg4=="1")))

(nrow(charity)-sum(sum(reg1=="1")+sum(reg2=="1")+sum(reg3=="1")+sum(reg4=="1")))/nrow(charity)

# reg5 prop
1-(sum(reg1=="1")/nrow(charity)+sum(reg2=="1")/nrow(charity)+sum(reg3=="1")/nrow(charity)+
     sum(reg4=="1")/nrow(charity))

# home
table(home)
Histogram(home,data = charity)
sum(home=="1")/nrow(charity)

# chld
table(chld)
Histogram(chld,data = charity)

# hinc
table(hinc)
Histogram(hinc,data = charity)

# genf
table(genf)
Histogram(genf,data=charity)

# wrat
table(wrat)
Histogram(wrat,data = charity)
SummaryStats(wrat,data=charity)


# avhv
par(mfrow=c(1,2))
hist(avhv, main = "Histogram of AVHV", col = "red") # already transformed
hist(log(avhv), main = "Histogram of log(AVHV)", col = "dodgerblue4")
skewness(avhv)
skewness(log(avhv)) # log transformation reduces skewness.

# incm
hist(incm, main = "Histogram of INCM", col = "red") # already transformed
hist(log(incm), main = "Histogram of log(INCM)", col = "dodgerblue4")
skewness(incm)
skewness(log(incm)) # log transformation reduces skewness.

# inca
hist(inca, main = "Histogram of INCA", col = "red") # already transformed
hist(log(inca), main = "Histogram of log(INCA)", col = "dodgerblue4")
skewness(inca)
skewness(log(inca)) # log transformation reduces skewness.

# plow
hist(plow, main = "Histogram of PLOW", col = "red") # already transformed
hist(sqrt(plow), main = "Histogram of sqrt(PLOW)", col = "dodgerblue4")
skewness(plow)
skewness(sqrt(plow)) # sqrt transformation reduces skewness.


# npro
par(mfrow=c(1,1))
hist(npro, main = "Histogram of NPRO", col = "red")
skewness(npro)
par(mfrow=c(1,2))


# tgif
hist(tgif, main = "Histogram of TGIF", col = "red")
hist(log(tgif), main = "Histogram of log(TGIF)", col = "dodgerblue4")
skewness(tgif)
skewness(log(tgif)) # log transformation reduces skewness.

# lgif
hist(lgif, main = "Histogram of LGIF", col = "red")
hist(log(lgif), main = "Histogram of log(LGIF)", col = "dodgerblue4")
skewness(lgif)
skewness(log(lgif)) # log transformation reduces skewness.

# rgif
hist(rgif, main = "Histogram of RGIF", col = "red")
hist(log(rgif), main = "Histogram of log(RGIF)", col = "dodgerblue4")
skewness(rgif)
skewness(log(rgif)) # log transformation reduces skewness.

# tdon
par(mfrow=c(1,1))
hist(tdon,main = "Histogram of TDON", col = "red")
skewness(tdon)
par(mfrow=c(1,2))

# tlag
hist(tlag, main = "Histogram of TLAG", col = "red") # already transformed
hist(log(tlag), main = "Histogram of log(TLAG)", col = "dodgerblue4")
skewness(tlag)
skewness(log(tlag)) # log transformation reduces skewness.

# agif
hist(agif, main = "Histogram of AGIF", col = "red") # already transformed
hist(log(agif), main = "Histogram of log(AGIF)", col = "dodgerblue4")
skewness(agif)
skewness(log(agif)) # log transformation reduces skewness.


par(mfrow=c(1,1))


# predictor transformations

charity.t <- charity
charity.t$avhv <- log(charity.t$avhv)
charity.t$incm <- log(charity.t$incm)
charity.t$inca <- log(charity.t$inca)
charity.t$plow <- sqrt(charity.t$plow)
charity.t$tgif <- log(charity.t$tgif)
charity.t$lgif <- log(charity.t$lgif)
charity.t$rgif <- log(charity.t$rgif)
charity.t$tlag <- log(charity.t$tlag)
charity.t$agif <- log(charity.t$agif)


# set up data for analysis

# create training data

data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,2:21]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995


# create validation data

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,2:21]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999


# create test data

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,2:21]

# standardize data

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)



# classification EDA

# correlation with donr
c1<- as.data.frame(sort(cor(x.train,c.train)[,1],decreasing=TRUE))
c2<- as.data.frame(sort(cor(x.valid,c.valid)[,1],decreasing=TRUE))
c3 <- cbind(c1,c2)
colnames(c3) <- c("training corr","validation corr")
c3 <- round(c3,4)
c3



##### CLASSIFICATION MODELING ######

# logistic regression

model.log1 <- glm(donr ~ reg1 + reg2 + home + chld + I(chld^2) + I(hinc^2) + 
                    genf + I(wrat^2) + I(avhv^2) + incm + poly(inca^3) + plow + npro + tgif + lgif +
                    poly(rgif^4) + I(tdon^2) + tlag + agif, data.train.std.c, family=binomial("logit"))

summary(model.log1)

post.valid.log1 <- predict(model.log1, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1,main="Logistic Model Profit Curve",xlab="Number of Mailings",ylab="Profit") # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit
# original: 1291.0 11642.5
# new: 1237 11808.5

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1, c.valid) # classification table

log1.class=chat.valid.log1
log1.acc <- mean(log1.class==c.valid)
log1.acc



# logistic regression GAM
library(gam)

model.gamlr1=gam(donr ~ reg1 + reg2 + home + s(chld,5) + I(hinc^2) + 
                   genf + wrat + s(wrat,5) + s(avhv,5) + s(incm,10) + s(inca,10) + poly(plow^5) + npro + tgif + lgif + 
                   s(tdon,10) + poly(tdon^3) + s(tlag,10) + agif, data.train.std.c, family=binomial("logit"))

summary(model.gamlr1)
post.valid.gamlr1 <- predict(model.gamlr1, data.valid.std.c, type="response") # n.valid post probs


# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.gamlr1 <- cumsum(14.5*c.valid[order(post.valid.gamlr1, decreasing=T)]-2)
plot(profit.gamlr1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.gamlr1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.gamlr1)) # report number of mailings and maximum profit


cutoff.gamlr1 <- sort(post.valid.gamlr1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.gamlr1 <- ifelse(post.valid.gamlr1>cutoff.gamlr1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.gamlr1, c.valid) # classification table

gamlr1.class=chat.valid.gamlr1
gamlr1.acc <- mean(gamlr1.class==c.valid)
gamlr1.acc



# linear discriminant analysis

library(MASS)

model.lda1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + I(chld^2) + hinc + I(hinc^2) + 
                    genf + I(wrat^2) + avhv + incm + inca + plow + npro + tgif + lgif +
                    rgif + tdon + I(tdon^2) + tlag + agif, data.train.std.c)
model.lda1
plot(model.lda1)
post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1)) # report number of mailings and maximum profit
# original: 1329.0 11624.5
# new: 1307 11770

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda1, c.valid) # classification table

# LDA1 prediction accuracy
lda1.class=chat.valid.lda1
lda1.acc <- mean(lda1.class==c.valid)
lda1.acc



# quadratic discriminant analysis

model.qda1=qda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                 avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
               data.train.std.c)

model.qda1

post.valid.qda1 <- predict(model.qda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.qda1 <- cumsum(14.5*c.valid[order(post.valid.qda1, decreasing=T)]-2)
plot(profit.qda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.qda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.qda1)) # report number of mailings and maximum profit


cutoff.qda1 <- sort(post.valid.qda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.qda1 <- ifelse(post.valid.qda1>cutoff.qda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.qda1, c.valid) # classification table

qda1.class=chat.valid.qda1
qda1.acc <- mean(qda1.class==c.valid)
qda1.acc


# k-nearest neighbors
library(class)

knn.trainlabels <- data.train.std.c[,21]

model.knn1=knn(data.train.std.c,data.valid.std.c,knn.trainlabels,k=1)
table(model.knn1,data.valid.std.c[,21])
model.knn1.acc <- (933+834)/2018

model.knn2=knn(data.train.std.c,data.valid.std.c,knn.trainlabels,k=2)
table(model.knn2,data.valid.std.c[,21])
model.knn2.acc <- (917+831)/2018

model.knn3=knn(data.train.std.c,data.valid.std.c,knn.trainlabels,k=3)
table(model.knn3,data.valid.std.c[,21])
model.knn3.acc <- (957+841)/2018

model.knn4=knn(data.train.std.c,data.valid.std.c,knn.trainlabels,k=4)
table(model.knn4,data.valid.std.c[,21])
model.knn4.acc <- (956+833)/2018

model.knn5=knn(data.train.std.c,data.valid.std.c,knn.trainlabels,k=5)
table(model.knn5,data.valid.std.c[,21])
model.knn5.acc <- (964+836)/2018

model.knn10=knn(data.train.std.c,data.valid.std.c,knn.trainlabels,k=10)
table(model.knn10,data.valid.std.c[,21])
model.knn10.acc <- (970+821)/2018

1-model.knn1.acc
1-model.knn2.acc
1-model.knn3.acc
1-model.knn4.acc
1-model.knn5.acc
1-model.knn10.acc


# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

# k=1
profit.knn1 <- cumsum(14.5*c.valid[order(model.knn1, decreasing=T)]-2)
plot(profit.knn1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.knn1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn1)) # report number of mailings and maximum profit
table(model.knn1, c.valid) # classification table

# k=2
profit.knn2 <- cumsum(14.5*c.valid[order(model.knn2, decreasing=T)]-2)
plot(profit.knn2) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.knn2) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn2)) # report number of mailings and maximum profit
table(model.knn2, c.valid) # classification table

# k=3
profit.knn3 <- cumsum(14.5*c.valid[order(model.knn3, decreasing=T)]-2)
plot(profit.knn3) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.knn3) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn3)) # report number of mailings and maximum profit
table(model.knn3, c.valid) # classification table

# k=4
profit.knn4 <- cumsum(14.5*c.valid[order(model.knn4, decreasing=T)]-2)
plot(profit.knn4) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.knn4) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn4)) # report number of mailings and maximum profit
table(model.knn4, c.valid) # classification table

# k=5
profit.knn5 <- cumsum(14.5*c.valid[order(model.knn5, decreasing=T)]-2)
plot(profit.knn5) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.knn5) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn5)) # report number of mailings and maximum profit
table(model.knn5, c.valid) # classification table

# k=10
profit.knn10 <- cumsum(14.5*c.valid[order(model.knn10, decreasing=T)]-2)
plot(profit.knn10) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.knn10) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.knn10)) # report number of mailings and maximum profit
table(model.knn10, c.valid) # classification table


# Support Vector Machines
library(e1071)

y=as.factor(data.train.std.c$donr)

model.svm=svm(y ~ reg1 + reg2 + reg3 + reg4 + home + chld + I(chld^2) + hinc + I(hinc^2) + 
                genf + I(wrat^2) + avhv + incm + inca + plow + npro + tgif + lgif +
                rgif + tdon + I(tdon^2) + tlag + agif, data.train.std.c,
              kernel ="polynomial", degree=3,cost=.1,scale=FALSE)
model.svm

post.valid.svm <- predict(model.svm, data.valid.std.c) # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.svm <- cumsum(14.5*c.valid[order(post.valid.svm, decreasing=T)]-2)
plot(profit.svm) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.svm) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.svm)) # report number of mailings and maximum profit

table(post.valid.svm, c.valid) # classification table

cutoff.svm <- sort(post.valid.svm, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid

svm.class=chat.valid.qda1
svm.acc <- mean(svm.class==c.valid)
svm.acc


# Results

# select GAMLR since it has maximum profit in the validation sample

post.test <- predict(model.gamlr1, data.test.std, type="response") # post probs for test data

# Oversampling adjustment for calculating number of mailings for test set

n.mail.valid <- which.max(profit.gamlr1)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)
# original
#    0    1 
# 1676  331

# gamlr
#    0    1 
# 1688  319

# based on this model we'll mail to the 319 highest posterior probabilities



##### PREDICTION MODELING ######

# Least squares regression

model.ls1 <- lm(damt ~ reg3 + reg4 + home + chld + hinc + 
                  genf + I(wrat^2) + incm + I(inca^2) + plow + tgif + lgif +
                  rgif + tdon + tlag + agif, data.train.std.y)

summary(model.ls1)

# check for multicollinearity
as.data.frame(sort(vif(model.ls1),decreasing=TRUE))

pred.valid.ls1 <- predict(model.ls1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error



# best subset selection with k-fold cross-validation

library(leaps)
regfit.full=regsubsets(damt~.,data.train.std.y,nvmax=20)
summary(regfit.full)
reg.summary=summary(regfit.full)
reg.summary$rsq
reg.summary$adjr2

par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables ",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(20,reg.summary$rss[20], col="red",cex=2,pch=20)

plot(reg.summary$adjr2 ,xlab="Number of Variables ",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(15,reg.summary$adjr2[15], col="red",cex=2,pch=20)


# cp
plot(reg.summary$cp,xlab="Number of Variables ",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(14,reg.summary$cp[14], col ="red",cex=2,pch =20)

which.min(reg.summary$bic)
plot(reg.summary$bic,xlab="Number of Variables ",ylab="BIC",type='l')
points(11,reg.summary$bic[11],col="red",cex=2,pch =20)

par(mfrow=c(1,1))
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")

coef(regfit.full,20)


# forward and backward stepwise selection

regfit.fwd=regsubsets(damt~.,data.train.std.y,nvmax=20,method ="forward")
summary(regfit.fwd)

regfit.bwd=regsubsets(damt~.,data.train.std.y,nvmax=20,method="backward")
summary(regfit.bwd)

# best subset
regfit.best=regsubsets(damt~.,data=data.train.std.y,nvmax=20)

test.mat=model.matrix(damt~.,data=data.valid.std.y)

val.errors=rep(NA,20)

for(i in 1:20){
  coefi=coef(regfit.best,id=i)
  pred=test.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((y.valid-pred)^2)
}

val.errors
which.min(val.errors)
coef(regfit.best,17)


pred.regfit.best <- pred
mean((y.valid - pred.regfit.best)^2) # mean prediction error
sd((y.valid - pred.regfit.best)^2)/sqrt(n.valid.y) # std error


predict.regsubsets=function (object,newdata,id ,...){
  form=as.formula (object$call [[2]])
  mat=model.matrix(form ,newdata )
  coefi=coef(object ,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
  }


k=10
folds=sample(1:k,nrow(data.valid.std.y),replace=TRUE)
cv.errors=matrix(NA,k,20,dimnames=list(NULL,paste(1:20)))

for(j in 1:k){
  best.fit=regsubsets(damt~.,data.train.std.y[folds!=j,],nvmax=20)
  for(i in 1:20){
    pred=predict(best.fit,data.valid.std.y[folds==j,],id=i)
    cv.errors[j,i]=mean((y.valid[folds==j]-pred)^2)
  }
}

mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors
which.min(mean.cv.errors)


# principal components regression

library(pls)

model.pcr1=pcr(damt ~ .,data=data.train.std.y,scale=FALSE,validation="CV")
summary(model.pcr1)

validationplot(model.pcr1,val.type="MSEP")
validationplot(model.pcr1,val.type="R2",main="% Variance Explained by Principal Components")
points(12,.61,col="red",cex=2,pch =20)

MSEP(model.pcr1)


pred.valid.pcr1 <- predict(model.pcr1,newdata = data.valid.std.y,ncomp=12) # validation predictions
mean((y.valid - pred.valid.pcr1)^2) # mean prediction error
sd((y.valid - pred.valid.pcr1)^2)/sqrt(n.valid.y) # std error


pcr.fit <- pcr(damt~.,data=data.valid.std.y,scale=FALSE,ncomp=12)
summary(pcr.fit)



# partial least squares

model.pls1=plsr(damt~.,data=data.train.std.y,scale=FALSE,validation="CV")
summary(model.pls1)

validationplot(model.pls1,val.type="MSEP")


# Test set prediction, MSE, sd

pred.valid.pls1 <- predict(model.pls1,newdata = data.valid.std.y,ncomp=3) # validation predictions
mean((y.valid - pred.valid.pls1)^2) # mean prediction error
sd((y.valid - pred.valid.pls1)^2)/sqrt(n.valid.y) # std error


MSEP(model.pls1)



# ridge regression

library(glmnet)

ridge.x=model.matrix(damt~.,data.train.std.y)[,-1]
ridge.x.valid=model.matrix(damt~.,data.valid.std.y)[,-1]
ridge.y=data.train.std.y$damt


grid=10^seq(10,-2,length=100)

model.ridge=glmnet(ridge.x,ridge.y,alpha=0,standardize=FALSE,lambda=grid)

dim(coef(model.ridge))

model.ridge$lambda[50]
coef(model.ridge)[,50]
sqrt(sum(coef(model.ridge)[-1,50]^2))

model.ridge$lambda[60]
coef(model.ridge)[,60]
sqrt(sum(coef(model.ridge)[-1,60]^2))

predict(model.ridge,s=50,type="coefficients")



model.ridge2=glmnet(ridge.x,ridge.y,alpha=0,lambda=grid,thresh=1e-12)
ridge.pred2=predict(model.ridge2,s=0,newx=ridge.x.valid)
mean((y.valid - ridge.pred2)^2)
sd((y.valid - ridge.pred2)^2)/sqrt(n.valid.y) # std error


plot(model.ridge2, xvar="lambda", label=T)


cv.out=cv.glmnet(ridge.x,ridge.y,alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam

ridge.pred3=predict(model.ridge2,s=bestlam,newx=ridge.x.valid)
mean((y.valid - ridge.pred3)^2)

out=glmnet(ridge.x.valid,y.valid,alpha=0)
predict(out,type="coefficients",s=bestlam)


# lasso

lasso.x=model.matrix(damt~.,data.train.std.y)[,-1]
lasso.x.valid=model.matrix(damt~.,data.valid.std.y)[,-1]
lasso.y=data.train.std.y$damt

model.lasso=glmnet(lasso.x,lasso.y,alpha=1,lambda=grid)
plot(model.lasso)


# We now perform cross-validation and compute the associated test error
lasso.cv.out=cv.glmnet(lasso.x,lasso.y,alpha=1)
plot(lasso.cv.out)
lasso.bestlam=lasso.cv.out$lambda.min
lasso.bestlam

lasso.pred=predict(model.lasso,s=cv.out$lambda.1se,newx=lasso.x.valid)
mean((y.valid - lasso.pred)^2)
sd((y.valid - lasso.pred)^2)/sqrt(n.valid.y) # std error

predict(model.lasso,s=cv.out$lambda.1se,type="coefficients")



# Results

# select model.ls1 since it has minimum mean prediction error in the validation sample

yhat.test <- predict(model.ls1, newdata = data.test.std) # test predictions
yhat.test <- ifelse(chat.test==0, 0, yhat.test) # remove prediction amounts for 0 pred




# FINAL RESULTS

# Save final results for both classification and regression

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="cab.csv", row.names=FALSE) # use your initials for the file name

