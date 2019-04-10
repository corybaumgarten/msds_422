library(ISLR)
names(Smarket)
dim(Smarket)
summary(Smarket)

pairs(Smarket,col=Direction)

cor(Smarket)
cor(Smarket[,-9])

attach(Smarket)
plot(Volume)


# 4.6.2 Logistic Regression


glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial)
summary (glm.fits)

coef(glm.fits)
summary(glm.fits)$coef

summary(glm.fits)$coef[,4]

glm.probs=predict(glm.fits,type="response")
glm.probs[1:10]

contrasts(Direction)

# In order to make a prediction as to whether the market will go up or down on a 
# particular day, we must convert these predicted probabilities into class labels, Up 
# or Down. The following two commands create a vector of class predictions based on 
# whether the predicted probability of a market increase is greater than or less than 
# 0.5.

# The first command creates a vector of 1,250 Down elements. The second line transforms 
# to Up all of the elements for which the predicted probability of a market increase 
# exceeds 0.5. Given these predictions, the table() function table() can be used to 
# produce a confusion matrix in order to determine how many observations were correctly 
# or incorrectly classified.

glm.pred=rep("Down",1250)
glm.pred[glm.probs >.5]="Up"


# table() function can be used to produce a confusion matrix in order to determine 
# how many observations were correctly or incorrectly classified.

table(glm.pred,Direction)


(507+145)/1250
mean(glm.pred==Direction)

# The diagonal elements of the confusion matrix indicate correct predictions, while the 
# off-diagonals represent incorrect predictions. Hence our model correctly predicted 
# that the market would go up on 507 days and that it would go down on 145 days, for a 
# total of 507 + 145 = 652 correct predictions.


# Need to evaluate error in test sample.

train=(Year<2005)
Smarket.2005=Smarket[!train,]
dim(Smarket.2005)
Direction.2005=Direction[!train]


glm.fits=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial,subset=train)
glm.probs=predict(glm.fits,Smarket.2005,type="response")

glm.pred=rep("Down",252)
glm.pred[glm.probs >.5]="Up"

table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005)
mean(glm.pred!=Direction.2005)

# refit logistic regression with just lag1 and lag2

glm.fits=glm(Direction~Lag1+Lag2,data=Smarket,family=binomial,subset=train)
glm.probs=predict(glm.fits,Smarket.2005,type="response")
glm.pred=rep("Down",252)
glm.pred[glm.probs >.5]="Up"

table(glm.pred,Direction.2005)
mean(glm.pred==Direction.2005)

106/(106+76)
106/(106+76+35+35)

1-(76/(76+35))

106/(106+35)

summary(glm.fits)




# Suppose that we want to predict the returns associated with particular values of 
# Lag1 and Lag2. In particular, we want to predict Direction on a day when Lag1 and 
# Lag2 equal 1.2 and 1.1, respectively, and on a day when they equal 1.5 and −0.8. We 
# do this using the predict() function.

predict(glm.fits,newdata=data.frame(Lag1=c(1.2 ,1.5),Lag2=c(1.1,-0.8)),type="response")



# 4.6.3 Linear Discriminant Analysis

library(MASS)
lda.fit=lda(Direction~Lag1+Lag2,data=Smarket,subset=train)
lda.fit

plot(lda.fit)

lda.pred=predict(lda.fit,Smarket.2005)
names(lda.pred)

lda.class=lda.pred$class
table(lda.class,Direction.2005)
mean(lda.class==Direction.2005)

106/(106+35)
35/(35+76)

106/252
35/(35+35+76+106)

sum(lda.pred$posterior[,1]>=.5)
sum(lda.pred$posterior[,1]<.5)

lda.pred$posterior[1:20,1]
lda.class[1:20]

sum(lda.pred$posterior[,1]>.9)


max(lda.pred$posterior[,1])
min(lda.pred$posterior[,1])


# 4.6.4 Quadratic Discriminant Analysis

qda.fit=qda(Direction~Lag1+Lag2,data=Smarket,subset=train)
qda.fit

qda.class=predict(qda.fit,Smarket.2005)$class
table(qda.class,Direction.2005)

121/(121+81)
30/(30+20)

mean(qda.class==Direction.2005)


# 4.6.5 K-Nearest Neighbors

library(class)
train.X=cbind(Lag1,Lag2)[train,]
test.X=cbind(Lag1,Lag2)[!train,]
train.Direction=Direction[train]

set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.2005)
(83+43)/252

knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)

mean(knn.pred==Direction.2005)

set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)



# 4.6.6 An Application to Caravan Insurance Data

dim(Caravan)
attach(Caravan)
summary(Purchase)
348/5822


standardized.X=scale(Caravan[,-86])
var(Caravan[,1])
var(Caravan[,2])
var(standardized.X[,1])
var(standardized.X[,2])


test=1:1000
train.X=standardized.X[-test,]
test.X=standardized.X[test,]
train.Y=Purchase[-test]
test.Y=Purchase[test]

set.seed(1)
knn.pred=knn(train.X,test.X,train.Y,k=1)
mean(test.Y!=knn.pred)
mean(test.Y!="No")

table(knn.pred ,test.Y)
9/(68+9)

knn.pred=knn(train.X,test.X,train.Y,k=3)
table(knn.pred,test.Y)
5/26

knn.pred=knn(train.X,test.X,train.Y,k=5)
table(knn.pred,test.Y)
4/15


glm.fits=glm(Purchase~.,data=Caravan,family=binomial,subset=-test)
glm.probs=predict(glm.fits,Caravan[test,],type="response")

glm.pred=rep("No",1000)
glm.pred[glm.probs >.5]="Yes"
table(glm.pred,test.Y)

glm.pred=rep("No",1000)
glm.pred[glm.probs >.25]="Yes"
table(glm.pred ,test.Y)
11/(22+11)


lda.fit=lda(Purchase~.,data=Caravan,subset=-test)
lda.probs=predict(lda.fit, Caravan[test,])$posterior[,2]
lda.pred=rep("No",1000)
lda.pred[lda.probs>.25]="Yes"
table(lda.pred,test.Y)


#logistic
11/(11+48)

#lda
13/(13+46)

