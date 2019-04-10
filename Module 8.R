library(tree)
library(ISLR)
attach(Carseats)

# 8.3.1 Fitting Classification Trees

High=ifelse(Sales<=8,"No","Yes")

Carseats=data.frame(Carseats,High)

# We now use the tree() function to fit a classification tree in order to predict High 
# using all variables but Sales. The syntax of the tree() function is quite similar to 
# that of the lm() function.

tree.carseats=tree(High~.-Sales,Carseats)

summary(tree.carseats)

plot(tree.carseats)
text(tree.carseats,pretty=0)

tree.carseats


# In order to properly evaluate the performance of a classification tree on these data, 
# we must estimate the test error rather than simply computing the training error. We 
# split the observations into a training set and a test set, build the tree using the 
# training set, and evaluate its performance on the test data. The predict() function can 
# be used for this purpose. In the case of a classification tree, the argument type="class" 
# instructs R to return the actual class prediction. This approach leads to correct 
# predictions for around 71.5 % of the locations in the test data set.

set.seed(2)
train=sample(1:nrow(Carseats),200)
Carseats.test=Carseats[-train,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
tree.pred=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred,High.test)

(86+57)/200



# Next, we consider whether pruning the tree might lead to improved results. The function 
# cv.tree() performs cross-validation in order to determine the optimal level of tree 
# complexity; cost complexity pruning is used in order to select a sequence of trees for 
# consideration. We use the argument FUN=prune.misclass in order to indicate that we want 
# the classification error rate to guide the cross-validation and pruning process, rather 
# than the default for the cv.tree() function, which is deviance. The cv.tree() function 
# reports the number of terminal nodes of each tree considered (size) as well as the 
# corresponding error rate and the value of the cost-complexity parameter used (k, which 
# corresponds to α in (8.4)).

set.seed(3)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)
names(cv.carseats)
cv.carseats


# Note that, despite the name, dev corresponds to the cross-validation error rate in this 
# instance. The tree with 9 terminal nodes results in the lowest cross-validation error 
# rate, with 50 cross-validation errors. We plot the error rate as a function of both 
# size and k.

par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")
par(mfrow=c(1,1))

# We now apply the prune.misclass() function in order to prune the tree to obtain the 
# nine-node tree.

prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats)
text(prune.carseats,pretty=0)

summary(prune.carseats)


# How well does this pruned tree perform on the test data set? Once again, we apply the 
# predict() function.

tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(94+60)/200


# Now 77 % of the test observations are correctly classified, so not only has the pruning 
# process produced a more interpretable tree, but it has also improved the classification 
# accuracy. If we increase the value of best, we obtain a larger pruned tree with lower 
# classification accuracy:

prune.carseats=prune.misclass(tree.carseats,best=13)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(86+62)/200

(91+63)/200


summary(prune.carseats)

# 8.3.2 Fitting Regression Trees

library(MASS)
set.seed(1)
train=sample(1:nrow(Boston),nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)

plot(tree.boston)
text(tree.boston,pretty=0)

# Now we use the cv.tree() function to see whether pruning the tree will improve 
# performance.

cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type='b')

# In this case, the most complex tree is selected by cross-validation. However, if we 
# wish to prune the tree, we could do so as follows, using the prune.tree() function:

prune.boston=prune.tree(tree.boston,best=5)
plot(prune.boston)
text(prune.boston,pretty=0)

# In keeping with the cross-validation results, we use the unpruned tree to make 
# predictions on the test set.

yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2)


# 8.3.3 Bagging and Random Forests

library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance =TRUE)
bag.boston
  
# The argument mtry=13 indicates that all 13 predictors should be considered for each 
# split of the tree—in other words, that bagging should be done. How well does this 
# bagged model perform on the test set?

yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag,boston.test)
abline (0,1)
mean((yhat.bag -boston.test)^2)


# The test set MSE associated with the bagged regression tree is 13.16, almost half 
# that obtained using an optimally-pruned single tree. We could change the number of 
# trees grown by randomForest() using the ntree argument:

bag.boston= randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=500)
yhat.bag = predict(bag.boston , newdata=Boston[-train,])
mean((yhat.bag -boston.test)^2)


plot(yhat.bag,boston.test)
abline (0,1)


# Here we use mtry = 6.
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston ,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)

set.seed(1)
rf.boston1=randomForest(medv~.,data=Boston,subset=train,mtry=4,importance=TRUE)
yhat.rf = predict(rf.boston1,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)

importance(rf.boston)

# Plots of these importance measures can be produced using the varImpPlot() function.

varImpPlot(rf.boston)


# 8.3.4 Boosting

library(gbm)
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train ,],distribution="gaussian",
                 n.trees=5000,interaction.depth=4)
summary(boost.boston)

# We see that lstat and rm are by far the most important variables. We can also produce 
# partial dependence plots for these two variables. These plots illustrate the marginal 
# effect of the selected variables on the response after integrating out the other 
# variables. In this case, as we might expect, median house prices are increasing with 
# rm and decreasing with lstat.

par(mfrow=c(1,2))
plot(boost.boston,i="dis")
plot(boost.boston,i="lstat")
par(mfrow=c(1,1))


# We now use the boosted model to predict medv on the test set:

yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost - boston.test)^2)


# The test MSE obtained is 11.8; similar to the test MSE for random forests and superior 
# to that for bagging. If we want to, we can perform boosting with a different value of 
# the shrinkage parameter λ in (8.10). The default value is 0.001, but this is easily 
# modified. Here we take λ = 0.2.

boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,
                 shrinkage=0.2,verbose=F)
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost - boston.test)^2)

set.seed(1)
boost.boston1=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=3,
                 shrinkage=0.01,verbose=F)
yhat.boost1=predict(boost.boston1,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost1 - boston.test)^2)


# In this case, using λ = 0.2 leads to a slightly lower test MSE than λ = 0.001.



