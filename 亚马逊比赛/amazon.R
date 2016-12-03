

Winning solution code and methodology
http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/5283/winning-solution-code-and-methodology

Python code to achieve 0.90 AUC with Logistic Regression  
http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/4838/python-code-to-achieve-0-90-auc-with-logistic-regression

Starter code in python with scikit-learn (AUC .885)  
http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/4797/starter-code-in-python-with-scikit-learn-auc-885  

Patterns in Training data set  
http://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/4886/patterns-in-training-data-set  


######################################
## 读取数据
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon\\data')

train <- read.csv('train.csv', head=T)
test <- read.csv('test.csv', head=T)
names(train) <- tolower(names(train))
names(test) <- tolower(names(test))
head(train)
head(test)

# 没有id字段的话就加上一个
train <- data.frame(id=paste('train', seq_len(nrow(train)), sep='_'), train)
# 测试样本没有应变量字段，加上一个
test <- data.frame(action=NA, test)
# 拼成一个数据集
test <- test[names(train)]
train$id <- as.character(train$id)
test$id <- as.character(test$id)
d <- rbind(train, test)
rm(train,test);gc()
d[sample(nrow(d),10),]

# 分解成 id、应变量和自变量
d <- d[order(d$id), ]
id <- d$id
y <- d$action
x <- d[, !names(d) %in% c('id','action','role_code')]

save(id, x, y, file='xyid_raw.RData')


######################################
## 第一次拟合 线性模型 逻辑回归
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_raw.RData')
d <- cbind(y,x)
idx_train <- !is.na(y)
train <- d[idx_train, ]
test <- d[!idx_train, ]

fit_logistic <- glm(data=train, y~., family=binomial(link=logit))
summary(fit_logistic)
res1 <- data.frame(id=id[!idx_train], Action=predict(fit_logistic, newdata=test, type='response'))
summary(res1)
write.csv(res1, file='submission\\res1.csv', row.names=F, quote=F)
# 得分 0.54243 1523/1687


######################################
## 第二次拟合 非线性模型 决策树
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_raw.RData')
d <- cbind(y,x)
idx_train <- !is.na(y)

d$y <- as.factor(d$y)
train <- d[idx_train, ]
test <- d[!idx_train, ]

require(rpart)
fit_tree <- rpart(data=train, y~., cp=0)
(xxx <- printcp(fit_tree))
cp_1 <- xxx[which.min(xxx[, 4]), 1]
fit_tree1 <- prune(fit_tree, cp=cp_1)
printcp(fit_tree1)
res2 <- data.frame(id=id[!idx_train], Action=predict(fit_tree1, newdata=test, type='prob')[,2])
summary(res2)
write.csv(res2, file='submission\\res2.csv', row.names=F, quote=F)
# 得分 0.71596 1165/1687


######################################
## 第三次拟合 非线性模型 加上一些衍生变量
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_raw.RData')

# 单变量的频数
cnt_1 <- NULL
for(i in 1:ncol(x)){
  vi <- as.character(x[, i])
  cnt1 <- table(vi)[vi]
  cnt_1 <- as.data.frame(cbind(cnt_1, cnt1))
  names(cnt_1)[ncol(cnt_1)] <- paste('cnt1', names(x)[i], sep='_')
  row.names(cnt_1) <- NULL
  cat(i, '\n')
}

# 二元变量组合的频数和占比
cnt_2 <- data.frame(xxxxx=seq_len(nrow(x)))
for(i in 1:(ncol(x)-1)){
  for(j in (i+1):ncol(x)){
    vi <- as.character(x[, i])
    vj <- as.character(x[, j])
    vij <- paste(vi, vj)
    cnt_i <- table(vi)[vi]
    cnt_j <- table(vj)[vj]
    cnt_ij <- table(vij)[vij]
    cnt2 <- data.frame(cnt_ij=cnt_ij, 
                       ratio_i=cnt_ij/cnt_i, 
                       ratio_j=cnt_ij/cnt_j)
    names(cnt2) <- paste('cnt2', names(x)[i], names(x)[j], names(cnt2), sep='_')
    cnt_2 <- as.data.frame(cbind(cnt_2, cnt2))
    row.names(cnt_2) <- NULL
    cat(i, j, '\n')
  }
}
cnt_2 <- cnt_2[, -1]

# 三元变量组合的频数
cnt_3 <- NULL
for(i in 1:(ncol(x)-2)){
  for(j in (i+1):(ncol(x)-1)){
    for(k in (j+1):ncol(x)){
      vi <- as.character(x[, i])
      vj <- as.character(x[, j])
      vk <- as.character(x[, k])
      vijk <- paste(vi, vj, vk)
      cnt3 <- table(vijk)[vijk]
      cnt_3 <- as.data.frame(cbind(cnt_3, cnt3))
      names(cnt_3)[ncol(cnt_3)] <- paste('cnt3', names(x)[i], names(x)[j], names(x)[k], sep='_')
      row.names(cnt_3) <- NULL
      cat(i, j, k, '\n')
    }
  }
}

xx <- cbind(cnt_1, cnt_2, cnt_3)
names(xx)
any(duplicated(names(xx)))

# 去掉取值不变的衍生变量
xx_train <- xx[!is.na(y), ]
range_x <- apply(xx_train, 2, function(z){max(z)-min(z)})
sum(range_x == 0)
x <- xx[, range_x > 0]

save(id, x, y, file='data\\xyid_cntratio.RData')

d <- cbind(y,x)
idx_train <- !is.na(y)

d$y <- as.factor(d$y)
train <- d[idx_train, ]
test <- d[!idx_train, ]

require(rpart)
fit_tree <- rpart(data=train, y~., cp=0)
(xxx <- printcp(fit_tree))
cp_1 <- xxx[which.min(xxx[, 4]), 1]
fit_tree1 <- prune(fit_tree, cp=cp_1)
printcp(fit_tree1)
res3 <- data.frame(id=id[!idx_train], Action=predict(fit_tree1, newdata=test, type='prob')[,2])
summary(res3)
write.csv(res3, file='submission\\res3.csv', row.names=F, quote=F)
# 得分 0.68295 1265/1687


######################################
## 第四次拟合 非线性模型 衍生变量加原始变量
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_raw.RData')
x1 <- x
y1 <- y
id1 <- id
load('data\\xyid_cntratio.RData')
x2 <- x
y2 <- y
id2 <- id

identical(id1,id2)
identical(y1,y2)
id <- id1
y <- y1
x <- cbind(x1,x2)
save(id, x, y, file='data\\xyid_raw_cntratio.RData')

d <- cbind(y,x)
idx_train <- !is.na(y)

d$y <- as.factor(d$y)
train <- d[idx_train, ]
test <- d[!idx_train, ]

require(rpart)
fit_tree <- rpart(data=train, y~., cp=0)
(xxx <- printcp(fit_tree))
cp_1 <- xxx[which.min(xxx[, 4]), 1]
fit_tree1 <- prune(fit_tree, cp=cp_1)
printcp(fit_tree1)
res4 <- data.frame(id=id[!idx_train], Action=predict(fit_tree1, newdata=test, type='prob')[,2])
summary(res4)
write.csv(res4, file='submission\\res4.csv', row.names=F, quote=F)
# 得分 0.68397 1261/1687


######################################
## 第五次拟合 更高级的模型 glmnet
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_cntratio.RData')

x <- as.matrix(x)
x <- scale(x)
idx_train <- !is.na(y)
y <- as.factor(y)

require(glmnet)
set.seed(114)
fit_glmnet <- cv.glmnet(x=x[idx_train, ], y=y[idx_train], 
                        family='binomial', alpha=0.5, standardize=F)
res5 <- data.frame(id=id[!idx_train], Action=predict(fit_glmnet, x[!idx_train, ], type='response', s='lambda.min')[, 1])
summary(res5)
write.csv(res5, file='submission\\res5.csv', row.names=F, quote=F)
# 得分 0.75786 1104/1687


######################################
## 第六次拟合 更高级的模型 gbm 原始变量
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_raw.RData')

idx_train <- !is.na(y)

require(gbm)
require(verification)
f_loss <- function(a, p){
  1-roc.area(as.integer(as.character(a)), p)$A
}
set.seed(114)
fit_gbm_1 <- gbm.fit(x=x[idx_train, ], y=y[idx_train], 
                     var.names=names(x), 
                     n.trees=50, 
                     interaction.depth=8, 
                     shrinkage=0.01, 
                     distribution='bernoulli', 
                     verbose=F)
ntree_best <- gbm.perf(fit_gbm_1, method='OOB')
cat(paste('ntry_tree=', 50, ' oob_ntree=', ntree_best, sep=''), '\n')
if(ntree_best >= 50){
  for(ntree in seq(50*2, 10000, 50)){
    set.seed(114)
    fit_gbm_1 <- gbm.more(fit_gbm_1, 50)
    ntree_best <- gbm.perf(fit_gbm_1, method='OOB')
    cat(paste('ntry_tree=', ntree, ' oob_ntree=', ntree_best, sep=''), '\n')
    if(ntree_best < (ntree-10)) break
  }
}
cat(paste('final_ntree=', ntree_best, sep=''), '\n')
res6 <- data.frame(id=id[!idx_train], Action=predict(fit_gbm_1, as.data.frame(x[!idx_train, ]), n.trees=ntree_best, type='response'))
summary(res6)
write.csv(res6, file='submission\\res6.csv', row.names=F, quote=F)
# 得分 0.80393 1020/1687


######################################
## 第七次拟合 更高级的模型 gbm 衍生变量
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_cntratio.RData')

idx_train <- !is.na(y)

require(gbm)
require(verification)
f_loss <- function(a, p){
  1-roc.area(as.integer(as.character(a)), p)$A
}
set.seed(114)
fit_gbm_2 <- gbm.fit(x=x[idx_train, ], y=y[idx_train], 
                     var.names=names(x), 
                     n.trees=50, 
                     interaction.depth=8, 
                     shrinkage=0.01, 
                     distribution='bernoulli', 
                     verbose=F)
ntree_best <- gbm.perf(fit_gbm_2, method='OOB')
cat(paste('ntry_tree=', 50, ' oob_ntree=', ntree_best, sep=''), '\n')
if(ntree_best >= 50){
  for(ntree in seq(50*2, 10000, 50)){
    set.seed(114)
    fit_gbm_2 <- gbm.more(fit_gbm_2, 50)
    ntree_best <- gbm.perf(fit_gbm_2, method='OOB')
    cat(paste('ntry_tree=', ntree, ' oob_ntree=', ntree_best, sep=''), '\n')
    if(ntree_best < (ntree-10)) break
  }
}
cat(paste('final_ntree=', ntree_best, sep=''), '\n')
res7 <- data.frame(id=id[!idx_train], Action=predict(fit_gbm_2, as.data.frame(x[!idx_train, ]), n.trees=ntree_best, type='response'))
summary(res7)
write.csv(res7, file='submission\\res7.csv', row.names=F, quote=F)
# 得分 0.86131 855/1687


######################################
## 第八次拟合 更高级的模型 gbm 原始变量 用交叉验证确定参数
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_raw.RData')

# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)

require(gbm)
require(verification)
f_loss <- function(a, p){
  1-roc.area(as.integer(as.character(a)), p)$A
}
out <- list()

# cv initialization
cv_out <- list()
models <- list()
for(i in 1:5){
  cat(i, '\t')
  idx_train <- which(id %in% unlist(ids[-i]))
  idx_test <- which(id %in% unlist(ids[i]))
  set.seed(114)
  models[[i]] <- gbm.fit(x=x[idx_train, ], y=y[idx_train], 
                         var.names=names(x), 
                         n.trees=50, 
                         interaction.depth=8, 
                         shrinkage=0.01, 
                         distribution='bernoulli', 
                         verbose=F)
  cv_out[[i]] <- data.frame(stringsAsFactors=F, 
                            id=id[idx_test], 
                            y_predict=predict(models[[i]], as.data.frame(x[idx_test, ]), 
                                              n.trees=50, type='response'), 
                            y_test=y[idx_test])
}
cv_out_tt <- do.call(rbind, cv_out)
metric_best <- f_loss(cv_out_tt$y_test, cv_out_tt$y_predict)
ntree_best <- 50
cat(paste('initial_ntree=', 50, ' - initial_loss=', round(metric_best, 6), sep=''), '\n')

# add at each step ntree_step trees, stop when no more gain in accuracy
loss_cnt <- 0
for(ntree in seq(50*2, 10000, 50)){
  for(i in 1:5){
    cat(i, '\t')
    idx_test <- which(id %in% unlist(ids[i]))
    set.seed(114)
    models[[i]] <- gbm.more(models[[i]], 50)
    cv_out[[i]][['y_predict']] <- predict(models[[i]], as.data.frame(x[idx_test, ]), 
                                          n.trees=ntree, type='response')
  }
  cv_out_tt <- do.call(rbind, cv_out)
  metric_now <- f_loss(cv_out_tt$y_test, cv_out_tt$y_predict)
  cat(paste('now_ntree=', ntree, 
            ' - min_loss=', round(metric_best, 6), 
            ' - now_loss=', round(metric_now, 6), sep=''), '\n')
  
  if(metric_now >= (metric_best+0.0001)) break
  if(metric_now >= (metric_best-0.0001)){
    loss_cnt <- loss_cnt+1
  }
  if(loss_cnt >= 5) break
  if(metric_now < (metric_best-0.0001)){
    metric_best <- metric_now
    ntree_best <- ntree
    loss_cnt <- 0
  }
}
# 1   2   3 	4 	5 	now_ntree=10000 - min_loss=0.16963 - now_loss=0.169561 

out$ntree_best <- ntree_best

# cv result
for(i in 1:5){
  cat(i, '\t')
  idx_test <- which(id %in% unlist(ids[i]))
  set.seed(114)
  cv_out[[i]][['y_predict']] <- predict(models[[i]], as.data.frame(x[idx_test, ]), 
                                        n.trees=ntree_best, type='response')
}
cv_out_tt <- do.call(rbind, cv_out)
metric_cv <- f_loss(cv_out_tt$y_test, cv_out_tt$y_predict)
cat(paste('final_ntree=', ntree_best, ' - cv_loss=', round(metric_cv, 6), sep=''), '\n')
idss <- unlist(ids)
idss <- data.frame(id=idss, fold=rep(1:5,times=sapply(ids,length)))
cv_out_tt <- merge(cv_out_tt,idss,by='id',sort=F)
metric_cv_fold <- by(cv_out_tt,cv_out_tt[['fold']],function(z){f_loss(z$y_test, z$y_predict)})
cat('cv_loss_fold =', round(metric_cv_fold, 6), '\n')
out$cv <- cv_out_tt
# final_ntree=9900 - cv_loss=0.16963 
# cv_loss_fold = 0.184119 0.1558 0.167031 0.185946 0.154015 

# train of full training set
idx_train <- which(id %in% unlist(ids))
idx_test <- which(!id %in% unlist(ids))
set.seed(114)
out$model <- gbm.fit(x=x[idx_train, ], y=y[idx_train], 
                     var.names=names(x), 
                     n.trees=ntree_best, 
                     interaction.depth=8, 
                     shrinkage=0.01, 
                     distribution='bernoulli', 
                     keep.data=F, 
                     verbose=F)
out$prediction <- data.frame(stringsAsFactors=F, 
                             id=id[idx_test], 
                             y_predict=predict(out$model, as.data.frame(x[idx_test, ]), 
                                               n.trees=ntree_best, type='response'), 
                             y_test=y[idx_test])
res8 <- data.frame(id=out$prediction$id, Action=out$prediction$y_predict)
summary(res8)
write.csv(res8, file='submission\\res8.csv', row.names=F, quote=F)
# 得分 0.85951 867/1687


######################################
## 第九次拟合 更高级的模型 gbm 衍生变量 用交叉验证确定参数
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_cntratio.RData')

# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)

require(gbm)
require(verification)
f_loss <- function(a, p){
  1-roc.area(as.integer(as.character(a)), p)$A
}
out <- list()

# cv initialization
cv_out <- list()
models <- list()
for(i in 1:5){
  cat(i, '\t')
  idx_train <- which(id %in% unlist(ids[-i]))
  idx_test <- which(id %in% unlist(ids[i]))
  set.seed(114)
  models[[i]] <- gbm.fit(x=x[idx_train, ], y=y[idx_train], 
                         var.names=names(x), 
                         n.trees=50, 
                         interaction.depth=8, 
                         shrinkage=0.01, 
                         distribution='bernoulli', 
                         verbose=F)
  cv_out[[i]] <- data.frame(stringsAsFactors=F, 
                            id=id[idx_test], 
                            y_predict=predict(models[[i]], as.data.frame(x[idx_test, ]), 
                                              n.trees=50, type='response'), 
                            y_test=y[idx_test])
}
cv_out_tt <- do.call(rbind, cv_out)
metric_best <- f_loss(cv_out_tt$y_test, cv_out_tt$y_predict)
ntree_best <- 50
cat(paste('initial_ntree=', 50, ' - initial_loss=', round(metric_best, 6), sep=''), '\n')

# add at each step ntree_step trees, stop when no more gain in accuracy
loss_cnt <- 0
for(ntree in seq(50*2, 10000, 50)){
  for(i in 1:5){
    cat(i, '\t')
    idx_test <- which(id %in% unlist(ids[i]))
    set.seed(114)
    models[[i]] <- gbm.more(models[[i]], 50)
    cv_out[[i]][['y_predict']] <- predict(models[[i]], as.data.frame(x[idx_test, ]), 
                                          n.trees=ntree, type='response')
  }
  cv_out_tt <- do.call(rbind, cv_out)
  metric_now <- f_loss(cv_out_tt$y_test, cv_out_tt$y_predict)
  cat(paste('now_ntree=', ntree, 
            ' - min_loss=', round(metric_best, 6), 
            ' - now_loss=', round(metric_now, 6), sep=''), '\n')
  
  if(metric_now >= (metric_best+0.0001)) break
  if(metric_now >= (metric_best-0.0001)){
    loss_cnt <- loss_cnt+1
  }
  if(loss_cnt >= 5) break
  if(metric_now < (metric_best-0.0001)){
    metric_best <- metric_now
    ntree_best <- ntree
    loss_cnt <- 0
  }
}
# 1   2 	3 	4 	5 	now_ntree=5500 - min_loss=0.129047 - now_loss=0.129168 

out$ntree_best <- ntree_best

# cv result
for(i in 1:5){
  cat(i, '\t')
  idx_test <- which(id %in% unlist(ids[i]))
  set.seed(114)
  cv_out[[i]][['y_predict']] <- predict(models[[i]], as.data.frame(x[idx_test, ]), 
                                        n.trees=ntree_best, type='response')
}
cv_out_tt <- do.call(rbind, cv_out)
metric_cv <- f_loss(cv_out_tt$y_test, cv_out_tt$y_predict)
cat(paste('final_ntree=', ntree_best, ' - cv_loss=', round(metric_cv, 6), sep=''), '\n')
idss <- unlist(ids)
idss <- data.frame(id=idss, fold=rep(1:5,times=sapply(ids,length)))
cv_out_tt <- merge(cv_out_tt,idss,by='id',sort=F)
metric_cv_fold <- by(cv_out_tt,cv_out_tt[['fold']],function(z){f_loss(z$y_test, z$y_predict)})
cat('cv_loss_fold =', round(metric_cv_fold, 6), '\n')
out$cv <- cv_out_tt
# final_ntree=5400 - cv_loss=0.129047 
# cv_loss_fold = 0.123546 0.110683 0.124943 0.160976 0.122808 

# train of full training set
idx_train <- which(id %in% unlist(ids))
idx_test <- which(!id %in% unlist(ids))
set.seed(114)
out$model <- gbm.fit(x=x[idx_train, ], y=y[idx_train], 
                     var.names=names(x), 
                     n.trees=ntree_best, 
                     interaction.depth=8, 
                     shrinkage=0.01, 
                     distribution='bernoulli', 
                     keep.data=F, 
                     verbose=F)
out$prediction <- data.frame(stringsAsFactors=F, 
                             id=id[idx_test], 
                             y_predict=predict(out$model, as.data.frame(x[idx_test, ]), 
                                               n.trees=ntree_best, type='response'), 
                             y_test=y[idx_test])
res9 <- data.frame(id=out$prediction$id, Action=out$prediction$y_predict)
summary(res9)
write.csv(res9, file='submission\\res9.csv', row.names=F, quote=F)
# 得分 0.89509 438/1687


######################################
## 模型集成 准备工作，生成基础分类器
######################################
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_1.txt')

rm(list=ls())
load('data\\xyid_cntratio.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_glmnet.R')
require(glmnet)
x <- as.matrix(x)
x <- scale(x)
y <- as.factor(y)
base_learner <- f_glmnet(x, y, id, ids,family='binomial', alpha=0.5, standardize=F)
save(base_learner, file='data\\base_learner_cntratio_glmnet.RData')

rm(list=ls())
load('data\\xyid_raw.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
base_learner <- f_gbm(x, y, id, ids, 
                      family='bernoulli', depth=8, shrk=0.01, 
                      ntree_max=15000, ntree_step=50, 
                      loss_threshold=0.0001, loss_cnt_threshold=5)
save(base_learner, file='data\\base_learner_raw_gbm.RData')

rm(list=ls())
load('data\\xyid_cntratio.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
base_learner <- f_gbm(x, y, id, ids, 
                      family='bernoulli', depth=8, shrk=0.01, 
                      ntree_max=15000, ntree_step=50, 
                      loss_threshold=0.0001, loss_cnt_threshold=5)
save(base_learner, file='data\\base_learner_cntratio_gbm.RData')
sink()

options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
require(glmnet)
require(gbm)
load('data\\base_learner_cntratio_glmnet.RData')
base_learner_glmnet_cntratio <- base_learner
load('data\\base_learner_raw_gbm.RData')
base_learner_gbm_raw <- base_learner
load('data\\base_learner_cntratio_gbm.RData')
base_learner_gbm_cntratio <- base_learner
rm(base_learner)

require(verification)
f_auc <- function(a, p){
  roc.area(as.integer(as.character(a)), p)$A
}

cv <- base_learner_glmnet_cntratio$cv
summary(cv)
cat('AUC_cv=', f_auc(as.integer(as.character(cv$y_test)),cv$y_predict), '\n', sep='')
metric_cv_fold <- by(cv,cv[['fold']],function(z){f_auc(as.integer(as.character(z$y_test)), z$y_predict)})
cat('AUC_cv_fold=', paste(round(metric_cv_fold, 6),collapse=' '), '\n', sep='')
boxplot(metric_cv_fold)

cv <- base_learner_gbm_raw$cv
summary(cv)
cat('AUC_cv=', f_auc(as.integer(as.character(cv$y_test)),cv$y_predict), '\n', sep='')
metric_cv_fold <- by(cv,cv[['fold']],function(z){f_auc(as.integer(as.character(z$y_test)), z$y_predict)})
cat('AUC_cv_fold=', paste(round(metric_cv_fold, 6),collapse=' '), '\n', sep='')
boxplot(metric_cv_fold)

cv <- base_learner_gbm_cntratio$cv
summary(cv)
cat('AUC_cv=', f_auc(as.integer(as.character(cv$y_test)),cv$y_predict), '\n', sep='')
metric_cv_fold <- by(cv,cv[['fold']],function(z){f_auc(as.integer(as.character(z$y_test)), z$y_predict)})
cat('AUC_cv_fold=', paste(round(metric_cv_fold, 6),collapse=' '), '\n', sep='')
boxplot(metric_cv_fold)


######################################
## 第十次拟合 模型集成，简单的求均值
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
load('data\\base_learner_cntratio_glmnet.RData')
base_learner_glmnet_cntratio <- base_learner
load('data\\base_learner_raw_gbm.RData')
base_learner_gbm_raw <- base_learner
load('data\\base_learner_cntratio_gbm.RData')
base_learner_gbm_cntratio <- base_learner
rm(base_learner)

cv1 <- base_learner_glmnet_cntratio$cv
cv2 <- base_learner_gbm_raw$cv
cv3 <- base_learner_gbm_cntratio$cv
prediction1 <- base_learner_glmnet_cntratio$prediction[c('id','y_predict')]
prediction2 <- base_learner_gbm_raw$prediction[c('id','y_predict')]
prediction3 <- base_learner_gbm_cntratio$prediction[c('id','y_predict')]
prediction123 <- merge(prediction1,merge(prediction2,prediction3,by='id'),by='id')
res10 <- data.frame(id=prediction123$id,Action=rowMeans(prediction123[,-1]))
summary(res10)
write.csv(res10, file='submission\\res10.csv', row.names=F, quote=F)
# 得分 0.89075 489/1687


######################################
## 第十一次拟合 模型集成，使用逻辑回归
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
load('data\\base_learner_cntratio_glmnet.RData')
base_learner_glmnet_cntratio <- base_learner
load('data\\base_learner_raw_gbm.RData')
base_learner_gbm_raw <- base_learner
load('data\\base_learner_cntratio_gbm.RData')
base_learner_gbm_cntratio <- base_learner
rm(base_learner)

cv1 <- base_learner_glmnet_cntratio$cv[,c('id','y_test','y_predict')]
cv2 <- base_learner_gbm_raw$cv[,c('id','y_test','y_predict')]
cv3 <- base_learner_gbm_cntratio$cv[,c('id','y_test','y_predict')]
prediction1 <- base_learner_glmnet_cntratio$prediction[,c('id','y_test','y_predict')]
prediction2 <- base_learner_gbm_raw$prediction[,c('id','y_test','y_predict')]
prediction3 <- base_learner_gbm_cntratio$prediction[,c('id','y_test','y_predict')]
d1 <- rbind(cv1, prediction1)
d2 <- rbind(cv2, prediction2)
d3 <- rbind(cv3, prediction3)
d123 <- merge(d1,merge(d2,d3,by='id'),by='id')
d <- d123[,c(1,4,3,5,7)]
names(d) <- c('id','y','x1','x2','x3')
d <- d[order(d$id), ]
id <- d$id
y <- d$y
x <- d[, !names(d) %in% c('id','y')]
d <- cbind(y,x)
idx_train <- !is.na(y)
train <- d[idx_train, ]
test <- d[!idx_train, ]
fit_logistic <- glm(data=train, y~., family=binomial(link=logit))
summary(fit_logistic)
res11 <- data.frame(id=id[!idx_train], Action=predict(fit_logistic, newdata=test, type='response'))
summary(res11)
write.csv(res11, file='submission\\res11.csv', row.names=F, quote=F)
# 得分 0.89482 443/1687


######################################
## 第十二第十三次拟合 模型集成，使用gbm
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
load('data\\base_learner_cntratio_glmnet.RData')
base_learner_glmnet_cntratio <- base_learner
load('data\\base_learner_raw_gbm.RData')
base_learner_gbm_raw <- base_learner
load('data\\base_learner_cntratio_gbm.RData')
base_learner_gbm_cntratio <- base_learner
rm(base_learner)

cv1 <- base_learner_glmnet_cntratio$cv[,c('id','y_test','y_predict')]
cv2 <- base_learner_gbm_raw$cv[,c('id','y_test','y_predict')]
cv3 <- base_learner_gbm_cntratio$cv[,c('id','y_test','y_predict')]
prediction1 <- base_learner_glmnet_cntratio$prediction[,c('id','y_test','y_predict')]
prediction2 <- base_learner_gbm_raw$prediction[,c('id','y_test','y_predict')]
prediction3 <- base_learner_gbm_cntratio$prediction[,c('id','y_test','y_predict')]
d1 <- rbind(cv1, prediction1)
d2 <- rbind(cv2, prediction2)
d3 <- rbind(cv3, prediction3)
d123 <- merge(d1,merge(d2,d3,by='id'),by='id')
d <- d123[,c(1,4,3,5,7)]
names(d) <- c('id','y','x1','x2','x3')
d <- d[order(d$id), ]
id <- d$id
y <- d$y
x <- d[, !names(d) %in% c('id','y')]
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
stacking_learner <- f_gbm(x, y, id, ids, 
                          family='bernoulli', depth=3, shrk=0.01, 
                          ntree_max=15000, ntree_step=50, 
                          loss_threshold=0.0001, loss_cnt_threshold=5)
save(stacking_learner, file='data\\stacking_learner_gbm.RData')
res12 <- data.frame(id=stacking_learner$prediction$id, Action=stacking_learner$prediction$y_predict)
summary(res12)
write.csv(res12, file='submission\\res12.csv', row.names=F, quote=F)
# 得分 0.90163 137/1687

ids <- id[!is.na(y)]
set.seed(911)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
stacking_learner <- f_gbm(x, y, id, ids, 
                          family='bernoulli', depth=3, shrk=0.01, 
                          ntree_max=15000, ntree_step=50, 
                          loss_threshold=0.0001, loss_cnt_threshold=5)
save(stacking_learner, file='data\\stacking_learner_gbm.RData')
res13 <- data.frame(id=stacking_learner$prediction$id, Action=stacking_learner$prediction$y_predict)
summary(res13)
write.csv(res13, file='submission\\res13.csv', row.names=F, quote=F)
# 得分 0.90174 134/1687


######################################
## 构造更多的基础分类器
######################################
# dummy
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_raw.RData')

# 构造dummy变量构成的稀疏矩阵
x2 <- data.frame(xxxxx=seq_len(nrow(x)))
for(i in 1:(ncol(x)-1)){
  for(j in (i+1):ncol(x)){
    vi <- as.character(x[, i])
    vj <- as.character(x[, j])
    xxx <- data.frame(vij=paste(vi, vj))
    names(xxx) <- paste(names(xxx), names(x)[i], names(x)[j], sep='_')
    x2 <- as.data.frame(cbind(x2, xxx))
    row.names(x2) <- NULL
    cat(i, j, '\n')
  }
}
rm(i,j,vi,vj,xxx)
x2 <- x2[, -1]

x3 <- data.frame(xxxxx=seq_len(nrow(x)))
for(i in 1:(ncol(x)-2)){
  for(j in (i+1):(ncol(x)-1)){
    for(k in (j+1):ncol(x)){
      vi <- as.character(x[, i])
      vj <- as.character(x[, j])
      vk <- as.character(x[, k])
      xxx <- data.frame(vijk=paste(vi, vj, vk))
      names(xxx) <- paste(names(xxx), names(x)[i], names(x)[j], names(x)[k], sep='_')
      x3 <- as.data.frame(cbind(x3, xxx))
      row.names(x3) <- NULL
      cat(i, j, k, '\n')
    }
  }
}
rm(i,j,k,vi,vj,vk,xxx)
x3 <- x3[, -1]

x <- cbind(x, x2, x3)
rm(x2,x3)

var_cat <- names(x)
n_row <- nrow(x)

for(i in 1:ncol(x)){
  x[, i] <- as.character(x[, i])
  cat(i, '\n')
}
rm(i)

x <- as.vector(as.matrix(x))
x <- paste(rep(var_cat, each=n_row), x, sep='_')
var_labels <- unique(x)
length(var_labels)
# 1595082
require(Matrix)
id_i <- rep(seq_len(n_row), length(var_cat))
id_j <- match(x, var_labels)
M <- Matrix(0, nrow=n_row, ncol=length(var_labels), 
            dimnames=list(seq_len(n_row), var_labels), sparse=T)
id_ij <- cbind(id_i, id_j)
M[id_ij] <- 1
x <- M

dim(x)
x2 <- x[!is.na(y), ]
a_cnt_train <- colSums(x2)
sum(a_cnt_train==0)
sum(a_cnt_train==nrow(x2))
x <- x[, a_cnt_train > 0 & a_cnt_train < nrow(x2)]
dim(x)
save(id, x, y, file='data\\xyid_dummy_Matrix.RData')

# dummy变量+glmnet
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_2_dummy_glmnet.txt')
rm(list=ls())
load('data\\xyid_dummy_Matrix.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_glmnet.R')
require(glmnet)
# x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_glmnet(x, y, id, ids,family='binomial', alpha=0.5, standardize=F)
save(base_learner, file='data\\base_learner_dummy_glmnet.RData')
sink()

# 原始变量+随机森林
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_3_raw_rf.txt')
rm(list=ls())
load('data\\xyid_raw.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_rf.R')
require(randomForest)
# x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_randomforest(x, y, id, ids, ntree=1000, mtry=2)
save(base_learner, file='data\\base_learner_raw_rf.RData')
sink()

# 原始变量+支持向量机
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_4_raw_svm.txt')
rm(list=ls())
load('data\\xyid_raw.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_svm.R')
require(kernlab)
x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_svm(x, y, id, ids, family='C-svc',
                      scale=T, cost=1, epsilon=0.01, kernel=c('rbfdot', 'vanilladot')[1])
save(base_learner, file='data\\base_learner_raw_svm_rbf.RData')
base_learner <- f_svm(x, y, id, ids, family='C-svc',
                      scale=T, cost=1, epsilon=0.01, kernel=c('rbfdot', 'vanilladot')[2])
save(base_learner, file='data\\base_learner_raw_svm_van.RData')
sink()

# 衍生变量+随机森林 跑挂了
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_5_cntratio_rf.txt')
rm(list=ls())
load('data\\xyid_cntratio.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_rf.R')
require(randomForest)
# x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_randomforest(x, y, id, ids, ntree=3000, mtry=10)
save(base_learner, file='data\\base_learner_cntratio_rf.RData')
sink()

# 衍生变量+随机森林
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_5_cntratio_rf.txt')
rm(list=ls())
load('data\\xyid_cntratio.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_rf.R')
require(randomForest)
# x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_randomforest(x, y, id, ids, ntree=1000, mtry=10)
save(base_learner, file='data\\base_learner_cntratio_rf.RData')
sink()

options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_5_cntratio_rf_2.txt')
rm(list=ls())
load('data\\xyid_cntratio.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_rf.R')
require(randomForest)
# x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_randomforest(x, y, id, ids, ntree=2000, mtry=10)
save(base_learner, file='data\\base_learner_cntratio_rf_2.RData')
sink()

# 衍生变量+支持向量机
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_6_cntratio_svm.txt')
rm(list=ls())
load('data\\xyid_cntratio.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_svm.R')
require(kernlab)
x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_svm(x, y, id, ids, family='C-svc',
                      scale=T, cost=1, epsilon=0.01, kernel=c('rbfdot', 'vanilladot')[1])
save(base_learner, file='data\\base_learner_cntratio_svm_rbf.RData')
base_learner <- f_svm(x, y, id, ids, family='C-svc',
                      scale=T, cost=1, epsilon=0.01, kernel=c('rbfdot', 'vanilladot')[2])
save(base_learner, file='data\\base_learner_cntratio_svm_van.RData')
sink()

# 衍生变量+gbm
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_7_cntratio_gbm.txt')
rm(list=ls())
load('data\\xyid_cntratio.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
base_learner <- f_gbm(x, y, id, ids, 
                      family='bernoulli', depth=20, shrk=0.01, 
                      ntree_max=15000, ntree_step=50, 
                      loss_threshold=0.0001, loss_cnt_threshold=5)
save(base_learner, file='data\\base_learner_cntratio_gbm_2.RData')
sink()


######################################
## 第十四十五十六次拟合 模型集成
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

files <- list.files(path='data', pattern='^base_learner_', full.names=T)

load(files[1])
d <- base_learner$cv[, c('id','y_test')]
for(i in seq_len(length(files))){
  load(files[i])
  d0 <- base_learner$cv[, c('id','y_predict','fold')]
  d <- merge(d, d0, by='id')
  names(d)[c(ncol(d)-1,ncol(d))] <- paste(names(d)[c(ncol(d)-1,ncol(d))], i, sep='_')
  cat(names(d), sep='\n')
}
rm(i,d0,base_learner)
head(d)
for(j in 1:length(files)) cat(eval(parse(text=paste('identical(d$fold_1,d$fold_',j,')',sep=''))),'\n')
rm(j)

require(verification)
f_auc <- function(a, p){
  roc.area(as.integer(as.character(a)), p)$A
}
metric_cv_fold <- matrix(NA,length(files),5)
row.names(metric_cv_fold) <- files
for(i in 1:length(files)){
  metric_cv_fold[i,] <- by(d,d[[paste('fold_',i,sep='')]],
                           function(z){f_auc(as.integer(as.character(z$y_test)),
                                             z[[paste('y_predict_',i,sep='')]])})
}
plot(1:5,1:5,ylim=c(0.5,1),type='n')
for(i in 1:length(files)) points(metric_cv_fold[i,],type='b',col=i)
rm(i,metric_cv_fold)

load(files[1])
d <- rbind(base_learner$cv[, c('id','y_test')],
           base_learner$prediction[, c('id','y_test')])
for(i in seq_len(length(files))){
  load(files[i])
  d0 <- rbind(base_learner$cv[, c('id','y_predict')],
              base_learner$prediction[, c('id','y_predict')])
  d <- merge(d, d0, by='id')
  names(d)[ncol(d)] <- paste('x', i, sep='_')
  cat(names(d), sep='\n')
}
rm(i,d0,base_learner)
d <- d[order(d$id), ]
id <- d$id
y <- d$y_test
x <- d[, !names(d) %in% c('id','y_test')]

# 简单地求均值
res14 <- data.frame(id=id[is.na(y)],Action=rowMeans(x)[is.na(y)])
summary(res14)
write.csv(res14, file='submission\\res14.csv', row.names=F, quote=F)
# 得分 0.91108 70/1687

# 用gbm做集成
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
stacking_learner <- f_gbm(x, y, id, ids, 
                          family='bernoulli', depth=6, shrk=0.01, 
                          ntree_max=15000, ntree_step=50, 
                          loss_threshold=0.0001, loss_cnt_threshold=5)
save(stacking_learner, file='data\\stacking_learner_gbm_15.RData')
res15 <- data.frame(id=stacking_learner$prediction$id, Action=stacking_learner$prediction$y_predict)
summary(res15)
write.csv(res15, file='submission\\res15.csv', row.names=F, quote=F)
# 得分 0.91051 73/1687

# 用gbm做集成
ids <- id[!is.na(y)]
set.seed(911)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
stacking_learner <- f_gbm(x, y, id, ids, 
                          family='bernoulli', depth=6, shrk=0.01, 
                          ntree_max=15000, ntree_step=50, 
                          loss_threshold=0.0001, loss_cnt_threshold=5)
save(stacking_learner, file='data\\stacking_learner_gbm_16.RData')
res16 <- data.frame(id=stacking_learner$prediction$id, Action=stacking_learner$prediction$y_predict)
summary(res16)
write.csv(res16, file='submission\\res16.csv', row.names=F, quote=F)
# 得分 0.91080 72/1687

for(i in seq_len(length(files))){
  load(files[i])
  res <- data.frame(id=base_learner$prediction$id, Action=base_learner$prediction$y_predict)
  summary(res)
  write.csv(res, file=paste('submission\\res16_base_',
                            gsub('.RData|data/base_learner_','',files[i]),'.csv',sep=''),
            row.names=F, quote=F)
}
require(verification)
f_auc <- function(a, p){
  roc.area(as.integer(as.character(a)), p)$A
}
for(i in seq_len(length(files))){
  load(files[i])
  cv <- base_learner$cv
  cat(f_auc(as.integer(as.character(cv$y_test)),cv$y_predict), '\n')
}


######################################
## 第十七十八十九次拟合 去掉不好的基础分类器，再做模型集成
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

files <- list.files(path='data', pattern='^base_learner_', full.names=T)
files <- files[-c(7,11,12)]

load(files[1])
d <- rbind(base_learner$cv[, c('id','y_test')],
           base_learner$prediction[, c('id','y_test')])
for(i in seq_len(length(files))){
  load(files[i])
  d0 <- rbind(base_learner$cv[, c('id','y_predict')],
              base_learner$prediction[, c('id','y_predict')])
  d <- merge(d, d0, by='id')
  names(d)[ncol(d)] <- paste('x', i, sep='_')
  cat(names(d), sep='\n')
}
rm(i,d0,base_learner)
d <- d[order(d$id), ]
id <- d$id
y <- d$y_test
x <- d[, !names(d) %in% c('id','y_test')]

# 简单地求均值
res17 <- data.frame(id=id[is.na(y)],Action=rowMeans(x)[is.na(y)])
summary(res17)
write.csv(res17, file='submission\\res17.csv', row.names=F, quote=F)
# 得分 0.91146 66/1687

# 用gbm做集成
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
stacking_learner <- f_gbm(x, y, id, ids, 
                          family='bernoulli', depth=6, shrk=0.01, 
                          ntree_max=15000, ntree_step=50, 
                          loss_threshold=0.0001, loss_cnt_threshold=5)
save(stacking_learner, file='data\\stacking_learner_gbm_18.RData')
res18 <- data.frame(id=stacking_learner$prediction$id, Action=stacking_learner$prediction$y_predict)
summary(res18)
write.csv(res18, file='submission\\res18.csv', row.names=F, quote=F)
# 得分 0.91098 72/1687

# 用gbm做集成
ids <- id[!is.na(y)]
set.seed(911)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
stacking_learner <- f_gbm(x, y, id, ids, 
                          family='bernoulli', depth=6, shrk=0.01, 
                          ntree_max=15000, ntree_step=50, 
                          loss_threshold=0.0001, loss_cnt_threshold=5)
save(stacking_learner, file='data\\stacking_learner_gbm_19.RData')
res19 <- data.frame(id=stacking_learner$prediction$id, Action=stacking_learner$prediction$y_predict)
summary(res19)
write.csv(res19, file='submission\\res19.csv', row.names=F, quote=F)
# 得分 0.91082 72/1687

require(verification)
f_auc <- function(a, p){
  roc.area(as.integer(as.character(a)), p)$A
}
for(i in seq_len(length(files))){
  load(files[i])
  cv <- base_learner$cv
  cat(f_auc(as.integer(as.character(cv$y_test)),cv$y_predict), '\n')
}


######################################
## 构造更多的基础分类器
######################################
# 更多衍生变量
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

load('data\\xyid_raw.RData')

v0 <- apply(x[, -c(1,2)], 1, paste, collapse='_')
xxx <- table(v0)[v0]
cnt_6 <- as.data.frame(xxx)
names(cnt_6) <- 'cnt6_role'
row.names(cnt_6) <- NULL

v0 <- apply(x[, -1], 1, paste, collapse='_')
xxx <- table(v0)[v0]
cnt_7 <- as.data.frame(xxx)
names(cnt_7) <- 'cnt7_role'
row.names(cnt_7) <- NULL

cnt_1 <- NULL
for(i in 1:ncol(x)){
  vi <- as.character(x[, i])
  cnt1 <- table(vi)[vi]
  cnt_1 <- as.data.frame(cbind(cnt_1, cnt1))
  names(cnt_1)[ncol(cnt_1)] <- paste('cnt1', names(x)[i], sep='_')
  row.names(cnt_1) <- NULL
  cat(i, '\n')
}

cnt_2 <- data.frame(xxxxx=seq_len(nrow(x)))
for(i in 1:(ncol(x)-1)){
  for(j in (i+1):ncol(x)){
    vi <- as.character(x[, i])
    vj <- as.character(x[, j])
    vij <- paste(vi, vj)
    cnt_i <- table(vi)[vi]
    cnt_j <- table(vj)[vj]
    cnt_ij <- table(vij)[vij]
    cnt2 <- data.frame(cnt_ij=cnt_ij, 
                       ratio_i=cnt_ij/cnt_i, 
                       ratio_j=cnt_ij/cnt_j)
    names(cnt2) <- paste('cnt2', names(x)[i], names(x)[j], names(cnt2), sep='_')
    cnt_2 <- as.data.frame(cbind(cnt_2, cnt2))
    row.names(cnt_2) <- NULL
    cat(i, j, '\n')
  }
}
cnt_2 <- cnt_2[, -1]

cnt_3 <- NULL
for(i in 1:(ncol(x)-2)){
  for(j in (i+1):(ncol(x)-1)){
    for(k in (j+1):ncol(x)){
      vi <- as.character(x[, i])
      vj <- as.character(x[, j])
      vk <- as.character(x[, k])
      vijk <- paste(vi, vj, vk)
      cnt3 <- table(vijk)[vijk]
      cnt_3 <- as.data.frame(cbind(cnt_3, cnt3))
      names(cnt_3)[ncol(cnt_3)] <- paste('cnt3', names(x)[i], names(x)[j], names(x)[k], sep='_')
      row.names(cnt_3) <- NULL
      cat(i, j, k, '\n')
    }
  }
}

cnt_4 <- NULL
for(i in 1:(ncol(x)-3)){
  for(j in (i+1):(ncol(x)-2)){
    for(k in (j+1):(ncol(x)-1)){
      for(l in (k+1):ncol(x)){
        vi <- as.character(x[, i])
        vj <- as.character(x[, j])
        vk <- as.character(x[, k])
        vl <- as.character(x[, l])
        vijkl <- paste(vi, vj, vk, vl)
        cnt4 <- table(vijkl)[vijkl]
        cnt_4 <- as.data.frame(cbind(cnt_4, cnt4))
        names(cnt_4)[ncol(cnt_4)] <- paste('cnt4', names(x)[i], names(x)[j], names(x)[k], names(x)[l], sep='_')
        row.names(cnt_4) <- NULL
        cat(i, j, k, l, '\n')
      }
    }
  }
}

xx <- cbind(cnt_6, cnt_7, cnt_1, cnt_2, cnt_3, cnt_4)
names(xx)
any(duplicated(names(xx)))

xx_train <- xx[!is.na(y), ]
range_x <- apply(xx_train, 2, function(z){max(z)-min(z)})
sum(range_x == 0)
x <- xx[, range_x > 0]

xt <- t(x)
xtu <- unique(xt)
dim(x)
dim(xt)
dim(xtu)
x <- t(xtu)

save(id, x, y, file='data\\xyid_cntratio_d4.RData')

# 衍生变量+gbm
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_8_cntratio_d4_gbm.txt')
rm(list=ls())
load('data\\xyid_cntratio_d4.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
base_learner <- f_gbm(x, y, id, ids, 
                      family='bernoulli', depth=20, shrk=0.01, 
                      ntree_max=15000, ntree_step=50, 
                      loss_threshold=0.0001, loss_cnt_threshold=5)
save(base_learner, file='data\\base_learner_cntratio_d4_gbm.RData')
sink()

# 衍生变量+随机森林
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_9_cntratio_d4_rf.txt')
rm(list=ls())
load('data\\xyid_cntratio_d4.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_rf.R')
require(randomForest)
# x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_randomforest(x, y, id, ids, ntree=1000, mtry=10)
save(base_learner, file='data\\base_learner_cntratio_d4_rf.RData')
sink()

# 衍生变量+随机森林
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_9_cntratio_d4_rf_2.txt')
rm(list=ls())
load('data\\xyid_cntratio_d4.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_rf.R')
require(randomForest)
# x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_randomforest(x, y, id, ids, ntree=1000, mtry=15)
save(base_learner, file='data\\base_learner_cntratio_d4_rf_2.RData')
sink()

# 衍生变量+glmnet
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_10_cntratio_d4_glmnet.txt')
rm(list=ls())
load('data\\xyid_cntratio_d4.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_glmnet.R')
require(glmnet)
x <- as.matrix(x)
x <- scale(x)
y <- as.factor(y)
base_learner <- f_glmnet(x, y, id, ids,family='binomial', alpha=0.5, standardize=F)
save(base_learner, file='data\\base_learner_cntratio_d4_glmnet.RData')
sink()

# 混合原始变量+衍生变量
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
rm(list=ls())
load('data\\xyid_raw.RData')
x1 <- x
y1 <- y
id1 <- id
load('data\\xyid_cntratio_d4.RData')
x2 <- x
y2 <- y
id2 <- id
identical(id1,id2)
identical(y1,y2)
id <- id1
y <- y1
x <- cbind(x1,x2)
save(id, x, y, file='data\\xyid_raw_cntratio_d4.RData')

# 混合变量+gbm
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_8_raw_cntratio_d4_gbm.txt')
rm(list=ls())
load('data\\xyid_raw_cntratio_d4.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
base_learner <- f_gbm(x, y, id, ids, 
                      family='bernoulli', depth=20, shrk=0.01, 
                      ntree_max=15000, ntree_step=50, 
                      loss_threshold=0.0001, loss_cnt_threshold=5)
save(base_learner, file='data\\base_learner_raw_cntratio_d4_gbm.RData')
sink()

# 混合变量+随机森林
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_9_raw_cntratio_d4_rf.txt')
rm(list=ls())
load('data\\xyid_raw_cntratio_d4.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_rf.R')
require(randomForest)
# x <- as.matrix(x)
# x <- scale(x)
y <- as.factor(y)
base_learner <- f_randomforest(x, y, id, ids, ntree=1000, mtry=10)
save(base_learner, file='data\\base_learner_raw_cntratio_d4_rf.RData')
sink()

# 混合变量+glmnet
options(stringsAsFactors=F)
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')
sink('log_amazon_10_raw_cntratio_d4_glmnet.txt')
rm(list=ls())
load('data\\xyid_raw_cntratio_d4.RData')
# ids for 5-fold cv
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_glmnet.R')
require(glmnet)
x <- as.matrix(x)
x <- scale(x)
y <- as.factor(y)
base_learner <- f_glmnet(x, y, id, ids,family='binomial', alpha=0.5, standardize=F)
save(base_learner, file='data\\base_learner_raw_cntratio_d4_glmnet.RData')
sink()


######################################
## 第二十二十一二十二次拟合 模型集成
######################################
options(stringsAsFactors=F)
rm(list=ls())
setwd('F:\\0_supstat\\kaggle_course\\1_amazon')

files <- list.files(path='data', pattern='^base_learner_', full.names=T)
files <- files[-c(11,18,19,5,8,4)]

for(i in seq_len(length(files))){
  load(files[i])
  res <- data.frame(id=base_learner$prediction$id, Action=base_learner$prediction$y_predict)
  summary(res)
  write.csv(res, file=paste('submission\\res20_base_',
                            gsub('.RData|data/base_learner_','',files[i]),'.csv',sep=''),
            row.names=F, quote=F)
}
require(verification)
f_auc <- function(a, p){
  roc.area(as.integer(as.character(a)), p)$A
}
for(i in seq_len(length(files))){
  load(files[i])
  cv <- base_learner$cv
  cat(f_auc(as.integer(as.character(cv$y_test)),cv$y_predict), '\n')
}

load(files[1])
d <- rbind(base_learner$cv[, c('id','y_test')],
           base_learner$prediction[, c('id','y_test')])
for(i in seq_len(length(files))){
  load(files[i])
  d0 <- rbind(base_learner$cv[, c('id','y_predict')],
              base_learner$prediction[, c('id','y_predict')])
  d <- merge(d, d0, by='id')
  names(d)[ncol(d)] <- paste('x', i, sep='_')
  cat(names(d), sep='\n')
}
rm(i,d0,base_learner)
d <- d[order(d$id), ]
id <- d$id
y <- d$y_test
x <- d[, !names(d) %in% c('id','y_test')]

# 简单地求均值
res20 <- data.frame(id=id[is.na(y)],Action=rowMeans(x)[is.na(y)])
summary(res20)
write.csv(res20, file='submission\\res20.csv', row.names=F, quote=F)
# 得分 0.90887 80/1687

# 用gbm做集成
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_gbm.R')
require(gbm)
stacking_learner <- f_gbm(x, y, id, ids, 
                          family='bernoulli', depth=6, shrk=0.01, 
                          ntree_max=15000, ntree_step=50, 
                          loss_threshold=0.0001, loss_cnt_threshold=5)
save(stacking_learner, file='data\\stacking_learner_gbm_21.RData')
res21 <- data.frame(id=stacking_learner$prediction$id, Action=stacking_learner$prediction$y_predict)
summary(res21)
write.csv(res21, file='submission\\res21.csv', row.names=F, quote=F)
# 得分 0.91128 67/1687

# 用随机森林做集成
ids <- id[!is.na(y)]
set.seed(114)
ids <- split(ids, sample(length(ids)) %% 5)
sapply(ids,length)
source('yibo_2_base_learner_rf.R')
require(randomForest)
y <- as.factor(y)
stacking_learner <- f_randomforest(x, y, id, ids, ntree=2000, mtry=4)
save(stacking_learner, file='data\\stacking_learner_rf_22.RData')
res22 <- data.frame(id=stacking_learner$prediction$id, Action=stacking_learner$prediction$y_predict)
summary(res22)
write.csv(res22, file='submission\\res22.csv', row.names=F, quote=F)
# 得分 0.90073 147/1687

# 简单地加权
w <- c(0.90116,0.76306,0.89706,0.89853,0.75786,0.89922,0.80318,
       0.86316,0.90303,0.76246,0.89817,0.86212,0.87282)
ww <- matrix(w,ncol=1)
res23 <- data.frame(id=id[is.na(y)],Action=(as.matrix(x) %*% ww)[is.na(y),1])
summary(res23)
write.csv(res23, file='submission\\res23.csv', row.names=F, quote=F)
# 0.90887 80/1687

