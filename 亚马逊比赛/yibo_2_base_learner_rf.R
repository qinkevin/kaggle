

# randomforest
# 目前暂时主要可用于doc-term矩阵，不支持缺失值，不能用于dummy矩阵
# x可以是矩阵或数据框
# y可以是数值型，也可以是0-1因子型，此时预测的是取1的概率
# f_randomforest(x, y, id, ids, ntree=1000, mtry=3)
f_randomforest <- function(x, y, id, ids, 
                           ntree=1000, mtry=3){
  # colnames(x) <- paste('x', seq_len(ncol(x)), sep='_')
  if(is.factor(y) & nlevels(y) != 2){
    stop('if you want to do classify, you must have 0-1 response(y)')
  }
  if(is.factor(y)){
    require(verification)
    f_loss <- function(a, p){
      1-roc.area(as.integer(as.character(a)), p)$A
    }
  }
  if(is.numeric(y)){
    f_loss <- function(a, p){
      sqrt(mean((p-a)^2))
    }
  }
  
  require(randomForest)
  out <- list()
  tmp <- list()
  
  if(length(ids)==1){
    idx_train <- which(id %in% unlist(ids))
    idx_test <- which(!id %in% unlist(ids))
    set.seed(114)
    out$model <- randomForest(x=x[idx_train, ], y=y[idx_train], 
                              ntree=ntree, mtry=mtry)
    if(!is.factor(y)){
      y_predict <- predict(out$model, x[idx_test, ], type='response')
    }
    if(is.factor(y)){
      y_predict <- predict(out$model, x[idx_test, ], type='prob')
      y_predict <- y_predict[, colnames(y_predict) == '1']
    }
    out$prediction <- data.frame(stringsAsFactors=F, 
                                 id=id[idx_test], 
                                 y_predict=y_predict, 
                                 y_test=y[idx_test])
    return(out)
  }
  if(length(ids)==5){
    for(i in 1:6){
      if(i <= 5){
        idx_train <- which(id %in% unlist(ids[-i]))
        idx_test <- which(id %in% unlist(ids[i]))
      } else{
        idx_train <- which(id %in% unlist(ids))
        idx_test <- which(!id %in% unlist(ids))
      }
      set.seed(114)
      out$model <- randomForest(x=x[idx_train, ], y=y[idx_train], 
                                ntree=ntree, mtry=mtry)
      if(!is.factor(y)){
        y_predict <- predict(out$model, x[idx_test, ], type='response')
      }
      if(is.factor(y)){
        y_predict <- predict(out$model, x[idx_test, ], type='prob')
        y_predict <- y_predict[, colnames(y_predict) == '1']
      }
      tmp[[i]] <- data.frame(stringsAsFactors=F, 
                             id=id[idx_test], 
                             y_predict=y_predict, 
                             y_test=y[idx_test])
      cat('randomforest', 6-i, '\n')
    }
    
    out$prediction <- tmp[[6]]
    out$cv <- do.call(rbind, tmp[1:5])
    metric_cv <- f_loss(out$cv$y_test, out$cv$y_predict)
    cat(paste('cv_loss=', round(metric_cv, 6), sep=''), '\n')
    idss <- unlist(ids)
    idss <- data.frame(id=idss, fold=rep(1:length(ids), times=sapply(ids,length)))
    out$cv <- merge(out$cv,idss,by='id',sort=F)
    metric_cv_fold <- by(out$cv,out$cv[['fold']],function(z){f_loss(z$y_test, z$y_predict)})
    cat('cv_loss_fold=', paste(round(metric_cv_fold, 6),collapse=' '), '\n', sep='')
    
    return(out)
  }
}

