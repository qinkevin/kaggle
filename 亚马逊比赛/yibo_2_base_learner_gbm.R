

# gbm
# 不做交叉验证，或做五折交叉验证
# 主要可用于doc-term矩阵或带因子的数据框（水平数不多于1024），支持缺失值
# x可以是矩阵也可以是数据框
# y必须是数值型（gaussian），可以是0-1数值（bernoulli），此时预测的是取1的概率
# f_gbm(x, y, id, ids, 
#       family=c('gaussian', 'bernoulli'), depth=6, shrk=0.01, 
#       ntree_max=10000, ntree_step=100, 
#       loss_threshold=0.0001, loss_cnt_threshold=5)
f_gbm <- function(x, y, id, ids, 
                  family=c('gaussian', 'bernoulli'), depth=6, shrk=0.01, 
                  ntree_max=10000, ntree_step=100, 
                  loss_threshold=0.0001, loss_cnt_threshold=5){
  # colnames(x) <- paste('x', seq_len(ncol(x)), sep='_')
  if(family == 'bernoulli' & any(!unique(na.omit(y)) %in% c(0,1))){
    stop('if you want to do classify, you must have 0-1 response(y)')
  }
  if(family == 'bernoulli'){
    require(verification)
    f_loss <- function(a, p){
      1-roc.area(as.integer(as.character(a)), p)$A
    }
  }
  if(family == 'gaussian'){
    f_loss <- function(a, p){
      sqrt(mean((p-a)^2))
    }
  }
  
  require(gbm)
  out <- list()
  
  if(length(ids)==1){
    idx_train <- which(id %in% unlist(ids))
    idx_test <- which(!id %in% unlist(ids))
    set.seed(114)
    out$model <- gbm.fit(x=x[idx_train, ], y=y[idx_train], 
                         var.names=names(x), 
                         n.trees=ntree_step, 
                         interaction.depth=depth, 
                         shrinkage=shrk, 
                         distribution=family, 
                         verbose=F)
    ntree_best <- gbm.perf(out$model, method='OOB')
    cat('ntry_tree=', ntree_step, ' oob_ntree=', ntree_best, '\n', sep='')
    if(ntree_best >= ntree_step){
      for(ntree in seq(ntree_step*2, ntree_max, ntree_step)){
        set.seed(114)
        out$model <- gbm.more(out$model, ntree_step)
        ntree_best <- gbm.perf(out$model, method='OOB')
        cat('ntry_tree=', ntree, ' oob_ntree=', ntree_best, '\n', sep='')
        if(ntree_best < (ntree-10)) break
      }
    }
    cat('final_ntree=', ntree_best, '\n', sep='')
    out$prediction <- data.frame(stringsAsFactors=F, 
                                 id=id[idx_test], 
                                 y_predict=predict(out$model, as.data.frame(x[idx_test, ]), 
                                                   n.trees=ntree_best, type='response'), 
                                 y_test=y[idx_test])
    return(out)
  }
  if(length(ids)==5){
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
                             n.trees=ntree_step, 
                             interaction.depth=depth, 
                             shrinkage=shrk, 
                             distribution=family, 
                             verbose=F)
      cv_out[[i]] <- data.frame(stringsAsFactors=F, 
                                id=id[idx_test], 
                                y_predict=predict(models[[i]], as.data.frame(x[idx_test, ]), 
                                                  n.trees=ntree_step, type='response'), 
                                y_test=y[idx_test])
    }
    cv_out_tt <- do.call(rbind, cv_out)
    metric_best <- f_loss(cv_out_tt$y_test, cv_out_tt$y_predict)
    ntree_best <- ntree_step
    cat('initial_ntree=', ntree_step, ' - initial_loss=', round(metric_best, 6), '\n', sep='')
    
    # add at each step ntree_step trees, stop when no more gain in accuracy
    loss_cnt <- 0
    for(ntree in seq(ntree_step*2, ntree_max, ntree_step)){
      for(i in 1:5){
        cat(i, '\t')
        idx_test <- which(id %in% unlist(ids[i]))
        set.seed(114)
        models[[i]] <- gbm.more(models[[i]], ntree_step)
        cv_out[[i]][['y_predict']] <- predict(models[[i]], as.data.frame(x[idx_test, ]), 
                                              n.trees=ntree, type='response')
      }
      cv_out_tt <- do.call(rbind, cv_out)
      metric_now <- f_loss(cv_out_tt$y_test, cv_out_tt$y_predict)
      cat('now_ntree=', ntree, ' - min_loss=', round(metric_best, 6), 
          ' - now_loss=', round(metric_now, 6), '\n', sep='')
      
      if(metric_now >= (metric_best+loss_threshold)) break
      if(metric_now >= (metric_best-loss_threshold)){
        loss_cnt <- loss_cnt+1
      }
      if(loss_cnt >= loss_cnt_threshold) break
      if(metric_now < (metric_best-loss_threshold)){
        metric_best <- metric_now
        ntree_best <- ntree
        loss_cnt <- 0
      }
    }
    
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
    cat('final_ntree=', ntree_best, ' - cv_loss=', round(metric_cv, 6), '\n', sep='')
    idss <- unlist(ids)
    idss <- data.frame(id=idss, fold=rep(1:length(ids), times=sapply(ids,length)))
    cv_out_tt <- merge(cv_out_tt,idss,by='id',sort=F)
    metric_cv_fold <- by(cv_out_tt,cv_out_tt[['fold']],function(z){f_loss(z$y_test, z$y_predict)})
    cat('cv_loss_fold=', paste(round(metric_cv_fold, 6),collapse=' '), '\n', sep='')
    out$cv <- cv_out_tt
    
    # train of full training set
    idx_train <- which(id %in% unlist(ids))
    idx_test <- which(!id %in% unlist(ids))
    set.seed(114)
    out$model <- gbm.fit(x=x[idx_train, ], y=y[idx_train], 
                         var.names=names(x), 
                         n.trees=ntree_best, 
                         interaction.depth=depth, 
                         shrinkage=shrk, 
                         distribution=family, 
                         keep.data=F, 
                         verbose=F)
    out$prediction <- data.frame(stringsAsFactors=F, 
                                 id=id[idx_test], 
                                 y_predict=predict(out$model, as.data.frame(x[idx_test, ]), 
                                                   n.trees=ntree_best, type='response'), 
                                 y_test=y[idx_test])
    # print(summary(out$model, n.trees=ntree_best, plotit=F))
    
    return(out)
  }
}

