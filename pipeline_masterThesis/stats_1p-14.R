' Statistics: bootstrapping from data, circular regression models on mice 
    individually, Sanity check: randomize data, Baysian plots, ... '

library(dplyr)
#library(ggplot2)
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

data_long <- read.csv("/ptmp/muellerg/Result_Fiji_92.csv")
colnames(data_long)<-c("sampleID","side","layer","z","y","x", "domDir","cortexDepth","correction")
data <- data_long[!(data_long$layer=="L1" |
                      data_long$layer=="L6" |
                      data_long$sampleID!=14),]
attach(data)
data$domDir <- NISTdegTOradian(data$domDir)
data$side <- as.factor(data$side)
data$layer <- as.factor(data$layer)



'data_long <- read.csv("~/Studium/MasterCMS/MasterThesis/DataPC/Result_92_1412141718/Result_Fiji_92.csv")
colnames(data_short)<-c("sampleID","side","layer","y","domDir","count_per_av")
dat <- data_short[!(data_short$layer=="L1" |
                      data_short$layer=="L6" |
                      data_short$sampleID!=14),]
sample <- sample_n(data, 7500)

ggplot(data, aes(x=domDir)) + geom_histogram(bins = 45)
#ggplot(dat, aes(x=domDir)) + geom_histogram(bins = 45)
ggplot(sample, aes(x=domDir)) + geom_histogram(bins = 45)
'



' Bootstrapping ca 10-20 times 2500 random samples from data_long:
    - add mean, mode LB, UB to list for later statistics
    - do for all sampleID individually'

bpnr_func <- function(sample, seed){
  fit <- bpnr(pred.I = domDir ~ side,
                     data = sample,
                     its = 10000, burn = 1000, n.lag = 3, seed = seed)
  Intercept <- NISTradianTOdeg(fit$circ.coef.means)[1,]
  sider <- NISTradianTOdeg(fit$circ.coef.means)[2,]
  beta1_1 <- fit$beta1[,1] # Intercept
  beta1_2 <- fit$beta1[,2] #sider
  beta2_1 <- fit$beta2[,1]
  beta2_2 <- fit$beta2[,2]
  model_fit <- fit(fit)[,1]
  return(list(Intercept, sider, beta1_1, beta1_2, beta2_1, beta2_2, model_fit))
}

Nsim = 25
its = 10000
seed = 2511
sample.Intercept <- matrix(0, nrow = Nsim,ncol=5)
sample.sider <- matrix(0, nrow = Nsim,ncol=5)
sample.beta1_1 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta1_2 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta2_1 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta2_2 <- matrix(0, nrow = Nsim,ncol=its)
sample.fit <- matrix(0, nrow = Nsim,ncol=5)
for (i in 1:Nsim){
  seed <- seed+i
  sample <- sample_n(data, 7500)
  new_params <- bpnr_func(sample, seed)
  sample.Intercept[i,] <- new_params[1][[1]]
  sample.sider[i,] <- new_params[2][[1]]
  sample.beta1_1[i,] <- new_params[3][[1]]
  sample.beta1_2[i,] <- new_params[4][[1]]
  sample.beta2_1[i,] <- new_params[5][[1]]
  sample.beta2_2[i,] <- new_params[6][[1]]
  sample.fit[i,] <- new_params[7][[1]]
}

library(MASS)
write.matrix(sample.Intercept, file="/ptmp/muellerg/bpnr1p_Intercept_14.csv")
write.matrix(sample.sider, file="/ptmp/muellerg/bpnr1p_side2_14.csv")
write.matrix(sample.fit, file="/ptmp/muellerg/bpnr1p_fit_14.csv")
write.matrix(sample.beta1_1, file="/ptmp/muellerg/bpnr1p_beta1_1_14.csv")
write.matrix(sample.beta1_2, file="/ptmp/muellerg/bpnr1p_beta1_2_14.csv")
write.matrix(sample.beta2_1, file="/ptmp/muellerg/bpnr1p_beta2_1_14.csv")
write.matrix(sample.beta2_2, file="/ptmp/muellerg/bpnr1p_beta2_2_14.csv")



' Randomize data as a Sanity check'




