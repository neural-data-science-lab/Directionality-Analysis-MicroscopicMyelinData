' Statistics: bootstrapping from data, circular regression models on mice 
    individually, Sanity check: randomize data, Baysian plots, ... '

library(dplyr)
#library(ggplot2)
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

data_long <- read.csv("/ptmp/muellerg/Result_Fiji_92-VCx.csv")
colnames(data_long)<-c("sampleID","side","layer","z","y","x", "domDir","cortexDepth","correction")
data <- data_long[!(data_long$layer=="L1" |
                      data_long$layer=="L6" |
                      data_long$sampleID!=14),]
attach(data)
data$domDir <- NISTdegTOradian(data$domDir)
data$side <- as.factor(data$side)
data$layer <- as.factor(data$layer)


' Bootstrapping ca 10-20 times 2500 random samples from data_long:
    - add mean, mode LB, UB to list for later statistics
    - do for all sampleID individually'

bpnr_func <- function(sample, seed){
  fit <- bpnr(pred.I = domDir ~ side + layer,
              data = sample,
              its = 10000, burn = 1000, n.lag = 3, seed = seed)
  Intercept <- NISTradianTOdeg(fit$circ.coef.means)[1,]
  sider <- NISTradianTOdeg(fit$circ.coef.means)[2,]
  layerL4 <- NISTradianTOdeg(fit$circ.coef.means)[3,]
  layerL5 <- NISTradianTOdeg(fit$circ.coef.means)[4,]
  siderlayerL4 <- NISTradianTOdeg(fit$circ.coef.means)[5,]
  siderlayerL5 <- NISTradianTOdeg(fit$circ.coef.means)[6,]
  layerL4layerL5 <- NISTradianTOdeg(fit$circ.coef.means)[7,]
  beta1_1 <- fit$beta1[,1] # Intercept
  beta1_2 <- fit$beta1[,2] #sider
  beta1_3 <- fit$beta1[,3] #layerL4
  beta1_4 <- fit$beta1[,4] #layerL5
  beta2_1 <- fit$beta2[,1]
  beta2_2 <- fit$beta2[,2]
  beta2_3 <- fit$beta2[,3]
  beta2_4 <- fit$beta2[,4]
  model_fit <- fit(fit)[,1]
  return(list(Intercept, sider, layerL4, layerL5, siderlayerL4, siderlayerL5, 
              layerL4layerL5, beta1_1, beta1_2, beta1_3, beta1_4, beta2_1, 
              beta2_2, beta2_3, beta2_4, model_fit))
}

Nsim = 25
its = 10000
seed = 2511
sample.Intercept <- matrix(0, nrow = Nsim,ncol=5)
sample.sider <- matrix(0, nrow = Nsim,ncol=5)
sample.layerL4 <- matrix(0, nrow = Nsim,ncol=5)
sample.layerL5 <- matrix(0, nrow = Nsim,ncol=5)
sample.siderlayerL4 <- matrix(0, nrow = Nsim,ncol=5)
sample.siderlayerL5 <- matrix(0, nrow = Nsim,ncol=5)
sample.layerL4layerL5 <- matrix(0, nrow = Nsim,ncol=5)
sample.beta1_1 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta1_2 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta1_3 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta1_4 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta2_1 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta2_2 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta2_3 <- matrix(0, nrow = Nsim,ncol=its)
sample.beta2_4 <- matrix(0, nrow = Nsim,ncol=its)
sample.fit <- matrix(0, nrow = Nsim,ncol=5)
for (i in 1:Nsim){
  seed <- seed+i
  sample <- sample_n(data, 7500)
  new_params <- bpnr_func(sample, seed)
  sample.Intercept[i,] <- new_params[1][[1]]
  sample.sider[i,] <- new_params[2][[1]]
  sample.layerL4[i,] <- new_params[3][[1]]
  sample.layerL5[i,] <- new_params[4][[1]]
  sample.siderlayerL4[i,] <- new_params[5][[1]]
  sample.siderlayerL5[i,] <- new_params[6][[1]]
  sample.layerL4layerL5[i,] <- new_params[7][[1]]
  sample.beta1_1[i,] <- new_params[8][[1]]
  sample.beta1_2[i,] <- new_params[9][[1]]
  sample.beta1_3[i,] <- new_params[10][[1]]
  sample.beta1_4[i,] <- new_params[11][[1]]
  sample.beta2_1[i,] <- new_params[12][[1]]
  sample.beta2_2[i,] <- new_params[13][[1]]
  sample.beta2_3[i,] <- new_params[14][[1]]
  sample.beta2_4[i,] <- new_params[15][[1]]
  sample.fit[i,] <- new_params[16][[1]]
}

library(MASS)
write.matrix(sample.Intercept, file="/ptmp/muellerg/bpnr2p_Intercept_VCx.csv")
write.matrix(sample.sider, file="/ptmp/muellerg/bpnr2p_sider_VCx.csv")
write.matrix(sample.layerL4, file="/ptmp/muellerg/bpnr2p_layerL4_VCx.csv")
write.matrix(sample.layerL5, file="/ptmp/muellerg/bpnr2p_layerL5_VCx.csv")
write.matrix(sample.siderlayerL4, file="/ptmp/muellerg/bpnr2p_siderlayerL4_VCx.csv")
write.matrix(sample.siderlayerL5, file="/ptmp/muellerg/bpnr2p_siderlayerL5_VCx.csv")
write.matrix(sample.layerL4layerL5, file="/ptmp/muellerg/bpnr2p_layerL4layerL5_VCx.csv")
write.matrix(sample.fit, file="/ptmp/muellerg/bpnr2p_fit_VCx.csv")
write.matrix(sample.beta1_1, file="/ptmp/muellerg/bpnr2p_beta1_1_VCx.csv")
write.matrix(sample.beta1_2, file="/ptmp/muellerg/bpnr2p_beta1_2_VCx.csv")
write.matrix(sample.beta1_3, file="/ptmp/muellerg/bpnr2p_beta1_3_VCx.csv")
write.matrix(sample.beta1_4, file="/ptmp/muellerg/bpnr2p_beta1_4_VCx.csv")
write.matrix(sample.beta2_1, file="/ptmp/muellerg/bpnr2p_beta2_1_VCx.csv")
write.matrix(sample.beta2_2, file="/ptmp/muellerg/bpnr2p_beta2_2_VCx.csv")
write.matrix(sample.beta2_3, file="/ptmp/muellerg/bpnr2p_beta2_3_VCx.csv")
write.matrix(sample.beta2_4, file="/ptmp/muellerg/bpnr2p_beta2_4_VCx.csv")







