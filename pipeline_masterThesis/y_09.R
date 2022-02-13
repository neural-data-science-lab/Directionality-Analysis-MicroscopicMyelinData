############## SAM, bc, AS ##############
library(dplyr)
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

data_long <- read.csv("/ptmp/muellerg/Result_Fiji_92.csv")
colnames(data_long)<-c("sampleID","side","layer","z","y","x", "domDir","cortexDepth","correction")
data <- data_long[!(data_long$layer=="L1" |
                      data_long$layer=="L6" |
                      data_long$sampleID!=09),]
attach(data)
data$domDir <- NISTdegTOradian(data$domDir)
data$side <- as.factor(data$side)
data$layer <- as.factor(data$layer)


' Bootstrapping ca 10-20 times 2500 random samples from data_long:
    - add mean, mode LB, UB to list for later statistics
    - do for all sampleID individually'

bpnr_func <- function(sample, seed){
  fit <- bpnr(pred.I = domDir ~ side + layer + y,
              data = sample,
              its = 10000, burn = 1000, n.lag = 3, seed = seed)
  SAM = fit$SAM
  bc = fit$b.c
  AS = fit$AS
  return(list(SAM, bc, AS))
}

Nsim = 25
its = 10000
seed = 2511

sample.SAM <- matrix(0, nrow = Nsim,ncol=its)
sample.bc <- matrix(0, nrow = Nsim,ncol=its)
sample.AS <- matrix(0, nrow = Nsim,ncol=its)

for (i in 1:Nsim){
  seed <- seed+i
  sample <- sample_n(data, 7500)
  new_params <- bpnr_func(sample, seed)
  
  sample.SAM[i,] <- new_params[1][[1]]
  sample.bc[i,] <- new_params[2][[1]]
  sample.AS[i,] <- new_params[3][[1]]
}

library(MASS)
write.matrix(sample.SAM, file="/ptmp/muellerg/bpnr3p_SAM_09.csv")
write.matrix(sample.bc, file="/ptmp/muellerg/bpnr3p_bc_09.csv")
write.matrix(sample.AS, file="/ptmp/muellerg/bpnr3p_AS_09.csv")

