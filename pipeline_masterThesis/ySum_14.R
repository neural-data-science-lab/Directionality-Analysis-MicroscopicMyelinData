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
                      data_long$sampleID!=14),]
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
  SAM <- matrix(0, nrow = 1,ncol=4)
  bc <- matrix(0, nrow = 1,ncol=4)
  AS <- matrix(0, nrow = 1,ncol=4)
  SAM[1,1] = NISTradianTOdeg(mean_circ(fit$SAM))
  SAM[1,2] = NISTradianTOdeg(mode_est_circ(fit$SAM))
  SAM[1,3:4] = NISTradianTOdeg(hpd_est_circ(fit$SAM))
  bc[1,1] = NISTradianTOdeg(mean_circ(fit$b.c))
  bc[1,2] = NISTradianTOdeg(mode_est_circ(fit$b.c))
  bc[1,3:4] = NISTradianTOdeg(hpd_est_circ(fit$b.c))
  AS[1,1] = NISTradianTOdeg(mean_circ(fit$AS))
  AS[1,2] = NISTradianTOdeg(mode_est_circ(fit$AS))
  AS[1,3:4] = NISTradianTOdeg(hpd_est_circ(fit$AS))
  return(list(SAM, bc, AS))
}

Nsim = 25
its = 10000
seed = 2511

sample.SAM <- matrix(0, nrow = Nsim,ncol=4)
sample.bc <- matrix(0, nrow = Nsim,ncol=4)
sample.AS <- matrix(0, nrow = Nsim,ncol=4)

for (i in 1:Nsim){
  seed <- seed+i
  sample <- sample_n(data, 7500)
  new_params <- bpnr_func(sample, seed)
  
  sample.SAM[i,] <- new_params[1][[1]]
  sample.bc[i,] <- new_params[2][[1]]
  sample.AS[i,] <- new_params[3][[1]]
}

y = matrix(0, nrow = 3,ncol=4)
y[1,] = colMeans(sample.SAM)
y[2,] = colMeans(sample.bc)
y[3,] = colMeans(sample.AS)

s = matrix(0, nrow = 3,ncol=4)
s[1,] = apply(sample.SAM, 2, sd)
s[2,] = apply(sample.bc, 2, sd)
s[3,] = apply(sample.AS, 2, sd)

library(MASS)
write.matrix(y, file="/ptmp/muellerg/mean_y_14.csv")
write.matrix(s, file="/ptmp/muellerg/std_y_14.csv")
#write.matrix(sample.AS, file="/ptmp/muellerg/bpnr3p_AS_14.csv")

