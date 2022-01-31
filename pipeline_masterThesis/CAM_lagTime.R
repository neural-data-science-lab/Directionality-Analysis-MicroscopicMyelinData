###### test for lag_time
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)
library(dplyr)

data_long <- read.csv("/ptmp/muellerg/Result_Fiji_92.csv")
colnames(data_long)<-c("sampleID","side","layer","z","y","x", "domDir","cortexDepth","correction")
colnames(data_long)<-c("sampleID","side","layer","z","y","x", "domDir","cortexDepth","correction")
data <- data_long[!(data_long$layer=="L1" |
                      data_long$layer=="L6" |
                      data_long$sampleID!=17),]
attach(data)
data$domDir <- NISTdegTOradian(data$domDir)
data$side <- as.factor(data$side)
data$layer <- as.factor(data$layer)

data <- sample_n(data, 5000)

fit.bpnr_lag1 = bpnr(pred.I = domDir ~ side,
                      data = data,
                      its = 10000, burn = 1000, n.lag = 1, seed = 101)
save(fit.bpnr_lag1, file = "/ptmp/muellerg/fit.bpnr_lag1_1p-17.rda")
gc()

#include fixed-effects, simplest within-subject factors
fit.bpnr_lag2 = bpnr(pred.I = domDir ~ side,
                      data = data,
                      its = 10000, burn =1000, n.lag = 2, seed = 101)
save(fit.bpnr_lag2, file = "/ptmp/muellerg/fit.bpnr_lag2_1p-17.rda")
gc()

fit.bpnr_lag3 = bpnr(pred.I = domDir ~ side,
                      data = data,
                      its = 10000, burn = 1000, n.lag = 3, seed = 101)
save(fit.bpnr_lag3, file = "/ptmp/muellerg/fit.bpnr_lag3_1p-17.rda")
gc()

fit.bpnr_lag5 = bpnr(pred.I = domDir ~ side,
                      data = data,
                      its = 10000, burn = 1000, n.lag = 5, seed = 101)
save(fit.bpnr_lag5, file = "/ptmp/muellerg/fit.bpnr_lag5_1p-17.rda")
gc()

