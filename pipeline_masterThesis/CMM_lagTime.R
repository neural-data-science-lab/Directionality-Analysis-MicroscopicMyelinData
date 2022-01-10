###### test for lag_time
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

data <- read.csv("/ptmp/muellerg/Result_Fiji_92_mode-short.csv")
colnames(data)<-c("sampleID","side","layer","y","domDir","count_per_av")
dataL1_18<-data[!(data$layer=="L1" | data$sampleID==18),]

data$sampleID <- as.factor(data$sampleID)
data_L1_18<-data[!(data$layer=="L1" | data$sampleID==18),]
attach(data_L1_18)

dataL1_18$domDir <- NISTdegTOradian(dataL1_18$domDir)
dataL1_18$side <- as.factor(dataL1_18$side)
dataL1_18$layer <- as.factor(dataL1_18$layer)
dataL1_18$sampleID <- as.numeric(dataL1_18$sampleID)
dataL1_18$side <- as.numeric(dataL1_18$side)
dataL1_18$layer <- as.numeric(dataL1_18$layer)
colnames(dataL1_18)<-c("sampleID","side_num","layer_num","y", "domDir","count_per_av") 
dataL1_18<-add_column(dataL1_18, side, .before = "side_num")
dataL1_18<-add_column(dataL1_18, layer, .before = "layer_num")
#dataL1_18$side_num <- as.factor(dataL1_18$side_num)
#dataL1_18$layer_num <- as.factor(dataL1_18$layer_num)
detach(data_L1_18)
attach(dataL1_18)

'fit.mode_lag1 = bpnme(pred.I = domDir ~ side_num + (1|sampleID),
                          data = dataL1_18,
                          its = 20000, burn = 3500, n.lag = 1, seed = 101)
save(fit.mode_lag1, file = "/ptmp/muellerg/fit.mode_lag1.rda")
gc()

#include fixed-effects, simplest within-subject factors
fit.mode_lag2 = bpnme(pred.I = domDir ~ side_num + (1|sampleID),
                          data = dataL1_18,
                          its = 20000, burn =3500, n.lag = 2, seed = 101)
save(fit.mode_lag2, file = "/ptmp/muellerg/fit.mode_lag2.rda")
gc()

fit.mode_lag3 = bpnme(pred.I = domDir ~ side_num + (1|sampleID),
                          data = dataL1_18,
                          its = 20000, burn = 3500, n.lag = 3, seed = 101)
save(fit.mode_lag3, file = "/ptmp/muellerg/fit.mode_lag3.rda")
gc()

fit.mode_lag5 = bpnme(pred.I = domDir ~ side_num + (1|sampleID),
                          data = dataL1_18,
                          its = 20000, burn = 3500, n.lag = 5, seed = 101)
save(fit.mode_lag5, file = "/ptmp/muellerg/fit.mode_lag5.rda")
gc()'

fit.mode_lag7 = bpnme(pred.I = domDir ~ side_num + (1|sampleID),
                          data = dataL1_18,
                          its = 20000, burn = 3500, n.lag = 7, seed = 101)
save(fit.mode_lag7, file = "/ptmp/muellerg/fit.mode_lag7.rda")