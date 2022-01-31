####### circular long_bpnr models for HPC ######
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

data_long <- read.csv("~/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/Result_Fiji_92.csv")
colnames(data_long)<-c("sampleID","side","layer","z","y","x", "domDir","cortexDepth","correction")
data <- data_long[!(data_long$layer=="L1" |
                      data_long$layer=="L6" |
                      data_long$sampleID!=12),]

data_ <- data
data_$sampleID <- as.factor(data_$sampleID)
attach(data_)
data$domDir <- NISTdegTOradian(data$domDir)
data$side <- as.factor(data$side)
data$layer <- as.factor(data$layer)
data$sampleID <- as.numeric(data$sampleID)
data$side <- as.numeric(data$side)
data$layer <- as.numeric(data$layer)
colnames(data)<-c("sampleID","side_num","layer_num","z","y","x", "domDir","cortexDepth","correction") 
data<-add_column(data, side, .before = "side_num")
data<-add_column(data, layer, .before = "layer_num")
#data$side_num <- as.factor(data$side_num)
#data$layer_num <- as.factor(data$layer_num)
detach(data_)
attach(data)


# Model comparison: bottom-up, 1. Model fit, 2. explained variance (part of random effect variances)
#Intercept-only model: only a fixed and random intercepts

fit.bpnr_1p = bpnr(pred.I = domDir ~ side_num,
                      data = data,
                      its = 20000, burn = 1500, n.lag = 5, seed = 101)
save(fit.bpnr_1p, file = "/ptmp/muellerg/fit.bpnr_1p.rda")
gc()

#include further higher-level factors e.g. between-subject factors
fit.bpnr_2p = bpnr(pred.I = domDir ~ side_num + layer_num,
                      data = data,
                      its = 20000, burn = 1500, n.lag = 5, seed = 101)
save(fit.bpnr_2p, file = "/ptmp/muellerg/fit.bpnr_2p.rda")
gc()

fit.bpnr_3p = bpnr(pred.I = domDir ~ side_num + layer_num + y,
                      data = dat,
                      its = 20000, burn = 1500, n.lag = 5, seed = 101)
save(fit.bpnr_3p, file = "/ptmp/muellerg/fit.bpnr_3p.rda")
gc()
