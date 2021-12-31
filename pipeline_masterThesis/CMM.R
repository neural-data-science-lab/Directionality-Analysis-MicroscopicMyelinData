####### circular mixed models for HPC ######
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

#### read in data
data <- read.csv("/ptmp/muellerg/Result_Fiji_92.csv")
colnames(data)<-c("sampleID","side","layer","z","y","x", "domDir","cortexDepth","correction") 
dataL1<-data[!(data$layer=="L1"),]
data$sampleID <- as.factor(data$sampleID)

data_L1<-data[!(data$layer=="L1"),]
data_L1L6<-data[!(data$layer=="L1" | data$layer=="L6"),]
attach(data_L1)

dataL1$domDir <- NISTdegTOradian(dataL1$domDir)
dataL1$side <- as.factor(dataL1$side)
dataL1$layer <- as.factor(dataL1$layer)
dataL1$sampleID <- as.numeric(dataL1$sampleID)
dataL1$side <- as.numeric(dataL1$side)
dataL1$layer <- as.numeric(dataL1$layer)
colnames(dataL1)<-c("sampleID","side_num","layer_num","z","y","x", "domDir","cortexDepth","correction") 
dataL1<-add_column(dataL1, side, .before = "side_num")
dataL1<-add_column(dataL1, layer, .before = "layer_num")
dataL1$side_num <- as.factor(dataL1$side_num)
dataL1$layer_num <- as.factor(dataL1$layer_num)
detach(data_L1)
attach(dataL1)

# Model comparison: bottom-up, 1. Model fit, 2. explained variance (part of random effect variances)
#Intercept-only model: only a fixed and random intercepts
fit.dataL1_IO = bpnme(pred.I = domDir ~ (1|sampleID),
                      data = dataL1,
                      its = 10000, burn = 1000, n.lag = 3, seed = 101)
save(fit.dataL1_IO, file = "/ptmp/muellerg/fit.dataL1_IO.rda")

#include fixed-effects, simplest within-subject factors
fit.dataL1_1p = bpnme(pred.I = domDir ~ side_num + (1|sampleID),
                      data = dataL1,
                      its = 10000, burn = 1000, n.lag = 3, seed = 101)
save(fit.dataL1_1p, file = "/ptmp/muellerg/fit.dataL1_1p.rda")

#include further higher-level factors e.g. between-subject factors
fit.dataL1_2p = bpnme(pred.I = domDir ~ side_num + layer_num + (1|sampleID),
                      data = dataL1,
                      its = 10000, burn = 1000, n.lag = 3, seed = 101)
save(fit.dataL1_2p, file = "/ptmp/muellerg/fit.dataL1_2p.rda")

fit.dataL1_3p = bpnme(pred.I = domDir ~ side_num + layer_num y + (1|sampleID),
                      data = dataL1,
                      its = 10000, burn = 1000, n.lag = 3, seed = 101)
save(fit.dataL1_3p, file = "/ptmp/muellerg/fit.dataL1_3p.rda")

#adding random slopes for 1st level predictors or cross-level interactions



