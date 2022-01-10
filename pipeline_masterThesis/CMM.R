####### circular mixed models for HPC ######
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

#### read in data
data <- read.csv("/ptmp/muellerg/Result_Fiji_92_mode-short.csv")
colnames(data)<-c("sampleID","side","layer","y","domDir","count_per_av")
dat<-data[!(data$layer=="L1"),] 

data$sampleID <- as.factor(data$sampleID)
dat_<-data[!(data$layer=="L1"),] 

##### reorganize data for bpnme
#dataL1
attach(dat_)
dat$domDir <- NISTdegTOradian(dat$domDir)
dat$side <- as.factor(dat$side)
dat$layer <- as.factor(dat$layer)
dat$sampleID <- as.numeric(dat$sampleID)
dat$side <- as.numeric(dat$side)
dat$layer <- as.numeric(dat$layer)
colnames(dat)<-c("sampleID","side_num","layer_num","y", "domDir","count_per_av") 
dat<-add_column(dat, side, .before = "side_num")
dat<-add_column(dat, layer, .before = "layer_num")
dat$side_num <- as.factor(dat$side_num)
dat$layer_num <- as.factor(dat$layer_num)
detach(dat_)
attach(dat)


# Model comparison: bottom-up, 1. Model fit, 2. explained variance (part of random effect variances)
#Intercept-only model: only a fixed and random intercepts
fit.mode_bpnme_IO = bpnme(pred.I = domDir ~ (1|sampleID),
                      data = dat,
                      its = 20000, burn = 3500, n.lag = 1, seed = 101)
save(fit.mode_bpnme_IO, file = "/ptmp/muellerg/fit.mode_bpnme_IOAll.rda")
gc()

#include fixed-effects, simplest within-subject factors
fit.mode_bpnme_1p = bpnme(pred.I = domDir ~ side_num + (1|sampleID),
                          data = dat,
                          its = 20000, burn = 3500, n.lag = 1, seed = 101)
save(fit.mode_bpnme_1p, file = "/ptmp/muellerg/fit.mode_bpnme_1pAll.rda")
gc()

#include further higher-level factors e.g. between-subject factors
fit.mode_bpnme_2p = bpnme(pred.I = domDir ~ side_num + layer_num + (1|sampleID),
                         data = dat,
                         its = 20000, burn = 3500, n.lag = 1, seed = 101)
save(fit.mode_bpnme_2p, file = "/ptmp/muellerg/fit.mode_bpnme_2pAll.rda")
gc()

fit.mode_bpnme_3p = bpnme(pred.I = domDir ~ side_num + layer_num + y + (1|sampleID),
                         data = dat,
                         its = 20000, burn = 3500, n.lag = 1, seed = 101)
save(fit.mode_bpnme_3p, file = "/ptmp/muellerg/fit.mode_bpnme_3pAll.rda")
gc()

