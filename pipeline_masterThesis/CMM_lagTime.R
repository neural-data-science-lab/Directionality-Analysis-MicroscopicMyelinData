###### test for lag_time
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

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
#dat$side_num <- as.factor(dat$side_num)
#dat$layer_num <- as.factor(dat$layer_num)
detach(dat_)
attach(dat)

fit.bpnme_lag1 = bpnme(pred.I = domDir ~ side_num + layer_num + y + (1|sampleID),
                          data = dat,
                          its = 20000, burn = 1500, n.lag = 1, seed = 101)
save(fit.bpnme_lag1, file = "/ptmp/muellerg/fit.bpnme_lag1.rda")
gc()

fit.bpnme_lag2 = bpnme(pred.I = domDir ~ side_num + layer_num + y + (1|sampleID),
                       data = dat,
                       its = 20000, burn = 1500, n.lag = 2, seed = 101)
save(fit.bpnme_lag2, file = "/ptmp/muellerg/fit.bpnme_lag2.rda")
gc()

fit.bpnme_lag3 = bpnme(pred.I = domDir ~ side_num + layer_num + y + (1|sampleID),
                       data = dat,
                       its = 20000, burn = 1500, n.lag = 3, seed = 101)
save(fit.bpnme_lag3, file = "/ptmp/muellerg/fit.bpnme_lag3.rda")
gc()

fit.bpnme_lag5 = bpnme(pred.I = domDir ~ side_num + layer_num + y + (1|sampleID),
                       data = dat,
                       its = 20000, burn = 1500, n.lag = 5, seed = 101)
save(fit.bpnme_lag5, file = "/ptmp/muellerg/fit.bpnme_lag5.rda")
gc()

fit.bpnme_lag7 = bpnme(pred.I = domDir ~ side_num + layer_num + y + (1|sampleID),
                       data = dat,
                       its = 20000, burn = 1500, n.lag = 7, seed = 101)
save(fit.bpnme_lag7, file = "/ptmp/muellerg/fit.bpnme_lag7.rda")
gc()