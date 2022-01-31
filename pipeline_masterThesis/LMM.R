####### LMM #######

library("lme4")
library("ggplot2")
library("lmerTest")
library("nlme")
library(dplyr)

# 1. read data and provide an overview plot
data <- read.csv("~/Studium/MasterCMS/MasterThesis/DataPC/Result_92_12141718/Result_Fiji_92_mode-short.csv")
#colnames(data)<-c("sampleID","side","layer","z","y","x", "domDir","cortexDepth","correction") 
colnames(data)<-c("sampleID","side","layer","y","domDir","count_per_av") 
dataL1_18<-data[!(data$layer=="L1" | data$sampleID==18),]
dataL1_18<-add_column(dataL1_18, dataL1_18$side, .after = "side")
dataL1_18<-add_column(dataL1_18, dataL1_18$layer, .after = "layer")
dataL1_18$`dataL1_18$side` <- as.factor(dataL1_18$`dataL1_18$side`)
dataL1_18$`dataL1_18$layer` <- as.factor(dataL1_18$`dataL1_18$layer`)
dataL1_18$`dataL1_18$side` <- as.numeric(dataL1_18$`dataL1_18$side`)
dataL1_18$`dataL1_18$layer` <- as.numeric(dataL1_18$`dataL1_18$layer`)
colnames(dataL1_18)<-c("sampleID","side","side_num", "layer","layer_num","y","domDir","count_per_av")
attach(dataL1_18)

#plots
ggplot(dataL1_18, aes(x=domDir)) + geom_density() + scale_color_grey() + theme_classic()
ggplot(dataL1_18[!(side_num=="r"),], aes(x=domDir)) + geom_histogram(aes(y=..density..), colour="black", fill="white", bins = 30) +
  geom_density(alpha=.2, fill="#FF6666") 
plot(domDir ~ side_num, data = dataL1_18)
ggplot(dataL1_18,aes(x=side_num,y=domDir)) + geom_point() + geom_smooth(method = "lm")

# 2. simple linear regression
fit_lm0 = lm(domDir~side, data = dataL1)
summary(fit_lm0)
anova(fit_lm0)
'ggplot(dataL1_18,aes(x=side_num,y=domDir)) + geom_point() + geom_smooth(method = "lm")
abline(lm(domDir ~ side_num, data = dataL1_18), col = "blue")
conf_interval <- predict(lm(domDir ~ side_num, data = dataL1_18), 
                         newdata = data.frame(side = seq(0, 9, by = 0.1)), 
                         interval = "confidence", level = 0.95)
lines(seq(0, 9, by = 0.1), conf_interval[,2], col = "blue", lty = 2)
lines(seq(0, 9, by = 0.1), conf_interval[,3], col = "blue", lty = 2)

ggplot(dataL1_18,aes(x=side_num,y=domDir)) + geom_smooth(method = "lm",level = 0.95) + 
  geom_point() + facet_wrap(~sampleID + side_num, nrow = 4, ncol = 4)'

fit_lm1 = lm(domDir~side_num + layer_num, data = dataL1_18)
summary(fit_lm1)
anova(fit_lm1)


# 3. linear mixed models
'domDir is predicted via side (fixed effects) and '
fit_lmer0 = lmer(domDir ~ side_num + (1 | sampleID), REML = FALSE, dataL1)
fit_lmer1 = lme(domDir~side, data=dataL1_18, random = ~1 | sampleID, method = "ML")
summary(fit_lmer0)
anova(fit_lmer0)
summary(fit_lmer1)
anova(fit_lmer1)

plot(fit_lmer1, resid(.,type="p")~fitted(.))
qqnorm(fit_lmer1, ~resid(.))
qqnorm(resid(fit_lmer1))
qqline(resid(fit_lmer1))
qqnorm(fit_lmer1, ~ranef(.),id=0.1,cex=0.7)

qqnorm(resid(fit_lmer0))
qqline(resid(fit_lmer0))
plot(fitted(fit_lmer0),resid(fit_lmer0), xlab="Fitted values", ylab="Residuals")
abline(0,0,col="red")

fit_lmer2 = lmer(domDir ~ side + (side | sampleID), REML = FALSE, dataL1_18)
summary(fit_lmer2)
anova(fit_lmer2)
qqnorm(resid(fit_lmer2))
qqline(resid(fit_lmer2))

fit_lmer3 = lmer(domDir ~ side + (1 | sampleID) + (1 | layer) + (1 | y), REML = FALSE, data_L1)
summary(fit_lmer3)
anova(fit_lmer3)
qqnorm(resid(fit_lmer3))
qqline(resid(fit_lmer3))
plot(fitted(fit_lmer3),resid(fit_lmer3), xlab="Fitted values", ylab="Residuals")
abline(0,0,col="red")


'intercept only model: domDir is predicted by intercept and a random error term for the intercept
Null-model'
fit_lmer5 <- lmer(domDir ~ 1 + (1 | sampleID), dataL1_18, REML = FALSE)
summary(fit_lmer5)
anova(fit_lmer5)
#qqnorm(resid(fit_lmer5))
#qqline(resid(fit_lmer5))


#4. overall anova
anova(fit_lmer5, fit_lmer3,fit_lmer2,fit_lmer1,fit_lmer0,fit_lm1,fit_lm0)




#5. circular data analysis
library(bpnreg)
library(circular)
library(NISTunits)
library(tibble)

# test statistics
fit(fit.Maps)
# coefficients of the fixed effects model
#categorical variable type: posterior estimates of the circular mean in deg -> 
#check whether there is an influence on outcome
NISTradianTOdeg(fit.Maps$circ.coef.means)
# Variance of the random effects in deg 
NISTradianTOdeg(fit.Maps$circ.res.varrand)



dataL1$domDir <- NISTdegTOradian(dataL1$domDir)
dataL1$side <- as.factor(dataL1$side)
dataL1$layer <- as.factor(dataL1$layer)
dataL1$sampleID <- as.numeric(dataL1$sampleID)
dataL1$side <- as.numeric(dataL1$side)
dataL1$layer <- as.numeric(dataL1$layer)
#colnames(dataL1)<-c("sampleID","side_num","layer_num","z","y","x", "domDir","cortexDepth","correction") 
colnames(dataL1)<-c("sampleID","side_num","layer_num","y", "domDir","count_per_av") 
dataL1<-add_column(dataL1, side, .before = "side_num")
dataL1<-add_column(dataL1, layer, .before = "layer_num")
dataL1$side_num <- as.factor(dataL1$side_num)
dataL1$layer_num <- as.factor(dataL1$layer_num)

detach(data_L1)
attach(dataL1)
subdata = dataL1[sample(nrow(dataL1), 200), ]
attach(subdata)

# Model comparison: bottom-up, 1. Model fit, 2. explained variance (part of random effect variances)
#Intercept-only model: only a fixed and random intercepts
fit.dataL1_IO = bpnme(pred.I = domDir ~ (1|sampleID),
                       data = dataL1,
                      its = 1000, burn = 100, n.lag = 3, seed = 101)

#include fixed-effects, simplest within-subject factors
fit.dataL1_1p = bpnme(pred.I = domDir ~ side_num + (1|sampleID),
                      data = dataL1,
                      its = 1000, burn = 100, n.lag = 3, seed = 101)

#include further higher-level factors e.g. between-subject factors
fit.dataL1_2p = bpnme(pred.I = domDir ~ side_num + layer_num + (1|sampleID),
                      data = dataL1,
                      its = 1000, burn = 100, n.lag = 3, seed = 101)

fit.dataL1_3p = bpnme(pred.I = domDir ~ side_num + layer_num + y + (1|sampleID),
                      data = dataL1,
                      its = 1000, burn = 100, n.lag = 3, seed = 101)

#adding random slopes for 1st level predictors or cross-level interactions


# read in files 
load(file = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/fit.bpnr_lag1-12.rda")
load(file = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_12141718/fit.mode_bpnr_2p-18.rda")
load(file = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_12141718/fit.mode_bpnr_3p-18.rda")
load(file = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_12141718/fit.mode_lag5.rda")
load(file = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_12141718/fit.mode_lag7.rda")
# test statistics
fit(fit.dataL1_18_lag2)
traceplot(fit.dataL1_18_lag1)
# coefficients of the fixed effects model
#categorical variable type: posterior estimates of the circular mean in deg -> 
#check whether there is an influence on outcome
NISTradianTOdeg(fit.dataL1_1pt$circ.coef.means)
# Variance of the random effects in deg 
NISTradianTOdeg(fit.dataL1_1pt$circ.res.varrand)


############## try circular ANOVA - dat
left <- data[!(data$layer=="L1" | data$sampleID==18),]$domDir
right <- data[!(data$layer=="L1" | data$sampleID==18),]$domDir
dat_cirular <- list(
  l = circular(left, type = "angles", units = "degrees"),
  r = circular(right, type = "angles", units = "degrees")
)

watson.test(circular(left, type = "angles", units = "degrees"))
watson.test(circular(right, type = "angles", units = "degrees"))
watson.williams.test(dat_cirular)
