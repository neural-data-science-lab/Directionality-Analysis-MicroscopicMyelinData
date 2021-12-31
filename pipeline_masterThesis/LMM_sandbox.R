#### try out LMM

# https://towardsdatascience.com/how-linear-mixed-model-works-350950a82911
library("lme4")
library("ggplot2")

head(sleepstudy,20)
summary(lm(Reaction~Days, data = sleepstudy))

ggplot(sleepstudy,aes(x=Days,y=Reaction)) + geom_point() + geom_smooth(method = "lm")

plot(Reaction ~ Days, data = sleepstudy)
abline(lm(Reaction ~ Days, data = sleepstudy), col = "blue")
conf_interval <- predict(lm(Reaction ~ Days, data = sleepstudy), 
                         newdata = data.frame(Days = seq(0, 9, by = 0.1)), 
                         interval = "confidence", level = 0.95)
lines(seq(0, 9, by = 0.1), conf_interval[,2], col = "blue", lty = 2)
lines(seq(0, 9, by = 0.1), conf_interval[,3], col = "blue", lty = 2)

ggplot(sleepstudy,aes(x=Days,y=Reaction)) + geom_smooth(method = "lm",level = 0.95) + 
  geom_point() + facet_wrap(~Subject, nrow = 3, ncol = 6)

### LMM
summary(lmer(Reaction ~ Days + (Days | Subject), sleepstudy))

sqrt(sum(residuals(lm(Reaction~Days,data=sleepstudy))^2)/(dim(sleepstudy)[1]-2))
sqrt(sum(resid(lmer(Reaction~Days+(Days|Subject),sleepstudy))^2)/(dim(sleepstudy)[1]-2))

# simple linear regression
fit1 <- lm(Reaction ~ Days, data = sleepstudy)
# random slopes and intercepts for the effect of Days for each individual (Subject)
fit2 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML = FALSE)
# Subject as a blocking factor - random effect is constant within each group
fit3 <- lmer(Reaction ~ Days + (1 | Subject), sleepstudy, REML = FALSE)
#random effects for both intercept and slope
fit4 <- lmer(Reaction ~ (Days | Subject), sleepstudy)
anova(fit4, fit3, fit2, fit1)


###################### Seminar09 #################
#### 1.) Analysis of longitudinal data
## a. simple linear regerssion: distance ~ age, interpret estimates for intercept, slope, model fit
library(nlme)
attach(Orthodont)
head(Orthodont)
fm1Orth.lm <- lm(distance ~ age, Orthodont)
sum.lm < - summary(fm1Orth.lm)
sum.lm
sum.lm$r.squared #proportion of explained variance via model
sum.lm$adj.r.squared #proportion of explained variance adjusted
anova(fm1Orth.lm)
ggplot(Orthodont,aes(x=age,y=distance)) + geom_point() + geom_smooth(method = "lm")

fm1Orth.lis <- lmList(distance ~ age | Subject, Orthodont)
plot(intervals(fm1Orth.lis)) # plot confidence interval for all models -> need random effects?

fm1Orth.lme <- lme(distance~age, data=Orthodont, random = ~age | Subject, method = "ML")
summary(fm1Orth.lme)
compOrth <- compareFits(coef(fm1Orth.lis), coef(fm1Orth.lme))
anova(fm1Orth.lme,fm1Orth.lm)
# fit for gender 
fm2Orth.lme <- update(fm1Orth.lme, fixed=distance~Sex*age)
summary(fm2Orth.lme)
anova(fm2Orth.lme)

# Model diagnostics for the LME fit:
plot(fm2Orth.lme, resid(.,type="p")~fitted(.) | Sex,id=0.05,adj=-0.3)
plot(fm1Orth.lme, resid(.,type="p")~fitted(.))
# There may be a few outliers
qqnorm(fm2Orth.lme, ~resid(.) | Sex)
qqnorm(fm1Orth.lme, ~resid(.))
# Normality assumption OK for within subject errors
qqnorm(fm2Orth.lme, ~ranef(.),id=0.1,cex=0.7)
qqnorm(fm1Orth.lme, ~ranef(.),id=0.1,cex=0.7)
# Normality assumption OK for random effects


###################### Seminar09 #################
#### 2.) Randomized Block Design
library(BHH2)
library(lme4)
data(penicillin.data) ## Download the data
plot(penicillin.data) ## which of these plots provides information
## about the relationship between treatment and response?
## how can you interpret the other plots?
mod0 <- lm(yield ~ blend + treat, penicillin.data)
summary(mod0)
anova(mod0)
## Specify the contr.sum option in R:
op <- options(contrasts = c("contr.sum", "contr.poly"))
## Fit the fixed effects model, using the blend and treatment as the explanatory variables:
mod1 <- lm(yield ~ blend + treat, penicillin.data)
summary(mod1)
anova(mod1)
## Fit the mixed effects model, using the blend as a blocking factor
mod2 <- lmer(yield ~ treat + (1|blend), penicillin.data)
summary(mod2)
## To see the estimates for the random effects:
ranef(mod2)$blend
## To see the estimates for the fixed effects:
coef(mod1)
## Regression diagnostics:
qqnorm(resid(mod2))
qqline(resid(mod2))
plot(fitted(mod2),resid(mod2), xlab="Fitted values", ylab="Residuals")
abline(0,0,col="red")
## Simple ANOVA
anova(mod2)
## ANOVA to compare the model of interest to the null model:
mod2.ml <- lmer(yield ~ treat + (1|blend), penicillin.data, REML=FALSE)
mod3.ml <- lmer(yield ~ 1 + (1|blend), penicillin.data,REML=FALSE)
anova(mod2.ml,mod3.ml)



