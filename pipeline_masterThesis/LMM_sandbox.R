###### Explore linear mixed models 

# 2. Randomized Block Design
#use now a different contrast option
library(BHH2)
library(lme4)
data(penicillin.data) 
plot(penicillin.data) 
#which of the plots provide information about treatment ~response
#how can the other plots be interpreted

#b. fit fixed effect models -> significant difference between treatments, blends, interepret, of coefficients
mod0 <- lm(yield ~ blend + treat, penicillin.data)
summary(mod0)
'90 <- (B1,TA) baseline categorie; estimate of yield for certain categories 
-> predictions e.g. (B2,TA) = 90-9, TA fixed, (B1,TB)=90+1, (B1,T2) = 90+2, (B3,TC)=90-7+5'
anova(mod0)

op <- options(contrasts = c("contr.sum", "contr.poly"))

#c. blend as a blocking factor
mod1 <- lmer(yield ~ treat + (1|blend), penicillin.data)
summary(mod1)
anova(mod1)
ranef(mod1)$blend ## To see the estimates for the random effects
coef(mod0) ## To see the estimates for the fixed effects

## Regression diagnostics:
qqnorm(resid(mod1))
qqline(resid(mod1))
plot(fitted(mod0),resid(mod1), xlab="Fitted values", ylab="Residuals")
abline(0,0,col="red")

## ANOVA to compare the model of interest to the null model:
mod2.ml <- lmer(yield ~ treat + (1|blend), penicillin.data, REML=FALSE)
mod3.ml <- lmer(yield ~ 1 + (1|blend), penicillin.data,REML=FALSE)
anova(mod2.ml,mod3.ml)

## Bootstrap to increase the accuracy in the second method:
set.seed(1)
nboot <- 1000
allLR <- numeric(nboot)
for(i in 1:nboot){
  new.yield <- unlist(simulate(mod3.ml))
  new.mod2.ml <- lmer(new.yield ~ treat + (1|blend), penicillin.data, REML=FALSE)
  new.mod3.ml <- lmer(new.yield ~ 1 + (1|blend), penicillin.data, REML=FALSE)
  allLR[i] <- 2*(logLik(new.mod2.ml) - logLik(new.mod3.ml))}
## We can use a qqplot to check whether the empirical distribution
## of the likelihood ratio statistic is chi-squared with 3 df
plot(qchisq((1:nboot)/(nboot+1),3), sort(allLR), ylab="Simulated LR Statistic",
     xlab="Chi-Squared distribution with 3 df")
abline(0,1)
mean(allLR > 4.0474)
library(BHH2)
library(lme4)
library(lmerTest) ## the package produces an approximation of p-values
op <- options(contrasts = c("contr.sum", "contr.poly"))
mod5 <- lmer(yield ~ treat + (1|blend), penicillin.data)
summary(mod5)
anova(mod5)

mod3 <- lm(yield ~ treat, penicillin.data)
mod4 <- lmer(yield ~ treat + (1|blend), penicillin.data, REML=TRUE)
2*(logLik(mod4) - logLik(mod3, REML=TRUE ))
set.seed(2)
nboot <- 1000
allLR <- numeric(nboot)
for(i in 1:nboot){
  new.yield <- unlist(simulate(mod3))
  new.mod3 <- lm(new.yield ~ treat, penicillin.data)
  new.mod4 <- lmer(new.yield ~ treat + (1|blend), penicillin.data, REML=TRUE)
  allLR[i] <- 2*(logLik(new.mod4) - logLik(new.mod3, REML=TRUE))
}
## P-value
mean(allLR > 2*(logLik(mod4) - logLik(mod3, REML=TRUE)))

#what should you know: 
'-repeated measurement (many days, many colonies from one plate, different mice -> hierarchical effects) <-> unrepeated
- how test effects: interaction, usual effect, Ftest: without and with interaction
-lin regression: same thing
-pvalue is not best thing to summerize data from mixed effects data'

