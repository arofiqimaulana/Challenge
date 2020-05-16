#install.packages(c("survival","survminer","concordance"))

## Load Package
library(survival)
library(survminer)
library(concordance)

## Load Dataset
data("lung")
head(lung)
options(scipen=999)

## Membuat tabel untuk kurva kaplan Meyer
fit <- survfit(Surv(time, status)~1, data = lung)
surv_table <- data.frame(time=fit$time,
                         n_risk=fit$n.risk,
                         n_event=fit$n.event,
                         n_censor=fit$n.censor,
                         sur_prob=fit$surv)
surv_table

## Kurva Kaplan Meyer (perhitungan secara mandiri)
plot(x=surv_table$time,
     y=surv_table$sur_prob,
     type="l", 
     ylab="survival probability",
     xlab="time")


## Kurva Kaplan Meyer (Disediakan oleh package)
ggsurvplot(fit,
           surv.median.line = "hv",
           ggtheme = theme_bw(),
           color="blue",
           legend="none",
           xlab = "Time in Days") +
           ggtitle("Kaplan Meier Curve Lung Dataset")


## Regresi Cox Proportional Hazard
lung$sex <- as.factor(lung$sex) # basline pada pembentukan variabel dummy
model_coxph <- coxph (Surv(time, status) ~ sex + age + ph.ecog + ph.karno + pat.karno +
                        meal.cal +wt.loss,data=lung)
model_coxph


options(scipen=999)
summary(model_coxph)

## Regresi Cox PH dengan mengasumsikan distirbusi exponential
exp_model <- survreg(Surv(time, status) ~ sex + age + ph.ecog + ph.karno + pat.karno +
                        meal.cal +wt.loss, data=lung, dist="exponential")
summary(exp_model)

## C-statistik for Evaluation Metrik
concordance(exp_model)

concordance(model_coxph)