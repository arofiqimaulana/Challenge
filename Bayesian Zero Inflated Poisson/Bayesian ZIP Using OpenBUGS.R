#Input Data
data=read.table("D:\\DATA FILARIASIS.txt",header=T)
attach(data)

#Diagram Batang
win.graph()
barplot(table(Y),xlab = "Banyak Kejadian Filariasis ", ylab ="Frekuensi" , font.lab=2,cex.lab=1, border = "white",col="blue",ylim=c(0,15))

#Uji Kolmogorov Smirnov
KS=function(Y){
   m=sort(unique(unlist(Y)),decreasing=FALSE)
   FS=ppois(m,mean(Y),lower.tail=TRUE)#cdf
   frek=as.matrix(table(Y))
   FT=cumsum(frek)/length(Y)#Frekuensi Komulatif
   D=max(abs(FT-FS))
   D
}
KS(Y)

#Pendeteksian Multikolinieritas
library(car)
model=lm(Y~X1+X2+X3+X4+X5+X6+X7)
VIF=vif(model)

#Regresi Poisson
model1=glm(formula=Y~X1+X2+X3+X4+X5+X6+X7,family="poisson",data=data)
summary(model1)

#Pengujian Overdispersi
Overdis_test = function(data,model){
	n=nrow(data) #banyak pengamatan 
	k=ncol(data)-1 #banyak prediktor
	pearson_Chi2= sum(residuals(model, type= "pearson")^2) 
	crit_value = qchisq(0.95,n-k-1) 
	pvalue=1-pchisq(pearson_Chi2,n-k-1) 
	cbind(pearson_Chi2,crit_value,pvalue) 
}

Overdis_test(data,model1)

#Regresi ZIP Metode Bayesian
library(R2OpenBUGS)
model=function(){
  for(j in 1 :N) {
   y[j]~dpois(mu[j]) 
    mu[j]<-(1-u[j])*lambda[j]
    u[j]~dbern(p[j]) 
 
    logit(p[j])<a0+a1*x1[j]+a2*x2[j]+a3*x3[j]+a4*x4[j]+a5*x5[j]+a6*x6[j]+ a7*x7[j]
    log(lambda[j])<b0+b1*x1[j]+b2*x2[j]+b3*x3[j]+b4*x4[j]+b5*x5[j]+b6*x6[j]+ b7*x7[j]
}

 
a0~dnorm(-0.003308,0.996409697) 
a1~dnorm(0.453427,3.114150214)
a2~dnorm(-0.078303,1.280850422)
a3~dnorm(0.024872,1.188802716)
a4~dnorm(-0.453678,2.287358884)
a5~dnorm(-0.0387,1.104960481)
a6~dnorm(-0.951789,1.316422976)
a7~dnorm(-1.081704,2.10760152)
b0~dnorm(0.039857,1.012332916)
b1~dnorm(-0.067612,228.943594)
b2~dnorm(-0.019018,41.28169832)
b3~dnorm(0.282384,27.90312785)
b4~dnorm(0.042059,508.6377887)
b5~dnorm(-0.046027,429.3637396)
b6~dnorm(-0.155483,102.1955028)
b7~dnorm(0.018603,475.272318)

}



#Input data dari
directory

data1=read.table("D:\\DATAR2OPENBUGS.txt",header =TRUE)

data=as.list(data1)

data$N=nrow(data1)



inits=function(){

list(a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0,b0=0,b1=0,b2=0,b3= 0,b4=0,b5=0,b6=0,b7=0) 
}

sim=bugs(data=data,inits=inits,
	parameters.to.save=c("a0","a1"," a2","a3","a4","a5","a6","a7","b0","b1","b2","b3","b4","b5","b6","b7"),
	model.file=model,n.chains=1,n.thin=50,n.iter=110000,n.burnin=10000,codaPkg=TRUE)

#Membuat MCMC objek
library(coda)
sim.coda=read.bugs(sim)
summary(sim.coda)

nchain=nchain(sim.coda) #banyak chains
npar=nvar(sim.coda)-1 #banyak parameter
n=niter(sim.coda) #banyak iterasi

#Monitoring Konvergensi Algoritma
library(lattice)

#Traceplot 
 
win.graph()
par(mfrow=c(2,2))
traceplot(sim.coda,col="blue")

#autokorelasi
win.graph()
par(mfrow=c(4,2))
autocorr.plot(sim.coda,lag.max=100,auto.layout =FALSE,col="blue")

#density plot
win.graph()
par(mfrow=c(4,2))
densplot(sim.coda,col="blue")

#Ringkasan Posterior
param=as.matrix(sim.coda,iters=FALSE,chains=FALSE)
library(mcmcse)
MEAN=apply(param,2,function(x) mcse(x,method="bm",g=NULL)$est)
SE=apply(param,2,function(x) mcse(x,method="bm",g=NULL)$se)
SD=apply(param,2,function(x) sd(x))
Ringkasan = cbind(MEAN,SD,SE)

#Credible Interval
q2.5=apply(param,2,function(x) quantile(x,c(0.025)))
median=apply(param,2,function(x) quantile(x,c(0.5)))
q97.5=apply(param,2,function(x) quantile(x,c(0.975)))
quantile=cbind(q2.5,median,q97.5)
cred_int=cbind(q2.5,q97.5)

#Ringkasan Posterior Full 
Ringkasan_full=cbind(MEAN,SD,SE,q2.5,median,q97.5) 
Ringkasan_full=as.matrix(Ringkasan_full) 
rownames(Ringkasan_full)=c("a0","a1","a2","a3","a4","a5","a6","a7","b0","b1","b2","b3","b4","b5","b6","b7","deviance")
colnames(Ringkasan_full)=c("Rata-Rata","sd","MC Error","2.5%","median","97.5%")