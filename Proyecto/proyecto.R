################################################################################
#                     PROYECTO FINAL APRENDIZAJE AUTOM?TICO
#         Juan Alberto Martínez López / Alberto Armijo Ruíz
################################################################################


# 1. default of credit card clients Data Set (Clasificación)

#Librerías utilizadas.
library("caret")
library("leaps")
library("e1071")
library("ROCR")
library("randomForest")
library("neuralnet")
library("ada")

set.seed(1)

credit_card =  read.csv("default_of_credict_card_clients.csv"
                        , sep=",", header = TRUE, row.names =1)
attach(credit_card)
summary(credit_card)

set.seed(1)
train = sample (nrow(credit_card), round(nrow(credit_card)*0.7)) 
credit_card.train = credit_card[train,]  
credit_card.test = credit_card[-train,]

anyNA(credit_card.train)

credit_card.train$SEX = ifelse(credit_card.train$SEX == 2, 0, 1)

ed.other = ifelse(credit_card.train$EDUCATION == 4, 1,0)
ed.university = ifelse(credit_card.train$EDUCATION == 2, 1, 0)
ed.high_school = ifelse(credit_card.train$EDUCATION == 3 | credit_card.train$EDUCATION == 2, 1, 0)
ed.school = ifelse(credit_card.train$EDUCATION == 1 | credit_card.train$EDUCATION == 2 | credit_card.train$EDUCATION == 3, 1, 0)

credit_card.train = cbind(credit_card.train,ed.other, ed.high_school, ed.school, ed.university)

# Borramos la columna EDUCATION.
credit_card.train = credit_card.train[,-which(colnames(credit_card.train) == "EDUCATION")]

# También tenemos que modificar la columna mariage. Introduciremos tres nueva columnas: marriage.married, marriage.single, marriage.others.
marriage.married = ifelse(credit_card.train$MARRIAGE == 1, 1,0)
marriage.single = ifelse(credit_card.train$MARRIAGE == 2, 1,0)
marriage.others = ifelse(credit_card.train$MARRIAGE == 3, 1,0)

# Introducimos los datos.
credit_card.train = cbind(credit_card.train, marriage.married, marriage.single, marriage.others)

# Borramos la variable MARRIAGE.
credit_card.train = credit_card.train[, -which(colnames(credit_card.train) == "MARRIAGE")]

colnames(credit_card.train)[which(colnames(credit_card.train)=="PAY_0")]="PAY_1"

trans = preProcess(credit_card.train, c("BoxCox") )
trainTransformado = predict(trans, credit_card.train)
summary(trainTransformado)

reemplazarCol = function(x){
  # Columna SEX
  x$SEX = ifelse(x$SEX == 2, 0, 1)
  
  # Columna EDUCATION.
  ed.other = ifelse(x$EDUCATION == 4, 1,0)
  ed.university = ifelse(x$EDUCATION == 2, 1, 0)
  ed.high_school = ifelse(x$EDUCATION == 3 | x$EDUCATION == 2, 1, 0)
  ed.school = ifelse(x$EDUCATION == 1 | x$EDUCATION == 2 | x$EDUCATION == 3, 1, 0)
  x = cbind(x,ed.other, ed.high_school, ed.school, ed.university)
  
  # Borramos la columna EDUCATION.
  x = x[,-which(colnames(x) == "EDUCATION")]
  
  # Columna MARRIAGE.
  marriage.married = ifelse(x$MARRIAGE == 1, 1,0)
  marriage.single = ifelse(x$MARRIAGE == 2, 1,0)
  marriage.others = ifelse(x$MARRIAGE == 3, 1,0)
  
  # Introducimos los datos.
  x = cbind(x, marriage.married, marriage.single, marriage.others)
  
  # Borramos la variable MARRIAGE.
  x = x[, -which(colnames(x) == "MARRIAGE")]
  
  # Cambiamos el nombre de la variables PAY_0.
  colnames(x)[which(colnames(x)=="PAY_0")]="PAY_1"
  x
}

# Función que realiza la transformación BoxCox.
preprocesar = function(x,pred=trans){
  transTest = predict(pred, x)
  transTest
}

# Función que engloba las funciones anteriores.
prepareTest = function(x){
  tr = reemplazarCol(x)
  tr= preprocesar(x)
  tr
}


pcaTransformation = prcomp(trainTransformado[,-default.payment.next.month], center=F, scale=F)
pcaTransformation$rotation


muestra_regsubsets = regsubsets(default.payment.next.month ~ ., data = trainTransformado, nvmax = 28, method = "exhaustive")
summary((muestra_regsubsets))
reg.sumary = summary(muestra_regsubsets)
par(mfrow=c(1,2))
plot(reg.sumary$cp, xlab="number of variables", ylab="cp", type="l")
which.min(reg.sumary$cp)
plot(reg.sumary$bic, xlab="number of variables", ylab="BIC", type="l")
which.min(reg.sumary$bic)
par(mfrow=c(1,1) )

# Función para calcular la solución dada una predicción.
calculateSol = function(x){
  prediction.model = rep(0,length(x))
  prediction.model[x >= 0.5] = 1
  
  prediction.model
}

# Función para calcular el Error.
calculateErrorClasification = function(calculated.solution, real.sol){
  er = sum(calculated.solution != real.sol)/length(calculated.solution)
  er
}

# Función que calcula el Error pasándole la predicción.
calculateError = function(model.prediction, labels){
  pred = calculateSol(model.prediction)
  return(calculateErrorClasification(pred, labels))
}

testTransformado = reemplazarCol(credit_card.test)
testTransformado = predict(trans, testTransformado)
summary(testTransformado)

m1 = glm(default.payment.next.month~LIMIT_BAL+SEX+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6+ed.other+ed.high_school+ed.school+ed.university+marriage.married+marriage.single, data=trainTransformado, family="binomial")

predtr.m1 = predict(m1, trainTransformado)
Ein.m1 = calculateError(predtr.m1, trainTransformado$default.payment.next.month)
Ein.m1

pred.m1 = predict(m1, testTransformado)
Eout.m1 = calculateError(pred.m1, testTransformado$default.payment.next.month)
Eout.m1

readline("Pulsa Enter para continuar")

m2 = glm(default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married+PAY_3+SEX+PAY_AMT2+ed.high_school+PAY_AMT4+marriage.single+marriage.others+BILL_AMT2,data=trainTransformado, family="binomial")
predtr.m2 = predict(m2, trainTransformado)
Ein.m2 = calculateError(predtr.m2, trainTransformado$default.payment.next.month)
Ein.m2

pred.m2 = predict(m2, testTransformado)
Eout.m2 = calculateError(pred.m2, testTransformado$default.payment.next.month)
Eout.m2

readline("Pulsa Enter para continuar")

m3= glm(default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married,data=trainTransformado, family="binomial" )
predtr.m3 = predict(m3, trainTransformado)
Ein.m3 = calculateError(predtr.m3, trainTransformado$default.payment.next.month)
Ein.m3

pred.m3 = predict(m3, testTransformado)
Eout.m3 = calculateError(pred.m3, testTransformado$default.payment.next.month)
Eout.m3

readline("Pulsa Enter para continuar")

linear.model = ROCR::prediction(pred.m3, testTransformado$default.payment.next.month)
perf = performance(linear.model, "tpr", "fpr")
auc.linear = performance(linear.model, measure = "auc")
print(auc.linear@y.values[[1]])
plot(perf)

readline("Pulsa Enter para continuar")

bt1 = ada::ada(default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married,data=trainTransformado, iter=10)
pred.bt1 = predict(bt1, trainTransformado)
Ein.bt1 = mean(pred.bt1 != trainTransformado$default.payment.next.month)
Ein.bt1

predTs.bt1 = predict(bt1, testTransformado)
Eout.bt1 = mean(predTs.bt1 != testTransformado$default.payment.next.month)
Eout.bt1

readline("Pulsa Enter para continuar")

bt2 = ada::ada(formula = default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married,data=trainTransformado, iter=30)
pred.bt2 = predict(bt2, trainTransformado)
Ein.bt2 = mean(pred.bt2 != trainTransformado$default.payment.next.month)
Ein.bt2

predTs.bt2 = predict(bt2, testTransformado)
Eout.bt2 = mean(predTs.bt2 != testTransformado$default.payment.next.month)
Eout.bt2

readline("Pulsa Enter para continuar")

bt3 = ada::ada(formula=default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married, data=trainTransformado, iter=5)
pred.bt3 = predict(bt3, trainTransformado)
Ein.bt3 = mean(pred.bt3 != trainTransformado$default.payment.next.month)
Ein.bt3

predTs.bt3 = predict(bt3, testTransformado)
Eout.bt3 = mean(predTs.bt3 != testTransformado$default.payment.next.month)
Eout.bt3

readline("Pulsa Enter para continuar")

predi = predict(bt3, testTransformado, type = "probs")
salida.boosting = predi[,2]
boosting.model = ROCR::prediction(salida.boosting, testTransformado$default.payment.next.month)
perf.boost = performance(boosting.model, "tpr", "fpr")
auc.boosting = performance(boosting.model, measure = "auc")
print(auc.boosting@y.values[[1]])
plot(perf.boost)

readline("Pulsa Enter para continuar")

nn1 = neuralnet(formula = default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married,data=trainTransformado,hidden = c(5,3,2), threshold = 0.8, stepmax = 1e+06)

plot(nn1)


n <- c("PAY_1", "PAY_2", "BILL_AMT1", "LIMIT_BAL","AGE","ed.school","PAY_AMT1","PAY_5","marriage.married")

pred.nn1 = compute(nn1, trainTransformado[names(trainTransformado) %in% n])

pred.nn1_ = pred.nn1$net.result*(max(pred.nn1$net.result)-min(pred.nn1$net.result))+min(pred.nn1$net.result)

calculateError(pred.nn1_,trainTransformado$default.payment.next.month)

pred.nn1Ts = compute(nn1, testTransformado[names(testTransformado) %in% n])

pred.nn1_test = pred.nn1Ts$net.result*(max(pred.nn1Ts$net.result)-min(pred.nn1Ts$net.result))+min(pred.nn1Ts$net.result)


calculateError(pred.nn1_test,testTransformado$default.payment.next.month)

readline("Pulsa Enter para continuar")

nn2 = neuralnet(formula = default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married,data=trainTransformado,hidden = c(5,3), threshold = 0.5, stepmax = 1e+06)

plot(nn2)

pred.nn2 = compute(nn2, trainTransformado[names(trainTransformado) %in% n])

pred.nn2_ = pred.nn2$net.result*(max(pred.nn2$net.result)-min(pred.nn2$net.result))+min(pred.nn2$net.result)

calculateError(pred.nn2_,trainTransformado$default.payment.next.month)

pred.nn2Ts = compute(nn2, testTransformado[names(testTransformado) %in% n])

pred.nn2_test = pred.nn2Ts$net.result*(max(pred.nn2Ts$net.result)-min(pred.nn2Ts$net.result))+min(pred.nn2Ts$net.result)


calculateError(pred.nn2_test,testTransformado$default.payment.next.month)

readline("Pulsa Enter para continuar")

nn3 = neuralnet(formula = default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married,data=trainTransformado,hidden = 5, threshold = 0.5, stepmax = 1e+06)

plot(nn3)

pred.nn3 = compute(nn3, trainTransformado[names(trainTransformado) %in% n])

pred.nn3_ = pred.nn3$net.result*(max(pred.nn3$net.result)-min(pred.nn3$net.result))+min(pred.nn3$net.result)

calculateError(pred.nn3_,trainTransformado$default.payment.next.month)

pred.nn3Ts = compute(nn3, testTransformado[names(testTransformado) %in% n])

pred.nn3_test = pred.nn3Ts$net.result*(max(pred.nn3Ts$net.result)-min(pred.nn3Ts$net.result))+min(pred.nn3Ts$net.result)


calculateError(pred.nn3_test,testTransformado$default.payment.next.month)

readline("Pulsa Enter para continuar")

nn4 = neuralnet(formula = default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married,data=trainTransformado,hidden = c(3,2), threshold = 0.4, stepmax = 1e+06)
plot(nn4)

pred.nn4 = compute(nn4, trainTransformado[names(trainTransformado) %in% n])

pred.nn4_ = pred.nn4$net.result*(max(pred.nn4$net.result)-min(pred.nn4$net.result))+min(pred.nn4$net.result)

calculateError(pred.nn4_,trainTransformado$default.payment.next.month)

pred.nn4Ts = compute(nn4, testTransformado[names(testTransformado) %in% n])

pred.nn4_test = pred.nn4Ts$net.result*(max(pred.nn4Ts$net.result)-min(pred.nn4Ts$net.result))+min(pred.nn4Ts$net.result)


calculateError(pred.nn4_test,testTransformado$default.payment.next.month)

readline("Pulsa Enter para continuar")

probsnn = compute(nn3, testTransformado[names(testTransformado) %in% n])
probsnn.pred = probsnn$net.result
nn.model = ROCR::prediction(probsnn.pred, testTransformado$default.payment.next.month)
perfnn = performance(nn.model, "tpr", "fpr")
auc.nn = performance(nn.model, measure = "auc")
print(auc.nn@y.values[[1]])
plot(perfnn)

readline("Pulsa Enter para continuar")

svm1 = e1071::svm(formula=default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married, data=trainTransformado, gamma=0.1, type="C", tolerance=0.01)

pred.svm1 = predict(svm1, trainTransformado)
predTs.svm1 = predict(svm1, testTransformado)
Ein.svm1 = mean(pred.svm1 != trainTransformado$default.payment.next.month)
Eout.smv1 = mean(predTs.svm1 != testTransformado$default.payment.next.month)
Ein.svm1
Eout.svm1

readline("Pulsa Enter para continuar")

svm2 = e1071::svm(formula=default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married, data=trainTransformado, gamma=0.01, type="C", tolerance=0.01)

pred.svm2 = predict(svm2, trainTransformado)
predTs.svm2 = predict(svm2, testTransformado)
Ein.svm2 = mean(pred.svm2 != trainTransformado$default.payment.next.month)
Eout.smv2 = mean(predTs.svm2 != testTransformado$default.payment.next.month)
Ein.svm2
Eout.svm2

readline("Pulsa Enter para continuar")

svm3 = e1071::svm(formula=default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married, data=trainTransformado, gamma=1, type="C", tolerance=0.01)

pred.svm3 = predict(svm3, trainTransformado)
predTs.svm3 = predict(svm3, testTransformado)
Ein.svm3 = mean(pred.svm3 != trainTransformado$default.payment.next.month)
Eout.smv3 = mean(predTs.svm3 != testTransformado$default.payment.next.month)
Ein.svm3
Eout.svm3

readline("Pulsa Enter para continuar")

svm4 = e1071::svm(formula=default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married, data=trainTransformado, gamma=15, type="C", tolerance=0.01)

pred.svm4 = predict(svm4, trainTransformado)
predTs.svm4 = predict(svm4, testTransformado)
Ein.svm4 = mean(pred.svm4 != trainTransformado$default.payment.next.month)
Eout.smv4 = mean(predTs.svm4 != testTransformado$default.payment.next.month)
Ein.svm4
Eout.svm4

readline("Pulsa Enter para continuar")

s.svm = e1071::svm(formula=default.payment.next.month~PAY_1+PAY_2+BILL_AMT1+LIMIT_BAL+AGE+ed.school+PAY_AMT1+PAY_5+marriage.married, data=trainTransformado, gamma=0.1, type="C", tolerance=0.01, probability=T, decision.values=T)
pred.s = predict(s.svm, testTransformado, probability=T)
probs.svm = attr(pred.s, "probabilities")[,1]
svm.model = ROCR::prediction(probs.svm, testTransformado$default.payment.next.month)
perf = performance(svm.model, "tpr", "fpr")
auc.svm = performance(svm.model, measure = "auc")
print(auc.svm@y.values[[1]])
plot(perf)

readline("Pulsa Enter para continuar")

n <- c("PAY_1", "PAY_2", "BILL_AMT1", "LIMIT_BAL","AGE","ed.school","PAY_AMT1","PAY_5","marriage.married")
rf = randomForest::randomForest(x=trainTransformado[, names(trainTransformado)%in%n], y=as.factor(trainTransformado$default.payment.next.month), data=trainTransformado, ntree=1000, maxnodes = 400)

pred.rf1 = predict(rf, trainTransformado)
Ein.rf1 = mean( pred.rf1 != trainTransformado$default.payment.next.month)
Ein.rf1

predTs.rf1 = predict(rf, testTransformado)
Eout.rf1 = mean( predTs.rf1 != testTransformado$default.payment.next.month)
Eout.rf1

readline("Pulsa Enter para continuar")

rf2 = randomForest::randomForest(x=trainTransformado[, names(trainTransformado)%in%n], y=as.factor(trainTransformado$default.payment.next.month), data=trainTransformado, ntree=1500, maxnodes = 400)

pred.rf2 = predict(rf2, trainTransformado)
Ein.rf2 = mean( pred.rf2 != trainTransformado$default.payment.next.month)
Ein.rf2

predTs.rf2 = predict(rf2, testTransformado)
Eout.rf2 = mean( predTs.rf2 != testTransformado$default.payment.next.month)
Eout.rf2

readline("Pulsa Enter para continuar")

rf3 = randomForest::randomForest(x=trainTransformado[, names(trainTransformado)%in%n], y=as.factor(trainTransformado$default.payment.next.month), data=trainTransformado, ntree=2000, maxnodes = 400)

pred.rf3 = predict(rf3, trainTransformado)
Ein.rf3 = mean( pred.rf3 != trainTransformado$default.payment.next.month)
Ein.rf3

predTs.rf3 = predict(rf3, testTransformado)
Eout.rf3 = mean( predTs.rf3 != testTransformado$default.payment.next.month)
Eout.rf3

readline("Pulsa Enter para continuar")

probs = predict(rf, testTransformado, type="prob")
probs.pred = probs[,2]
rf.model = ROCR::prediction(probs.pred, testTransformado$default.payment.next.month)
perf = performance(rf.model, "tpr", "fpr")
auc.rf = performance(rf.model, measure = "auc")
print(auc.rf@y.values[[1]])
plot(perf)