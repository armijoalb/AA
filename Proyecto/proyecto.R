setwd("~/Documentos/Segundo_Cuatrimestre/AA/Proyecto")

credictCardData <- read.csv("default_of_credict_card_clients.csv"
                          , sep=",", header = TRUE, row.names =1)
View(credictCardData)
summary(credictCardData)
dim(credictCardData)
