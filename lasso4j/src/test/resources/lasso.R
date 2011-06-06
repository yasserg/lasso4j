# glmnet package needs to be installed and loaded.

diab <- read.table("C:/Yasser/workspace-jee/lasso4j/src/main/resources/diabetes.data",header=T)
fit1 <- glmnet(as.matrix(diab[,-11]), diab[,11])
