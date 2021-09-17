library(openxlsx)
library(gplots)
library(RColorBrewer)
rm(list = ls())

path = "/results" 

perm_clus = function(data_vec)
{
  sor_vec = c(0, 1, 2, 3, 4)
  a = data_vec[1]
  b = data_vec[13]
  c = data_vec[70]
  d = data_vec[105]
  e = setdiff(sor_vec, c(a, b, c, d))
  data_vec = ifelse(data_vec == a, 4, 
                    ifelse(data_vec == b, 0,
                           ifelse(data_vec == c, 3,
                                  ifelse(data_vec == d, 1,
                                         ifelse(data_vec == e, 2, data_vec)))))
  
  return(data_vec)    
}

data1 = read.csv(paste0(path, "/minlp_pam50-4d.csv"))
data1 = data1[c(1:232), ]
vec1 = perm_clus(data1$value)
#data1$value = ifelse(data1$value == 0, 4, 
                  #ifelse(data1$value == 1, 0,
                         #ifelse(data1$value == 2, 3,
                                #ifelse(data1$value == 3, 1,
                                       #ifelse(data1$value == 4, 2, data1$value)))))
data2 = read.csv(paste0(path, "/em_pam50-4d.csv"))
data2 = data2[c(1:232), ]
vec2 = perm_clus(data2$value)

data3 = read.csv(paste0(path, "/kmeans_pam50-4d.csv"))
data3 = data3[c(1:232), ]
vec3 = perm_clus(data3$value)

data4 = read.csv(paste0(path, "/pam_pam50-4d.csv"))
vec5 = perm_clus(data4$PAM)

vec0 = vec1
vec0[140:232] = 5
data = as.data.frame(cbind(vec0, vec5, vec2, vec3, vec1))
colnames(data) = c("True", "PAM", "EM", "K-Means", "MINLP")
#data$PAM = ifelse(data$PAM == 1, 4, 
                  #ifelse(data$PAM == 2, 0,
                         #ifelse(data$PAM == 5, 3,
                                #ifelse(data$PAM == 3, 1,
                                       #ifelse(data$PAM == 4, 2,data$PAM)))))

#pam50 = as.data.frame(data$PAM)
#colnames(pam50) = c("PAM")

data_v2 = data[c(140:232), ]
data_v3 = data_v2[order(-data_v2$MINLP),]
data = data[-c(140:232), ]
data = rbind(data, data_v3)

#write.xlsx(data, file = paste0(path, "final_processed.xlsx"))
#write.xlsx(pam50, file = paste0(path, "pam_pam50-4d.xlsx"))

# Accuracy of K-Means and EM on the pam50 data set
counter1 = vector()
counter2 = vector()
counter3 = vector()
a1 = vec0[1:139] # True assignments
a2 = vec2[1:139] # EM
a3 = vec3[1:139] # K-Means
a4 = vec5[1:139] # PAM

for(i in 1:length(a1))
{
  counter1[i] = ifelse(test = a1[i] == a2[i], yes = 1, no = 0)
}

for(j in 1:length(a1))
{
  counter2[j] = ifelse(test = a1[j] == a3[j], yes = 1, no = 0)
}

for(k in 1:length(a1))
{
  counter3[k] = ifelse(test = a1[k] == a4[k], yes = 1, no = 0)
}


em_acc = (sum(counter1)/length(a2))*100
em_acc
kmeans_acc = (sum(counter2)/length(a3))*100
kmeans_acc
pam_acc = (sum(counter3)/length(a4))*100
pam_acc

# Concordance b/w MINLP and EM/K-Means/PAM on the pam50 data set
countera = vector()
counterb = vector()
counterc = vector()
a_1 = vec1[140:232] # MINLP
a_2 = vec2[140:232] # EM
a_3 = vec3[140:232] # K-Means
a_4 = vec5[140:232] # PAM

for(m in 1:length(a_1))
{
  countera[m] = ifelse(test = a_1[m] == a_2[m], yes = 1, no = 0) # MINLP vs EM
}

for(n in 1:length(a_1))
{
  counterb[n] = ifelse(test = a_1[n] == a_3[n], yes = 1, no = 0) # MINLP vs K-Means
}

for(o in 1:length(a_1))
{
  counterc[o] = ifelse(test = a_1[o] == a_4[o], yes = 1, no = 0) # MINLP vs PAM
}

em_con = (sum(countera)/length(a_2))*100
em_con
kmeans_con = (sum(counterb)/length(a_3))*100
kmeans_con
pam_con = (sum(counterc)/length(a_4))*100
pam_con

# No of different samples from MINLP for EM, K-Means and PAM
232 - (sum(countera) + sum(counter1)) # EM
232 - (sum(counterb) + sum(counter2)) # K-Means
232 - (sum(counterc) + sum(counter3)) # PAM

# Plotting the heatmap using heatmap.2

mat_data = data.matrix(data[, 1:ncol(data)])
mat_data = t(mat_data)
palette = colorRampPalette(c("#d7191c", "#fdae61", "#ffffbf", "#abdda4", "#2b83ba", "grey"))
cnames = c("pam50", "PAM", "EM", "K-Means", "MINLP")

#png(paste0(path,"/heatmaps_in_r.png"),    # create PNG for the heat map        
#width = 5*300,        # 5 x 300 pixels
#height = 3*300,
#res = 300,            # 300 pixels per inch
#pointsize = 8)        # smaller font size

#1
heatmap.2(mat_data, notecol = "black",density.info="none", trace="none", 
          col = palette, Rowv = FALSE, Colv = FALSE, dendrogram = 'none', 
          labRow = cnames, labCol = FALSE, cexRow = 1.1, xlab = "Sample")

legend(x = "left", legend = c("Normal", "Basal", "Her2", "LumA", "LumB"), fill = c("#2b83ba", "#d7191c", "#abdda4", "#fdae61", "#ffffbf"), cex = 1.0)

#text(0.6, 0.9, "Sample", cex = 1.1)

#dev.off()
