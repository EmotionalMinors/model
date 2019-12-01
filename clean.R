setwd("/Users/sherry/Desktop/emotional_minors/datasets/subset/")
temp <- read.csv("all.txt", header = FALSE, quote = "", sep = "\t", stringsAsFactors = FALSE)
frame1 <- temp[temp$V12 !="", ]
frame2 <- temp[temp$V12 =="" & temp$V13 != "", ]
frame3 <- temp[temp$V12 =="" & temp$V13 =="", ]
#adding headers
name_1 <- c("unique_id", "asin", "id", "product_name", "product_type", 
            "product_type2", "helpful", "rating", "title", "date", "reviewer", "location", "text")
name_2 <- c("unique_id", "asin", "id", "product_name", "product_type", 
            "product_type2", "helpful", "rating", "title", "date", "reviewer", "location", "text")
name_3 <- c("unique_id", "id", "product_name", "product_type", "helpful", "rating", "title", "date",
            "reviewer","location", "text", "bs1", "bs2")
names(frame1) <- name_1
names(frame2) <- name_2
names(frame3) <- name_3
#data merging
frame1 <- frame1[ , !names(frame1) %in% c("asin", "product_type2")]
frame2 <- frame2[, !names(frame2) %in% c("asin", "product_type2")]
frame2$location <- NA
frame3 <- frame3[, !names(frame3) %in% c("bs1", "bs2")]
frame <- rbind(frame1, frame2, frame3)
frame$rating <- as.numeric(frame$rating)
frame <- frame[!is.na(frame$rating), ]

#check random sentences
View(frame[12000* runif(30, 0, 1), ])

write.csv(frame, "cleanset.csv")


