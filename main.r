library(stringr)
library(tm)
library(RTextTools)
library(caret)

# read data
raw_data <- read.csv('SMSSpamCollection.tsv', header=F, sep='\t',
                      col.names=c('label', 'sms'), quote="", colClasses=c("factor", "character"))
raw_data$label <- as.numeric(raw_data$label)
summary(raw_data)

# setup regex patterns
phone_regex = "[0-9]{11}|[0-9]{4}[ -][0-9]{3}[ -][0-9]{4}"
smsnumber_regex = "\\b[0-9]{5}\\b"

number_regex = "[0-9]+"
code_regex = "[0-9]+[a-zA-Z]+[0-9]*|[a-zA-Z]+[0-9]+[a-zA-Z]*"

char_to_remove = "[^0-9a-zA-Z _]"
char_to_remove2 = "'"
char_to_remove3 = "_"
money_regex = "[£$][0-9.,]+"
exmark_regex = "!"

all_upper = "\\b[A-Z]{2,}\\b"

html_special_regex = "&?#?[xX]?[a-zA-Z0-9]+;"
url_regex = "http:[a-zA-Z0-9.\\/=?&\\-_#]+|www[a-zA-Z0-9.\\/=?&\\-_#]+"

n = nrow(raw_data)
temp_data <- raw_data$sms
#print(temp_data)

# apply regex to remove/replace certain words/characters
# remove "_"
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(char_to_remove3, ' ', x,perl=TRUE)))
# remove html special characters
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(html_special_regex, ' ', x,perl=TRUE)))
# replace url with "_url_"
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(url_regex, '_url_', x,perl=TRUE)))
# replace all-cap words with "word _upper_" 
for (i in c(1:n)){
  upper_list <- str_extract_all(temp_data[i], all_upper, simplify=F)
  if (length(upper_list[[1]]) > 0){
    temp_line = temp_data[i]
    for (word in upper_list[[1]]){
      temp_line <- gsub(paste("\\b", word, "\\b",sep=""), paste(word, " _upper_ ", sep=""), temp_line)
    }
    temp_data[i] <- temp_line
  }
}
# replace phone numbers with "_phone_"
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(phone_regex, ' _phone_ ', x,perl=TRUE)))
# replace sms numbers with "_smsnumber_"
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(smsnumber_regex, ' _smsnumber_ ', x,perl=TRUE)))
# replace mention of money with "_money_"
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(money_regex, ' _money_ ', x,perl=TRUE)))
# replace code (char mixed with numbers) with "_code_"
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(code_regex, ' _code_ ', x,perl=TRUE)))
# replace other numbers with "_number_"
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(number_regex, ' _number_ ', x,perl=TRUE)))
# replace exclamation mark with "_exmark_"
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(exmark_regex, ' _exmark_ ', x,perl=TRUE)))
# replace "'" with empty char
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(char_to_remove2, '', x,perl=TRUE)))
# replace other punctuation with white space
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(char_to_remove, ' ', x,perl=TRUE)))
# remove continues white spaces
temp_data <- unlist(lapply(temp_data, FUN=function(x) gsub(' +', ' ', x,perl=TRUE)))
# to lower
temp_data <- unlist(lapply(temp_data, FUN=function(x) tolower(x)))
# split text to words
temp_data_word <- sapply(temp_data, FUN=function(x) strsplit(x, "\\s+"), USE.NAMES=F)
#print(temp_data_word)

# do stemming
library(SnowballC)
stem_data_word <- sapply(temp_data_word, FUN=function(x) wordStem(x), USE.NAMES=F)
#print(stem_data_word)
# filter out text that is too short and put words back to text
filter = sapply(stem_data_word, FUN=function(x) !(length(x) == 1 & x[1] == "") , USE.NAMES=F)
stem_data <- sapply(stem_data_word,FUN=function(x) paste(x,collapse=' '),USE.NAMES=F)
#print(filter)
#print(stem_data)
# apply the filter
filtered_labels = raw_data$label[filter]
filtered_data = stem_data[filter]
cleaned_data = data.frame(label=filtered_labels)
cleaned_data$sms = filtered_data
#cleaned_data = cbind(sms=filtered_data, label=filtered_labels)
#print(class(cleaned_data$sms))
n_data = nrow(cleaned_data)
all_index = c(1:n_data)

# work flow for classification
# text -> corpus -> doc_term_matrix -> container -> train model -> test model
corp <- VCorpus(VectorSource(cleaned_data$sms))
doc_matrix <- DocumentTermMatrix(corp, control = list(bound=list(global=c(2,Inf)), wordLengths=c(2,Inf), 
                                                      removeNumbers=F, removePunctuation=F, stemWords=F,
                                                      removeStopwords=F, tolower=F,
                                                      removeSparseTerms=0.995, weighting=weightTf))

#doc_matrix <- create_matrix(cleaned_data$sms, minDocFreq=2, minWordLength=2, 
#                            removeNumbers=F, removePunctuation=F, stemWords=F,
#                            removeStopwords=F, 
#                            removeSparseTerms=0.995, weighting=weightTf)

# create k-fold evaluation index
folds <- createFolds(cleaned_data$label, k = 5, list = TRUE, returnTrain = FALSE)
for (i in c(1:5)){
  test_index <- folds[i][[1]]
  train_index <- all_index[sapply(all_index, FUN=function(x) !(x %in% test_index) , USE.NAMES=F)]
  doc_container <- create_container(doc_matrix, cleaned_data$label,
                                    trainSize=train_index, testSize=test_index,
                                    virgin=F)
  print("debug")
  SVM <- train_model(doc_container, "SVM")
  RF <- train_model(doc_container, "RF")
  NB <- train_model()
  print("debug")
  print(SVM)
  #print(classify_model(doc_container, SVM))
  SVM_result = classify_model(doc_container, SVM)
  analytics <- create_analytics(doc_container, cbind(SVM_result))
  summary(analytics)
  incorrect = test_index[sapply(analytics@document_summary$CONSENSUS_INCORRECT, FUN=function(x) x==1)]
  print(cleaned_data$sms[incorrect])
}



